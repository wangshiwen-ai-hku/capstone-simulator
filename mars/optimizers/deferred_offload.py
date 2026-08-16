"""CP-SAT search backend for the assign-or-defer formulation."""

from __future__ import annotations

from dataclasses import replace
import math
from time import perf_counter

from .base import (
    SchedulingPlan,
    SolveStatus,
    background_resource_demand,
)
from .binary_offload import BinaryOffloadOptimizer
from .formulation import FormulationSpec, PreparedSolve
from .formulations.assign_or_defer import (
    ASSIGN_OR_DEFER_SPEC,
    AssignOrDeferFormulation,
    AssignOrDeferModel,
)
from .policy import ObjectiveMetric, deferred_offload_policy
from .state import OptimizerSolveState, SolveTraceContext, SolveTracePhase

try:
    from ortools.sat.python import cp_model as _cp_model
except ImportError:  # pragma: no cover - environment diagnosis
    _cp_model = None


class DeferredOffloadOptimizer(BinaryOffloadOptimizer):
    """Use CP-SAT to select one placement or defer each ready task."""

    optimizer_id = "deferred_offload"
    optimizer_version = "1"
    solve_work_unit = "cp_sat_branch"
    supported_formulation_ids = frozenset({"assign_or_defer"})
    default_formulation_id = "assign_or_defer"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 2.0,
        delta: float = 1.0,
        formulation: AssignOrDeferFormulation | None = None,
    ) -> None:
        weights = (alpha, beta, gamma, delta)
        if not all(math.isfinite(value) and value >= 0.0 for value in weights):
            raise ValueError("deferred-offload weights must be finite and non-negative")
        if delta <= 0.0 or not any(value > 0.0 for value in weights):
            raise ValueError("deferred-offload requires a positive deferral weight")
        self.default_policy = deferred_offload_policy(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        self.default_formulation = formulation or AssignOrDeferFormulation()
        self.optimizer_config_digest = (
            f"cp-sat.v1:a={alpha}:b={beta}:g={gamma}:d={delta}"
        )

    @staticmethod
    def supports_formulation(spec: FormulationSpec) -> bool:
        return spec == ASSIGN_OR_DEFER_SPEC

    def _solve(
        self,
        prepared: PreparedSolve,
        solve_started: float,
        state: OptimizerSolveState,
        context: SolveTraceContext,
    ) -> SchedulingPlan:
        if _cp_model is None:  # pragma: no cover - environment diagnosis
            raise RuntimeError(
                "deferred_offload requires the optional OR-Tools dependency"
            )
        cp_model = _cp_model

        problem = prepared.problem
        formulation = prepared.formulation
        model = prepared.model
        assert isinstance(formulation, AssignOrDeferFormulation)
        assert isinstance(model, AssignOrDeferModel)

        cp = cp_model.CpModel()
        assignment_vars = {
            (candidate.task_id, candidate.node_id): cp.new_bool_var(
                f"assign__{candidate.task_id}__{candidate.node_id}"
            )
            for options in model.candidate_options
            for candidate in options
        }
        deferred_vars = {
            task_id: cp.new_bool_var(f"defer__{task_id}")
            for task_id in model.ordered_task_ids
        }
        for task_id, options in zip(
            model.ordered_task_ids,
            model.candidate_options,
            strict=True,
        ):
            cp.add(
                sum(
                    assignment_vars[candidate.task_id, candidate.node_id]
                    for candidate in options
                )
                + deferred_vars[task_id]
                == 1
            )

        resource_scale = 1_000
        utilization_scale = 10_000
        objective_scale = 1_000_000
        u_max = cp.new_int_var(0, utilization_scale, "maximum_utilization")
        now_ms = problem.epoch.now_ms
        for node in problem.node_specs:
            candidates = tuple(
                candidate
                for options in model.candidate_options
                for candidate in options
                if candidate.node_id == node.node_id
            )
            active = tuple(
                reservation
                for reservation in problem.existing_node_reservations
                if reservation.node_id == node.node_id
                and reservation.start_ms <= now_ms < reservation.finish_ms
            )
            background = background_resource_demand(problem, node.node_id)
            resources = (
                (node.cpu_capacity, background.cpu_units, "cpu_units"),
                (node.gpu_capacity, background.gpu_units, "gpu_units"),
                (node.memory_gb, background.memory_gb, "memory_gb"),
            )
            for capacity, background_units, demand_field in resources:
                capacity_units = round(capacity * resource_scale)
                if capacity_units <= 0:
                    continue
                occupied_units = round(
                    (
                        background_units
                        + sum(
                            getattr(reservation.demand, demand_field)
                            for reservation in active
                        )
                    )
                    * resource_scale
                )
                selected_units = sum(
                    round(getattr(candidate.resource_demand, demand_field) * resource_scale)
                    * assignment_vars[candidate.task_id, candidate.node_id]
                    for candidate in candidates
                )
                cp.add(occupied_units + selected_units <= capacity_units)
                cp.add(
                    u_max * capacity_units
                    >= (occupied_units + selected_units) * utilization_scale
                )
            cp.add(
                len(active)
                + sum(
                    assignment_vars[candidate.task_id, candidate.node_id]
                    for candidate in candidates
                )
                <= node.max_concurrency
            )

        weights = {
            objective.metric: objective.weight
            for objective in problem.policy.objectives
        }
        total_priority = max(
            1.0,
            sum(max(0.0, float(task.priority)) for task in problem.epoch.ready_tasks),
        )
        total_budget = max(
            1.0,
            sum(max(1.0, task.spec.latency_budget_ms) for task in problem.epoch.ready_tasks),
        )
        success_weight = weights.get(ObjectiveMetric.EXPECTED_WEIGHTED_SUCCESS_RATIO, 0.0)
        communication_weight = weights.get(ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO, 0.0)
        utilization_weight = weights.get(ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION, 0.0)
        deferred_weight = weights.get(ObjectiveMetric.DEFERRED_PRIORITY_PENALTY, 0.0)
        task_by_id = problem.task_by_id
        objective_terms = []
        for options in model.candidate_options:
            for candidate in options:
                variable = assignment_vars[candidate.task_id, candidate.node_id]
                task = task_by_id[candidate.task_id]
                coefficient = (
                    -success_weight
                    * max(0.0, float(task.priority))
                    * candidate.success_probability
                    / total_priority
                    + communication_weight
                    * candidate.communication_ms
                    / total_budget
                )
                objective_terms.append(round(coefficient * objective_scale) * variable)
        objective_terms.extend(
            round(
                deferred_weight
                * (2 ** task_by_id[task_id].priority)
                * objective_scale
            )
            * deferred_vars[task_id]
            for task_id in model.ordered_task_ids
        )
        objective_terms.append(
            round(utilization_weight * objective_scale / utilization_scale) * u_max
        )
        cp.minimize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(
            0.001,
            problem.solve_limits.solve_budget_ms / 1_000.0,
        )
        solver.parameters.num_search_workers = 1
        status = solver.solve(cp)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError(
                "deferred offload found no feasible assign-or-defer decision: "
                f"{solver.status_name(status)}"
            )

        selections = tuple(
            next(
                (
                    candidate
                    for candidate in options
                    if solver.value(
                        assignment_vars[candidate.task_id, candidate.node_id]
                    )
                ),
                None,
            )
            for options in model.candidate_options
        )
        decision = formulation.decision(model, selections)
        materialized = formulation.materialize(
            problem,
            model,
            decision,
            optimizer_id=self.optimizer_id,
        )
        draft = self._draft_plan(
            prepared,
            materialized,
            iteration_count=int(solver.num_branches),
        )
        evaluation = formulation.evaluate(problem, model, draft)
        solve_status = (
            SolveStatus.OPTIMAL
            if status == cp_model.OPTIMAL
            else SolveStatus.FEASIBLE
        )
        elapsed_ms = self._elapsed_ms(solve_started)
        final = replace(
            draft,
            solve_status=solve_status,
            solve_elapsed_ms=elapsed_ms,
            termination_reason=solver.status_name(status),
            objective_value=(evaluation.objective_key[0] if evaluation.objective_key else 0.0),
            objective_key=evaluation.objective_key,
            objective_evaluations=evaluation.objective_evaluations,
            constraint_evaluations=evaluation.constraint_evaluations,
            diagnostics={
                "cp_sat_objective": solver.objective_value / objective_scale,
                "deferred_task_count": len(materialized.deferred_task_ids),
            },
        )
        state.record(
            context,
            SolveTracePhase.COMPLETED,
            iteration=int(solver.num_branches),
            elapsed_ms=elapsed_ms,
            solve_status=solve_status,
            termination_reason=solver.status_name(status),
            has_incumbent=True,
            evaluated_work_units=int(solver.num_branches),
            total_work_units=model.total_decisions,
            objective_key=evaluation.objective_key,
            objective_components={
                item.objective_id: item.raw_value
                for item in evaluation.objective_evaluations
            },
            selected_targets={
                task_id: candidate.node_id if candidate is not None else "deferred"
                for task_id, candidate in zip(
                    model.ordered_task_ids,
                    selections,
                    strict=True,
                )
            },
            details={
                "ready_task_count": len(model.ordered_task_ids),
                "deferred_task_count": len(materialized.deferred_task_ids),
            },
        )
        return final

    def _validate_prepared(self, prepared: PreparedSolve) -> None:
        if not isinstance(prepared.model, AssignOrDeferModel):
            raise TypeError("deferred_offload requires AssignOrDeferModel")
        if not isinstance(prepared.formulation, AssignOrDeferFormulation):
            raise TypeError("deferred_offload requires AssignOrDeferFormulation")
        if not self.supports_formulation(prepared.request.formulation_spec):
            raise ValueError("deferred_offload does not support this formulation")
        if prepared.request.optimizer_id != self.optimizer_id:
            raise ValueError("prepared solve optimizer identity does not match")
        if (
            prepared.request.optimizer_version != self.optimizer_version
            or prepared.request.optimizer_config_digest != self.optimizer_config_digest
        ):
            raise ValueError("prepared solve optimizer configuration does not match")


__all__ = ["DeferredOffloadOptimizer"]
