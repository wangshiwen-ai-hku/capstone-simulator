"""Bounded exhaustive optimizer for a compiled placement formulation."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
import math
from time import perf_counter

from .base import (
    CandidateMaterializationError,
    SchedulingPlan,
    SchedulingProblem,
    SolveStatus,
)
from .formulation import (
    FormulationSpec,
    PreparedSolve,
    build_solve_request,
    compile_solve_request,
    formulation_failure_status,
    prepare_solve,
)
from .formulations.one_hot import (
    ONE_HOT_PLACEMENT_SPEC,
    OneHotPlacementFormulation,
    OneHotPlacementModel,
)
from .policy import binary_offload_policy
from .state import (
    OptimizerSolveState,
    SolveTraceContext,
    SolveTracePhase,
)


class BinaryOffloadOptimizer:
    """Enumerate every decision in a one-hot placement model.

    ``binary_offload`` remains the compatibility-facing optimizer ID.  Its
    search strategy is bounded exhaustive enumeration; the independently
    selectable ``one_hot_placement`` formulation owns the decision domain,
    policy encoding, and SchedulingPlan materialization.
    """

    optimizer_id = "binary_offload"
    optimizer_version = "4"
    optimizer_config_digest = "bounded-exhaustive.v1"
    solve_work_unit = "placement_combination"
    supported_formulation_ids = frozenset({"one_hot_placement"})
    default_formulation_id = "one_hot_placement"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 2.0,
        formulation: OneHotPlacementFormulation | None = None,
    ) -> None:
        # Kept as a compatibility preset.  Explicit Policy selection remains
        # independent of both optimizer and formulation in the core API.
        self.default_policy = binary_offload_policy(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        self.default_formulation = (
            formulation or OneHotPlacementFormulation()
        )
        if (
            self.default_formulation.spec.formulation_id
            not in self.supported_formulation_ids
        ):
            raise ValueError(
                "binary_offload received an unsupported default formulation"
            )

    def solve(self, problem: SchedulingProblem) -> SchedulingPlan:
        """Solve with the compatibility default formulation, without state."""

        return self.solve_formulated(
            prepare_solve(problem, self, self.default_formulation)
        )

    @staticmethod
    def supports_formulation(spec: FormulationSpec) -> bool:
        """Return whether this backend implements the exact spec contract."""

        return spec == ONE_HOT_PLACEMENT_SPEC

    def solve_with_state(
        self,
        problem: SchedulingProblem,
        state: OptimizerSolveState,
        *,
        context: SolveTraceContext | None = None,
    ) -> SchedulingPlan:
        """Compatibility stateful entry point using the default formulation."""

        solve_started = perf_counter()
        request = build_solve_request(
            problem,
            self,
            self.default_formulation,
        )
        if context is None:
            context = state.begin(
                problem,
                optimizer_id=self.optimizer_id,
                optimizer_version=self.optimizer_version,
                work_unit=self.solve_work_unit,
                solve_request=request,
            )
        try:
            prepared = compile_solve_request(
                request,
                self.default_formulation,
            )
        except Exception as exc:
            state.record(
                context,
                SolveTracePhase.FAILED,
                elapsed_ms=self._elapsed_ms(solve_started),
                solve_status=formulation_failure_status(exc),
                termination_reason=f"{type(exc).__name__}: {exc}",
                details={
                    "ready_task_count": len(problem.epoch.ready_tasks)
                },
            )
            raise
        return self.solve_formulated_with_state(
            prepared,
            state,
            context=context,
            _solve_started=solve_started,
        )

    def solve_formulated(self, prepared: PreparedSolve) -> SchedulingPlan:
        """Solve one independently prepared formulation request."""

        state = OptimizerSolveState(
            session_id=f"standalone:{prepared.problem.problem_id}"
        )
        return self.solve_formulated_with_state(prepared, state)

    def solve_formulated_with_state(
        self,
        prepared: PreparedSolve,
        state: OptimizerSolveState,
        *,
        context: SolveTraceContext | None = None,
        _solve_started: float | None = None,
    ) -> SchedulingPlan:
        """Solve while appending an auditable caller-owned trace."""

        self._validate_prepared(prepared)
        problem = prepared.problem
        solve_started = (
            perf_counter() - prepared.compilation_elapsed_ms / 1000.0
            if _solve_started is None
            else _solve_started
        )
        if context is None:
            context = state.begin(
                problem,
                optimizer_id=self.optimizer_id,
                optimizer_version=self.optimizer_version,
                work_unit=self.solve_work_unit,
                solve_request=prepared.request,
            )
        elif (
            context.problem_id != problem.problem_id
            or context.optimizer_id != self.optimizer_id
            or context.solve_request_id
            != prepared.request.solve_request_id
        ):
            raise ValueError(
                "solve trace context does not match the prepared solve"
            )
        try:
            return self._solve(
                prepared,
                solve_started,
                state,
                context,
            )
        except Exception as exc:
            if not any(
                entry.context.solve_id == context.solve_id
                and entry.phase is SolveTracePhase.FAILED
                for entry in state.entries
            ):
                state.record(
                    context,
                    SolveTracePhase.FAILED,
                    elapsed_ms=self._elapsed_ms(solve_started),
                    solve_status=(
                        SolveStatus.INFEASIBLE
                        if isinstance(exc, ValueError)
                        and "no feasible" in str(exc).lower()
                        else SolveStatus.ERROR
                    ),
                    termination_reason=f"{type(exc).__name__}: {exc}",
                    details={
                        "ready_task_count": len(
                            problem.epoch.ready_tasks
                        )
                    },
                )
            raise

    def _solve(
        self,
        prepared: PreparedSolve,
        solve_started: float,
        state: OptimizerSolveState,
        context: SolveTraceContext,
    ) -> SchedulingPlan:
        problem = prepared.problem
        formulation = prepared.formulation
        model = prepared.model
        assert isinstance(formulation, OneHotPlacementFormulation)
        assert isinstance(model, OneHotPlacementModel)
        total_combinations = model.total_decisions
        best_key: tuple[float, ...] | None = None
        best_tie_break: tuple[str, ...] | None = None
        best_plan: SchedulingPlan | None = None
        evaluated = 0
        bounded_status: SolveStatus | None = None

        for selection in product(*model.candidate_options):
            if (
                problem.solve_limits.max_iterations
                and evaluated >= problem.solve_limits.max_iterations
            ):
                bounded_status = SolveStatus.ITERATION_LIMIT
                break
            if prepared.time_limit_reached(
                now_monotonic=perf_counter(),
                solve_started_monotonic=solve_started,
            ):
                bounded_status = SolveStatus.TIME_LIMIT
                break

            evaluated += 1
            decision = formulation.decision(model, selection)
            if not formulation.is_decision_feasible(
                problem,
                model,
                decision,
            ):
                if prepared.time_limit_reached(
                    now_monotonic=perf_counter(),
                    solve_started_monotonic=solve_started,
                ):
                    bounded_status = SolveStatus.TIME_LIMIT
                    break
                continue
            try:
                materialized = formulation.materialize(
                    problem,
                    model,
                    decision,
                    optimizer_id=self.optimizer_id,
                )
            except CandidateMaterializationError:
                if prepared.time_limit_reached(
                    now_monotonic=perf_counter(),
                    solve_started_monotonic=solve_started,
                ):
                    bounded_status = SolveStatus.TIME_LIMIT
                    break
                continue
            draft = self._draft_plan(
                prepared,
                materialized,
                iteration_count=evaluated,
            )
            evaluation = formulation.evaluate(problem, model, draft)
            if prepared.time_limit_reached(
                now_monotonic=perf_counter(),
                solve_started_monotonic=solve_started,
            ):
                bounded_status = SolveStatus.TIME_LIMIT
                break
            if evaluation.has_hard_violation:
                continue
            evaluated_key = evaluation.objective_key
            if any(not math.isfinite(value) for value in evaluated_key):
                continue
            tie_break = tuple(item.node_id for item in selection)
            if (
                best_key is None
                or evaluated_key < best_key
                or (
                    evaluated_key == best_key
                    and tie_break < (best_tie_break or ())
                )
            ):
                best_key = evaluated_key
                best_tie_break = tie_break
                best_plan = replace(
                    draft,
                    objective_value=(
                        evaluated_key[0] if evaluated_key else 0.0
                    ),
                    objective_key=evaluated_key,
                    objective_evaluations=(
                        evaluation.objective_evaluations
                    ),
                    constraint_evaluations=(
                        evaluation.constraint_evaluations
                    ),
                )
                state.record(
                    context,
                    SolveTracePhase.INCUMBENT,
                    iteration=evaluated,
                    elapsed_ms=self._elapsed_ms(solve_started),
                    has_incumbent=True,
                    evaluated_work_units=evaluated,
                    total_work_units=total_combinations,
                    objective_key=evaluated_key,
                    objective_components={
                        item.objective_id: item.raw_value
                        for item in evaluation.objective_evaluations
                    },
                    selected_targets={
                        item.task_id: item.node_id for item in selection
                    },
                    details={
                        "ready_task_count": len(model.ordered_task_ids),
                    },
                )

        if best_plan is None or best_key is None:
            if bounded_status is SolveStatus.TIME_LIMIT:
                self._record_failure(
                    state,
                    context,
                    solve_started,
                    status=SolveStatus.TIME_LIMIT,
                    reason="solve_budget_reached_without_incumbent",
                    evaluated=evaluated,
                    total=total_combinations,
                    ready_task_count=len(model.ordered_task_ids),
                )
                raise TimeoutError(
                    "binary offload solve budget expired before an incumbent"
                )
            if bounded_status is SolveStatus.ITERATION_LIMIT:
                self._record_failure(
                    state,
                    context,
                    solve_started,
                    status=SolveStatus.ITERATION_LIMIT,
                    reason="max_iterations_reached_without_incumbent",
                    evaluated=evaluated,
                    total=total_combinations,
                    ready_task_count=len(model.ordered_task_ids),
                )
                raise RuntimeError(
                    "binary offload iteration limit reached before an incumbent"
                )
            self._record_failure(
                state,
                context,
                solve_started,
                status=SolveStatus.INFEASIBLE,
                reason="formulation_search_found_no_feasible_assignment",
                evaluated=evaluated,
                total=total_combinations,
                ready_task_count=len(model.ordered_task_ids),
            )
            raise ValueError(
                "binary offload problem has no feasible complete assignment"
            )

        exhaustive = bounded_status is None
        solve_status = (
            SolveStatus.OPTIMAL if exhaustive else bounded_status
        )
        assert solve_status is not None
        termination_reason = (
            "exhaustive_one_hot_search_complete"
            if exhaustive
            else "solve_budget_reached_with_incumbent"
            if solve_status is SolveStatus.TIME_LIMIT
            else "max_iterations_reached_with_incumbent"
        )
        assignment_reason = (
            "optimal within one-hot placement formulation"
            if exhaustive
            else "best one-hot placement incumbent within solve limits"
        )
        elapsed_ms = self._elapsed_ms(solve_started)
        final_plan = replace(
            best_plan,
            solve_status=solve_status,
            solve_elapsed_ms=elapsed_ms,
            iteration_count=evaluated,
            termination_reason=termination_reason,
            assignments=tuple(
                replace(item, reason=assignment_reason)
                for item in best_plan.assignments
            ),
            diagnostics={
                "total_combinations": total_combinations,
                "formulation_exhausted": exhaustive,
            },
        )
        state.record(
            context,
            SolveTracePhase.COMPLETED,
            iteration=evaluated,
            elapsed_ms=elapsed_ms,
            solve_status=solve_status,
            termination_reason=termination_reason,
            has_incumbent=True,
            evaluated_work_units=evaluated,
            total_work_units=total_combinations,
            objective_key=best_key,
            objective_components={
                item.objective_id: item.raw_value
                for item in final_plan.objective_evaluations
            },
            selected_targets={
                item.task_id: item.target_node_id
                for item in final_plan.assignments
            },
            details={
                "ready_task_count": len(model.ordered_task_ids),
                "communication_time_ms": sum(
                    item.communication_ms
                    for item in final_plan.assignments
                ),
                "formulation_exhausted": exhaustive,
            },
        )
        return final_plan

    def _draft_plan(
        self,
        prepared: PreparedSolve,
        materialized,
        *,
        iteration_count: int,
    ) -> SchedulingPlan:
        problem = prepared.problem
        spec = prepared.request.formulation_spec
        return SchedulingPlan(
            problem_id=problem.problem_id,
            snapshot_id=problem.snapshot.snapshot_id,
            policy_id=problem.policy.policy_id,
            policy_version=problem.policy.version,
            epoch_id=problem.epoch.epoch_id,
            optimizer_id=self.optimizer_id,
            optimizer_version=self.optimizer_version,
            solve_request_id=prepared.request.solve_request_id,
            metric_contract_id=problem.metric_contract_id,
            formulation_id=spec.formulation_id,
            formulation_version=spec.formulation_version,
            formulation_digest=spec.formulation_digest,
            solve_status=SolveStatus.FEASIBLE,
            iteration_count=iteration_count,
            assignments=materialized.assignments,
            node_reservations=materialized.node_reservations,
            transfer_reservations=materialized.transfer_reservations,
        )

    def _record_failure(
        self,
        state: OptimizerSolveState,
        context: SolveTraceContext,
        solve_started: float,
        *,
        status: SolveStatus,
        reason: str,
        evaluated: int,
        total: int,
        ready_task_count: int,
    ) -> None:
        state.record(
            context,
            SolveTracePhase.FAILED,
            iteration=evaluated,
            elapsed_ms=self._elapsed_ms(solve_started),
            solve_status=status,
            termination_reason=reason,
            has_incumbent=False,
            evaluated_work_units=evaluated,
            total_work_units=total,
            details={"ready_task_count": ready_task_count},
        )

    @staticmethod
    def _validate_prepared(prepared: PreparedSolve) -> None:
        if not isinstance(prepared.model, OneHotPlacementModel):
            raise TypeError(
                "binary_offload requires OneHotPlacementModel"
            )
        if not isinstance(
            prepared.formulation,
            OneHotPlacementFormulation,
        ):
            raise TypeError(
                "binary_offload requires OneHotPlacementFormulation"
            )
        if (
            not BinaryOffloadOptimizer.supports_formulation(
                prepared.request.formulation_spec
            )
        ):
            raise ValueError(
                "binary_offload does not support the selected formulation "
                "version or materializer contract"
            )
        if prepared.request.optimizer_id != BinaryOffloadOptimizer.optimizer_id:
            raise ValueError(
                "prepared solve optimizer identity does not match binary_offload"
            )
        if (
            prepared.request.optimizer_version
            != BinaryOffloadOptimizer.optimizer_version
            or prepared.request.optimizer_config_digest
            != BinaryOffloadOptimizer.optimizer_config_digest
        ):
            raise ValueError(
                "prepared solve optimizer version/config does not match "
                "binary_offload"
            )

    @staticmethod
    def _elapsed_ms(solve_started: float) -> float:
        return (perf_counter() - solve_started) * 1000.0


__all__ = ["BinaryOffloadOptimizer"]
