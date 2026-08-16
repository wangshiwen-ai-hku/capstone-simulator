"""Deterministic built-in optimizers for the shared scheduling problem."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import product
import math
from time import perf_counter

from ..domain.execution import Assignment, ExecutionMode
from ..domain.transfer import TransferReservation
from .base import (
    CandidateMaterializationError,
    CandidateEstimate,
    OptimizerRegistry,
    PlannedResourceReservation,
    SchedulingPlan,
    SchedulingProblem,
    SolveStatus,
)
from .evaluation import candidate_proxy_key
from .formulation import FormulationError, FormulationSpec, PreparedSolve
from .formulations.one_hot import (
    ONE_HOT_PLACEMENT_SPEC,
    OneHotPlacementFormulation,
    OneHotPlacementModel,
)
from .materialization import (
    build_assignment,
    build_node_reservation,
    materialize_candidate,
)


class HeuristicOptimizer:
    """Deterministic solver that follows the policy carried by the problem."""

    optimizer_id = "heuristic"
    # The legacy greedy algorithm is unchanged; formulation support is an
    # additive interface, so retain its public version for compatibility.
    optimizer_version = "1"
    optimizer_config_digest = "deterministic-greedy.v1"
    solve_work_unit = "ready_task"
    supported_formulation_ids = frozenset({"one_hot_placement"})

    @staticmethod
    def supports_formulation(spec: FormulationSpec) -> bool:
        return spec == ONE_HOT_PLACEMENT_SPEC

    def solve(self, problem: SchedulingProblem) -> SchedulingPlan:
        solve_started = perf_counter()
        link_available = {
            link_id: max(problem.epoch.now_ms, available)
            for link_id, available in problem.link_available_ms.items()
        }
        assignments: list[Assignment] = []
        node_reservations: list[PlannedResourceReservation] = []
        reservations_by_node: dict[
            str, list[PlannedResourceReservation]
        ] = defaultdict(list)
        for reservation in problem.existing_node_reservations:
            reservations_by_node[reservation.node_id].append(
                reservation
            )
        transfer_reservations: list[TransferReservation] = []

        ordered_tasks = sorted(
            problem.epoch.ready_tasks,
            key=lambda item: _task_order(problem, item.task_id),
        )
        iteration_limit = problem.solve_limits.max_iterations
        solved_tasks = (
            ordered_tasks[:iteration_limit]
            if iteration_limit
            else ordered_tasks
        )
        deferred_tasks = list(
            ordered_tasks[len(solved_tasks):]
        )
        time_limit_reached = False

        for index, task in enumerate(solved_tasks):
            if (
                (perf_counter() - solve_started) * 1000
                >= problem.solve_limits.solve_budget_ms
            ):
                deferred_tasks = [
                    *solved_tasks[index:],
                    *deferred_tasks,
                ]
                time_limit_reached = True
                break
            materialized = []
            for candidate in problem.candidates[task.task_id]:
                if not candidate.feasible:
                    continue
                try:
                    materialized.append(
                        materialize_candidate(
                            problem,
                            candidate,
                            reservations_by_node,
                            link_available,
                        )
                    )
                except CandidateMaterializationError:
                    continue
            if not materialized:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        target_node_id="",
                        execution_mode=ExecutionMode.DROP,
                        estimated_start_ms=max(
                            problem.epoch.now_ms,
                            task.arrival_time_ms,
                        ),
                        estimated_finish_ms=max(
                            problem.epoch.now_ms,
                            task.arrival_time_ms,
                        ),
                        compute_ms=0.0,
                        communication_ms=0.0,
                        energy_j=0.0,
                        reason=(
                            "no feasible node under declarative placement "
                            "and link constraints"
                        ),
                        optimizer_id=self.optimizer_id,
                        epoch_id=problem.epoch.epoch_id,
                        output_size_mb=0.0,
                        success_probability=0.0,
                    )
                )
                continue

            chosen, reservations, next_links, compute_start = self._choose(
                problem,
                task.task_id,
                materialized,
            )
            assignment = build_assignment(
                problem,
                chosen,
                reservations,
                optimizer_id=self.optimizer_id,
                reason=_reason(problem.policy.policy_id),
            )
            assignments.append(assignment)
            node_reservations.append(
                resource_reservation := build_node_reservation(
                    problem,
                    chosen,
                    compute_start_ms=compute_start,
                    reservation_id=(
                        f"plan:{problem.epoch.epoch_id}:"
                        f"{task.task_id}:{chosen.node_id}"
                    ),
                )
            )
            reservations_by_node[chosen.node_id].append(
                resource_reservation
            )
            transfer_reservations.extend(reservations)
            link_available.update(next_links)

        deferred_task_ids = tuple(
            task.task_id for task in deferred_tasks
        )
        solve_elapsed_ms = (perf_counter() - solve_started) * 1000
        return SchedulingPlan(
            problem_id=problem.problem_id,
            snapshot_id=problem.snapshot.snapshot_id,
            policy_id=problem.policy.policy_id,
            policy_version=problem.policy.version,
            epoch_id=problem.epoch.epoch_id,
            optimizer_id=self.optimizer_id,
            optimizer_version=self.optimizer_version,
            solve_status=(
                SolveStatus.TIME_LIMIT
                if time_limit_reached
                else SolveStatus.ITERATION_LIMIT
                if deferred_task_ids
                else SolveStatus.FEASIBLE
            ),
            solve_elapsed_ms=solve_elapsed_ms,
            iteration_count=len(assignments),
            termination_reason=(
                "solve_budget_reached"
                if time_limit_reached
                else "max_iterations_reached"
                if deferred_task_ids
                else "deterministic_heuristic_complete"
            ),
            assignments=tuple(assignments),
            node_reservations=tuple(node_reservations),
            transfer_reservations=tuple(transfer_reservations),
            deferred_task_ids=deferred_task_ids,
            diagnostics={
                "task_count": len(problem.epoch.ready_tasks),
                "scheduled_count": sum(
                    item.execution_mode is not ExecutionMode.DROP
                    for item in assignments
                ),
                "solve_budget_ms": problem.solve_budget_ms,
                "policy_id": problem.policy.policy_id,
            },
        )

    def solve_formulated(self, prepared: PreparedSolve) -> SchedulingPlan:
        """Return a feasible decision in the selected formulation domain.

        This is primarily the recovery path for a bounded solver.  It ranks
        candidates with the shared policy proxy, but every returned schedule is
        decoded by the formulation and later receives authoritative validation.
        """

        if not isinstance(prepared.model, OneHotPlacementModel) or not isinstance(
            prepared.formulation,
            OneHotPlacementFormulation,
        ):
            raise TypeError(
                "heuristic currently supports only one-hot placement"
            )
        if prepared.request.optimizer_id != self.optimizer_id:
            raise ValueError(
                "prepared solve optimizer identity does not match heuristic"
            )
        if (
            prepared.request.optimizer_version != self.optimizer_version
            or prepared.request.optimizer_config_digest
            != self.optimizer_config_digest
        ):
            raise ValueError(
                "prepared solve optimizer version/config does not match "
                "heuristic"
            )
        if not self.supports_formulation(
            prepared.request.formulation_spec
        ):
            raise ValueError(
                "heuristic does not support the selected formulation "
                "version or materializer contract"
            )
        problem = prepared.problem
        model = prepared.model
        formulation = prepared.formulation
        solve_started = (
            perf_counter() - prepared.compilation_elapsed_ms / 1000.0
        )
        ranked_options = tuple(
            tuple(
                sorted(
                    options,
                    key=lambda candidate: (
                        *candidate_proxy_key(
                            problem,
                            task_id,
                            candidate,
                        ),
                        candidate.node_id,
                    ),
                )
            )
            for task_id, options in zip(
                model.ordered_task_ids,
                model.candidate_options,
                strict=True,
            )
        )
        evaluated = 0
        for selection in product(*ranked_options):
            if (
                problem.solve_limits.max_iterations
                and evaluated >= problem.solve_limits.max_iterations
            ):
                raise RuntimeError(
                    "heuristic iteration limit reached before a one-hot "
                    "incumbent"
                )
            if prepared.time_limit_reached(
                now_monotonic=perf_counter(),
                solve_started_monotonic=solve_started,
            ):
                raise TimeoutError(
                    "heuristic solve budget expired before a one-hot incumbent"
                )
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
                    raise TimeoutError(
                        "heuristic solve budget expired before a one-hot "
                        "incumbent"
                    )
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
                    raise TimeoutError(
                        "heuristic solve budget expired before a one-hot "
                        "incumbent"
                    )
                continue
            spec = prepared.request.formulation_spec
            draft = SchedulingPlan(
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
                assignments=materialized.assignments,
                node_reservations=materialized.node_reservations,
                transfer_reservations=(
                    materialized.transfer_reservations
                ),
            )
            evaluation = formulation.evaluate(
                problem,
                model,
                draft,
            )
            if prepared.time_limit_reached(
                now_monotonic=perf_counter(),
                solve_started_monotonic=solve_started,
            ):
                raise TimeoutError(
                    "heuristic solve budget expired before a one-hot incumbent"
                )
            if evaluation.has_hard_violation or any(
                not math.isfinite(value)
                for value in evaluation.objective_key
            ):
                continue
            elapsed_ms = (perf_counter() - solve_started) * 1000.0
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
                solve_elapsed_ms=elapsed_ms,
                iteration_count=evaluated,
                termination_reason="greedy_one_hot_incumbent_found",
                assignments=tuple(
                    replace(
                        assignment,
                        reason="greedy feasible one-hot placement",
                    )
                    for assignment in materialized.assignments
                ),
                node_reservations=materialized.node_reservations,
                transfer_reservations=(
                    materialized.transfer_reservations
                ),
                diagnostics={
                    "total_combinations": model.total_decisions,
                    "formulation_exhausted": False,
                },
            )
        raise FormulationError(
            "heuristic found no feasible one-hot placement assignment"
        )

    def _choose(
        self,
        problem: SchedulingProblem,
        task_id: str,
        materialized: list[
            tuple[
                CandidateEstimate,
                tuple[TransferReservation, ...],
                dict[str, float],
                float,
            ]
        ],
    ) -> tuple[
        CandidateEstimate,
        tuple[TransferReservation, ...],
        dict[str, float],
        float,
    ]:
        return min(
            materialized,
            key=lambda item: (
                *candidate_proxy_key(
                    problem,
                    task_id,
                    item[0],
                ),
                item[0].node_id,
            ),
        )


def built_in_registry() -> OptimizerRegistry:
    from .binary_offload import BinaryOffloadOptimizer
    from .deferred_offload import DeferredOffloadOptimizer

    registry = OptimizerRegistry()
    registry.register(HeuristicOptimizer())
    registry.register(BinaryOffloadOptimizer())
    registry.register(DeferredOffloadOptimizer())
    return registry


def _task_order(
    problem: SchedulingProblem,
    task_id: str,
) -> tuple[float, int, str]:
    task = problem.task_by_id[task_id]
    if problem.critical_tail_ms:
        slack = (
            task.deadline_time_ms
            - max(problem.epoch.now_ms, task.arrival_time_ms)
            - problem.critical_tail_ms.get(task_id, 0.0)
        )
        return (slack, -task.priority, task.task_id)
    return (task.arrival_time_ms, -task.priority, task.task_id)


def _reason(policy_id: str) -> str:
    return {
        "local_first": "local-first policy under declarative constraints",
        "edge_first": "edge-first policy under declarative constraints",
        "rule_based": "declarative placement rule policy",
        "dag_deadline": (
            "DAG deadline/critical-tail/data-locality batch policy"
        ),
        "greedy_cost": "minimum estimated finish time and energy policy",
    }.get(policy_id, f"policy {policy_id}")
