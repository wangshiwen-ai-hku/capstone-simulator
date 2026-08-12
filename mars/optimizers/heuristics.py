"""Deterministic built-in optimizers for the shared scheduling problem."""

from __future__ import annotations

from collections import defaultdict
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
from .materialization import (
    build_assignment,
    build_node_reservation,
    materialize_candidate,
)


class HeuristicOptimizer:
    """Deterministic solver that follows the policy carried by the problem."""

    optimizer_id = "heuristic"
    optimizer_version = "1"
    solve_work_unit = "ready_task"

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

    registry = OptimizerRegistry()
    registry.register(HeuristicOptimizer())
    registry.register(BinaryOffloadOptimizer())
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
