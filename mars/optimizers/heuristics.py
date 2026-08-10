"""Deterministic built-in optimizers for the shared scheduling problem."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
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
    background_resource_demand,
    execution_mode_for_candidate,
)
from .evaluation import candidate_objective_key


class HeuristicOptimizer:
    """Deterministic solver that follows the policy carried by the problem."""

    optimizer_id = "heuristic"

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
                        _materialize_candidate(
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
            link_ids = tuple(
                dict.fromkeys(
                    link_id
                    for transfer in chosen.transfers
                    for link_id in transfer.path_link_ids
                )
            )
            assignment = Assignment(
                task_id=task.task_id,
                target_node_id=chosen.node_id,
                execution_mode=execution_mode_for_candidate(task, chosen),
                estimated_start_ms=chosen.start_ms,
                estimated_finish_ms=chosen.finish_ms,
                compute_ms=chosen.compute_ms,
                communication_ms=chosen.communication_ms,
                energy_j=chosen.energy_j,
                reason=_reason(problem.policy.policy_id),
                input_locations=chosen.input_locations,
                transfer_link_ids=link_ids,
                optimizer_id=self.optimizer_id,
                epoch_id=problem.epoch.epoch_id,
                output_size_mb=chosen.output_size_mb,
                success_probability=chosen.success_probability,
            )
            assignments.append(assignment)
            node_reservations.append(
                resource_reservation := PlannedResourceReservation(
                    reservation_id=(
                        f"plan:{problem.epoch.epoch_id}:"
                        f"{task.task_id}:{chosen.node_id}"
                    ),
                    epoch_id=problem.epoch.epoch_id,
                    task_id=task.task_id,
                    node_id=chosen.node_id,
                    start_ms=compute_start,
                    finish_ms=chosen.finish_ms,
                    demand=chosen.resource_demand,
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
            optimizer_version="1",
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
                *candidate_objective_key(
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


def _materialize_candidate(
    problem: SchedulingProblem,
    candidate: CandidateEstimate,
    reservations_by_node: dict[
        str, list[PlannedResourceReservation]
    ],
    link_available: dict[str, float],
) -> tuple[
    CandidateEstimate,
    tuple[TransferReservation, ...],
    dict[str, float],
    float,
]:
    cursor = max(problem.epoch.now_ms, candidate.ready_time_ms)
    next_links = dict(link_available)
    reservations: list[TransferReservation] = []
    for index, transfer in enumerate(candidate.transfers):
        if not transfer.path_link_ids or transfer.transfer_time_ms <= 0:
            continue
        transfer_start = max(
            cursor,
            *(
                next_links.get(link_id, problem.epoch.now_ms)
                for link_id in transfer.path_link_ids
            ),
        )
        transfer_finish = transfer_start + transfer.transfer_time_ms
        reservation = TransferReservation(
            reservation_id=(
                f"transfer:{problem.epoch.epoch_id}:{candidate.task_id}:"
                f"{index}:{candidate.node_id}"
            ),
            epoch_id=problem.epoch.epoch_id,
            task_id=candidate.task_id,
            transfer_id=transfer.transfer_id,
            path_link_ids=transfer.path_link_ids,
            start_ms=transfer_start,
            finish_ms=transfer_finish,
            size_mb=transfer.size_mb,
        )
        reservations.append(reservation)
        for link_id in transfer.path_link_ids:
            next_links[link_id] = transfer_finish
        cursor = transfer_finish

    compute_start = _earliest_resource_start(
        problem,
        candidate,
        cursor,
        reservations_by_node.get(candidate.node_id, []),
    )
    start_ms = (
        reservations[0].start_ms if reservations else compute_start
    )
    finish_ms = compute_start + candidate.compute_ms
    communication_ms = sum(
        item.finish_ms - item.start_ms for item in reservations
    )
    return (
        replace(
            candidate,
            start_ms=start_ms,
            finish_ms=finish_ms,
            communication_ms=communication_ms,
        ),
        tuple(reservations),
        next_links,
        compute_start,
    )


def _earliest_resource_start(
    problem: SchedulingProblem,
    candidate: CandidateEstimate,
    earliest_ms: float,
    existing: list[PlannedResourceReservation],
) -> float:
    """Find the first interval that satisfies concurrency and capacity."""

    node = problem.node_by_id[candidate.node_id]
    background = background_resource_demand(problem, candidate.node_id)
    if (
        background.cpu_units + candidate.resource_demand.cpu_units
        > node.cpu_capacity + 1e-9
        or background.gpu_units + candidate.resource_demand.gpu_units
        > node.gpu_capacity + 1e-9
        or background.memory_gb + candidate.resource_demand.memory_gb
        > node.memory_gb + 1e-9
    ):
        raise CandidateMaterializationError(
            f"candidate {candidate.task_id}:{candidate.node_id} exceeds "
            "capacity after observed background load"
        )
    duration_ms = candidate.compute_ms
    cursor = max(
        earliest_ms,
        problem.node_available_ms.get(
            candidate.node_id,
            problem.epoch.now_ms,
        ),
    )
    while True:
        finish = cursor + duration_ms
        overlapping = [
            reservation
            for reservation in existing
            if reservation.start_ms < finish - 1e-9
            and reservation.finish_ms > cursor + 1e-9
        ]
        boundaries = sorted(
            {
                cursor,
                *(
                    max(cursor, reservation.start_ms)
                    for reservation in overlapping
                ),
                *(
                    min(finish, reservation.finish_ms)
                    for reservation in overlapping
                ),
            }
        )
        feasible = True
        for point in boundaries:
            if point >= finish - 1e-9:
                continue
            active = [
                reservation
                for reservation in overlapping
                if reservation.start_ms <= point + 1e-9
                and reservation.finish_ms > point + 1e-9
            ]
            if len(active) + 1 > node.max_concurrency:
                feasible = False
                break
            if (
                background.cpu_units
                + sum(item.demand.cpu_units for item in active)
                + candidate.resource_demand.cpu_units
                > node.cpu_capacity + 1e-9
                or background.gpu_units
                + sum(item.demand.gpu_units for item in active)
                + candidate.resource_demand.gpu_units
                > node.gpu_capacity + 1e-9
                or background.memory_gb
                + sum(item.demand.memory_gb for item in active)
                + candidate.resource_demand.memory_gb
                > node.memory_gb + 1e-9
            ):
                feasible = False
                break
        if feasible:
            return cursor
        releases = [
            reservation.finish_ms
            for reservation in overlapping
            if reservation.finish_ms > cursor + 1e-9
        ]
        if not releases:
            return cursor
        cursor = min(releases)


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
