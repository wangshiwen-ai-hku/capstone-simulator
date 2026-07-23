"""Deterministic built-in optimizers for the canonical scheduling problem."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from ..models import (
    Assignment,
    ExecutionMode,
    NodeKind,
    TaskInstance,
    TransferReservation,
    resolved_placement_constraints,
)
from .base import (
    CandidateEstimate,
    OptimizerRegistry,
    PlannedResourceReservation,
    SchedulingPlan,
    SchedulingProblem,
)


class HeuristicOptimizer:
    """Joint ready-batch optimizer implementing one deterministic policy."""

    def __init__(self, optimizer_id: str) -> None:
        if optimizer_id not in {
            "dag_deadline",
            "rule_based",
            "local_first",
            "edge_first",
            "greedy_cost",
        }:
            raise ValueError(f"unknown heuristic optimizer {optimizer_id!r}")
        self.optimizer_id = optimizer_id

    def solve(self, problem: SchedulingProblem) -> SchedulingPlan:
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
        objective_value = 0.0

        for task in sorted(
            problem.epoch.ready_tasks,
            key=lambda item: _task_order(problem, item.task_id),
        ):
            materialized = [
                _materialize_candidate(
                    problem,
                    candidate,
                    reservations_by_node,
                    link_available,
                )
                for candidate in problem.candidates[task.task_id]
                if candidate.feasible
            ]
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
                    )
                )
                continue

            chosen, reservations, next_links, compute_start = self._choose(
                problem,
                task.task_id,
                materialized,
            )
            constraints = resolved_placement_constraints(task)
            mode = (
                ExecutionMode.LOCAL
                if chosen.node_id == task.source_node_id
                else ExecutionMode.EDGE
                if chosen.node_kind is NodeKind.EDGE
                else ExecutionMode.PEER
                if chosen.node_kind is NodeKind.ROBOT
                else ExecutionMode.CLOUD
            )
            if (
                mode is ExecutionMode.LOCAL
                and constraints.preferred_node_kinds
                and NodeKind.ROBOT not in constraints.preferred_node_kinds
            ):
                mode = ExecutionMode.FALLBACK_LOCAL

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
                execution_mode=mode,
                estimated_start_ms=chosen.start_ms,
                estimated_finish_ms=chosen.finish_ms,
                compute_ms=chosen.compute_ms,
                communication_ms=chosen.communication_ms,
                energy_j=chosen.energy_j,
                reason=_reason(self.optimizer_id),
                input_locations=chosen.input_locations,
                transfer_link_ids=link_ids,
                optimizer_id=self.optimizer_id,
                epoch_id=problem.epoch.epoch_id,
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
            objective_value += chosen.finish_ms + chosen.energy_j

        return SchedulingPlan(
            epoch_id=problem.epoch.epoch_id,
            optimizer_id=self.optimizer_id,
            assignments=tuple(assignments),
            node_reservations=tuple(node_reservations),
            transfer_reservations=tuple(transfer_reservations),
            objective_value=objective_value,
            diagnostics={
                "task_count": len(problem.epoch.ready_tasks),
                "scheduled_count": sum(
                    item.execution_mode is not ExecutionMode.DROP
                    for item in assignments
                ),
                "solve_budget_ms": problem.solve_budget_ms,
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
        task = problem.task_by_id[task_id]
        if self.optimizer_id == "local_first":
            return min(
                materialized,
                key=lambda item: (
                    not item[0].is_source,
                    item[0].finish_ms,
                    item[0].energy_j,
                    item[0].node_id,
                ),
            )
        if self.optimizer_id == "edge_first":
            return min(
                materialized,
                key=lambda item: (
                    item[0].node_kind is not NodeKind.EDGE,
                    item[0].finish_ms,
                    item[0].energy_j,
                    item[0].node_id,
                ),
            )
        if self.optimizer_id == "rule_based":
            constraints = resolved_placement_constraints(task)
            source_snapshot = problem.snapshot_by_id[task.source_node_id]
            should_offload = (
                NodeKind.EDGE in constraints.preferred_node_kinds
                or source_snapshot.cpu_util > 0.8
                or source_snapshot.gpu_util > 0.8
                or task.spec.compute_demand > 2.5
            )
            return min(
                materialized,
                key=lambda item: (
                    (
                        item[0].node_kind is not NodeKind.EDGE
                        if should_offload
                        else not item[0].is_source
                    ),
                    item[0].finish_ms,
                    item[0].energy_j,
                    item[0].node_id,
                ),
            )
        if self.optimizer_id == "dag_deadline":
            critical_tail = problem.critical_tail_ms.get(task_id, 0.0)

            def dag_score(
                item: tuple[
                    CandidateEstimate,
                    tuple[TransferReservation, ...],
                    dict[str, float],
                    float,
                ],
            ) -> tuple[float, int, float, float, str]:
                candidate = item[0]
                projected_finish = candidate.finish_ms + critical_tail
                lateness = max(
                    0.0,
                    projected_finish - task.deadline_time_ms,
                )
                locality_penalty = (
                    len(
                        set(candidate.input_locations)
                        - {candidate.node_id}
                    )
                    * 2.0
                )
                return (
                    lateness,
                    _preference_rank(task, candidate),
                    projected_finish + locality_penalty,
                    candidate.energy_j,
                    candidate.node_id,
                )

            return min(materialized, key=dag_score)
        return min(
            materialized,
            key=lambda item: (
                _preference_rank(task, item[0]),
                item[0].finish_ms,
                item[0].energy_j,
                item[0].node_id,
            ),
        )


def built_in_registry() -> OptimizerRegistry:
    registry = OptimizerRegistry()
    for optimizer_id in (
        "dag_deadline",
        "rule_based",
        "local_first",
        "edge_first",
        "greedy_cost",
    ):
        registry.register(HeuristicOptimizer(optimizer_id))
    return registry


def _preference_rank(
    task: TaskInstance,
    candidate: CandidateEstimate,
) -> int:
    preferred = resolved_placement_constraints(
        task
    ).preferred_node_kinds
    if not preferred:
        return 0
    try:
        return preferred.index(candidate.node_kind)
    except ValueError:
        return len(preferred)


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
                sum(item.demand.cpu_units for item in active)
                + candidate.resource_demand.cpu_units
                > node.cpu_capacity + 1e-9
                or sum(item.demand.gpu_units for item in active)
                + candidate.resource_demand.gpu_units
                > node.gpu_capacity + 1e-9
                or sum(item.demand.memory_gb for item in active)
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


def _reason(optimizer_id: str) -> str:
    return {
        "local_first": "local-first optimizer under declarative constraints",
        "edge_first": "edge-first optimizer under declarative constraints",
        "rule_based": "declarative placement rule optimizer",
        "dag_deadline": (
            "DAG deadline/critical-tail/data-locality batch optimizer"
        ),
        "greedy_cost": "minimum estimated finish time and energy optimizer",
    }[optimizer_id]
