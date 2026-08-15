"""Shared candidate materialization for built-in optimizers."""

from __future__ import annotations

from dataclasses import replace

from ..domain.execution import Assignment
from ..domain.transfer import TransferReservation
from .base import (
    CandidateEstimate,
    CandidateMaterializationError,
    PlannedResourceReservation,
    SchedulingProblem,
    background_resource_demand,
    execution_mode_for_candidate,
)


def materialize_candidate(
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

    compute_start = earliest_resource_start(
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


def build_assignment(
    problem: SchedulingProblem,
    candidate: CandidateEstimate,
    transfer_reservations: tuple[TransferReservation, ...],
    *,
    optimizer_id: str,
    reason: str,
) -> Assignment:
    """Build the canonical Assignment projection of a materialized candidate."""

    task = problem.task_by_id[candidate.task_id]
    link_ids = tuple(
        dict.fromkeys(
            link_id
            for reservation in transfer_reservations
            for link_id in reservation.path_link_ids
        )
    )
    return Assignment(
        task_id=candidate.task_id,
        target_node_id=candidate.node_id,
        execution_mode=execution_mode_for_candidate(task, candidate),
        estimated_start_ms=candidate.start_ms,
        estimated_finish_ms=candidate.finish_ms,
        compute_ms=candidate.compute_ms,
        communication_ms=candidate.communication_ms,
        energy_j=candidate.energy_j,
        reason=reason,
        input_locations=candidate.input_locations,
        transfer_link_ids=link_ids,
        optimizer_id=optimizer_id,
        epoch_id=problem.epoch.epoch_id,
        output_size_mb=candidate.output_size_mb,
        success_probability=candidate.success_probability,
    )


def build_node_reservation(
    problem: SchedulingProblem,
    candidate: CandidateEstimate,
    *,
    compute_start_ms: float,
    reservation_id: str,
) -> PlannedResourceReservation:
    """Build the resource reservation paired with one Assignment."""

    return PlannedResourceReservation(
        reservation_id=reservation_id,
        epoch_id=problem.epoch.epoch_id,
        task_id=candidate.task_id,
        node_id=candidate.node_id,
        start_ms=compute_start_ms,
        finish_ms=candidate.finish_ms,
        demand=candidate.resource_demand,
    )


def earliest_resource_start(
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


__all__ = [
    "build_assignment",
    "build_node_reservation",
    "earliest_resource_start",
    "materialize_candidate",
]
