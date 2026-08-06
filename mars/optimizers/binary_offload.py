"""Exact binary local-or-edge optimizer for the Stage 2 experiment."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import math
from time import perf_counter

from ..domain.execution import Assignment, ExecutionMode
from ..domain.topology import NodeKind
from ..domain.transfer import TransferReservation
from .base import (
    CandidateEstimate,
    PlannedResourceReservation,
    SchedulingPlan,
    SchedulingProblem,
    SolveStatus,
)
from .heuristics import _materialize_candidate


class BinaryOffloadOptimizer:
    """Exhaustively solve the frozen binary local-or-edge objective."""

    optimizer_id = "binary_offload"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 0.01,
        gamma: float = 2.0,
    ) -> None:
        weights = (alpha, beta, gamma)
        if not all(math.isfinite(value) and value >= 0 for value in weights):
            raise ValueError("binary-offload weights must be finite and non-negative")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.solve_history: list[dict[str, object]] = []

    def solve(self, problem: SchedulingProblem) -> SchedulingPlan:
        """Return the exact optimum over every local/edge combination."""

        solve_started = perf_counter()
        tasks = tuple(problem.epoch.ready_tasks)
        candidate_options = tuple(
            self._binary_candidates(problem, task.task_id, task.source_node_id)
            for task in tasks
        )
        best_score = math.inf
        best_selection: tuple[CandidateEstimate, ...] | None = None
        best_materialized: tuple[
            tuple[Assignment, ...],
            tuple[PlannedResourceReservation, ...],
            tuple[TransferReservation, ...],
        ] | None = None
        evaluated = 0

        for selection in product(*candidate_options):
            evaluated += 1
            if not self._is_feasible(problem, selection):
                continue
            try:
                materialized = self._materialize(problem, selection)
            except (RuntimeError, ValueError):
                continue
            score = self._objective(
                problem,
                selection,
                materialized[1],
            )
            if score < best_score - 1e-9:
                best_score = score
                best_selection = selection
                best_materialized = materialized

        if best_selection is None or best_materialized is None:
            raise ValueError("binary offload problem has no feasible assignment")

        assignments, node_reservations, transfer_reservations = (
            best_materialized
        )
        elapsed_ms = (perf_counter() - solve_started) * 1000.0
        success_reward, communication_ms, maximum_utilization = (
            self._objective_components(
                problem,
                best_selection,
                node_reservations,
            )
        )
        self.solve_history.append(
            {
                "epoch_id": problem.epoch.epoch_id,
                "ready_task_count": len(tasks),
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "success_reward": success_reward,
                "communication_time_ms": communication_ms,
                "maximum_resource_utilization": maximum_utilization,
                "objective_value": best_score,
                "solve_elapsed_ms": elapsed_ms,
                "enumerated_combinations": evaluated,
                "assignments": {
                    item.task_id: item.node_id
                    for item in best_selection
                },
            }
        )
        return SchedulingPlan(
            problem_id=problem.problem_id,
            snapshot_id=problem.snapshot.snapshot_id,
            policy_id=problem.policy.policy_id,
            policy_version=problem.policy.version,
            epoch_id=problem.epoch.epoch_id,
            optimizer_id=self.optimizer_id,
            optimizer_version="1",
            solve_status=SolveStatus.OPTIMAL,
            solve_elapsed_ms=elapsed_ms,
            iteration_count=evaluated,
            termination_reason="exhaustive_binary_search_complete",
            assignments=assignments,
            node_reservations=node_reservations,
            transfer_reservations=transfer_reservations,
            diagnostics={
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "binary_objective_value": best_score,
                "enumerated_combinations": evaluated,
            },
        )

    @staticmethod
    def _binary_candidates(
        problem: SchedulingProblem,
        task_id: str,
        source_node_id: str,
    ) -> tuple[CandidateEstimate, ...]:
        candidates = tuple(
            candidate
            for candidate in problem.candidates[task_id]
            if candidate.feasible
        )
        local = tuple(
            item for item in candidates if item.node_id == source_node_id
        )
        edge = tuple(
            item for item in candidates if item.node_kind is NodeKind.EDGE
        )
        if len(local) != 1 or len(edge) > 1:
            raise ValueError(
                f"task {task_id} requires exactly one source candidate "
                "and at most one edge candidate"
            )
        return (local[0], *edge)

    def _objective(
        self,
        problem: SchedulingProblem,
        selection: tuple[CandidateEstimate, ...],
        reservations: tuple[PlannedResourceReservation, ...],
    ) -> float:
        success_reward, communication_ms, maximum_utilization = (
            self._objective_components(
                problem,
                selection,
                reservations,
            )
        )
        return (
            -self.alpha * success_reward
            + self.beta * communication_ms
            + self.gamma * maximum_utilization
        )

    def _objective_components(
        self,
        problem: SchedulingProblem,
        selection: tuple[CandidateEstimate, ...],
        reservations: tuple[PlannedResourceReservation, ...],
    ) -> tuple[float, float, float]:
        task_by_id = problem.task_by_id
        return (
            sum(
                task_by_id[item.task_id].priority
                * item.success_probability
                for item in selection
            ),
            sum(item.communication_ms for item in selection),
            self._maximum_utilization(problem, reservations),
        )

    @staticmethod
    def _maximum_utilization(
        problem: SchedulingProblem,
        reservations: tuple[PlannedResourceReservation, ...],
    ) -> float:
        all_reservations = (
            *problem.existing_node_reservations,
            *reservations,
        )
        now_ms = problem.epoch.now_ms
        peak = max(
            (
                value
                for snapshot in problem.node_snapshots
                for value in (snapshot.cpu_util, snapshot.gpu_util)
            ),
            default=0.0,
        )
        for node in problem.node_specs:
            node_reservations = tuple(
                item
                for item in all_reservations
                if item.node_id == node.node_id
            )
            active_existing = tuple(
                item
                for item in problem.existing_node_reservations
                if (
                    item.node_id == node.node_id
                    and item.start_ms <= now_ms < item.finish_ms
                )
            )
            snapshot = problem.snapshot_by_id[node.node_id]
            existing_cpu = sum(
                item.demand.cpu_units for item in active_existing
            ) / node.cpu_capacity
            existing_gpu = (
                sum(item.demand.gpu_units for item in active_existing)
                / node.gpu_capacity
                if node.gpu_capacity > 0
                else 0.0
            )
            background_cpu = (
                snapshot.cpu_util
                if snapshot.cpu_util > existing_cpu + 1e-9
                else 0.0
            )
            background_gpu = (
                snapshot.gpu_util
                if snapshot.gpu_util > existing_gpu + 1e-9
                else 0.0
            )
            boundaries = {
                now_ms,
                *(
                    point
                    for item in node_reservations
                    for point in (item.start_ms, item.finish_ms)
                    if point >= now_ms
                ),
            }
            for point in boundaries:
                active = tuple(
                    item
                    for item in node_reservations
                    if item.start_ms <= point < item.finish_ms
                )
                cpu_util = background_cpu + sum(
                    item.demand.cpu_units for item in active
                ) / node.cpu_capacity
                gpu_util = background_gpu
                if node.gpu_capacity > 0:
                    gpu_util += sum(
                        item.demand.gpu_units for item in active
                    ) / node.gpu_capacity
                elif any(item.demand.gpu_units > 0 for item in active):
                    return math.inf
                peak = max(peak, cpu_util, gpu_util)
        return peak

    @staticmethod
    def _is_feasible(
        problem: SchedulingProblem,
        selection: tuple[CandidateEstimate, ...],
    ) -> bool:
        local_energy_by_node = defaultdict(float)
        task_by_id = problem.task_by_id
        snapshots = problem.snapshot_by_id

        for candidate in selection:
            task = task_by_id[candidate.task_id]
            if not snapshots[candidate.node_id].online:
                return False
            if candidate.node_id == task.source_node_id:
                local_energy_by_node[candidate.node_id] += candidate.energy_j

        for node_id, requested_energy in local_energy_by_node.items():
            snapshot = snapshots[node_id]
            remaining_energy = snapshot.remaining_energy_j
            if (
                remaining_energy is not None
                and requested_energy > remaining_energy + 1e-9
            ):
                return False
        return True

    def _materialize(
        self,
        problem: SchedulingProblem,
        selection: tuple[CandidateEstimate, ...],
    ) -> tuple[
        tuple[Assignment, ...],
        tuple[PlannedResourceReservation, ...],
        tuple[TransferReservation, ...],
    ]:
        link_available = dict(problem.link_available_ms)
        assignments: list[Assignment] = []
        node_reservations: list[PlannedResourceReservation] = []
        reservations_by_node: dict[
            str, list[PlannedResourceReservation]
        ] = defaultdict(list)
        for reservation in problem.existing_node_reservations:
            reservations_by_node[reservation.node_id].append(reservation)
        transfer_reservations: list[TransferReservation] = []

        for candidate in selection:
            (
                materialized,
                task_transfers,
                next_links,
                compute_start,
            ) = _materialize_candidate(
                problem,
                candidate,
                reservations_by_node,
                link_available,
            )
            link_available.update(next_links)
            transfer_reservations.extend(task_transfers)
            link_ids = tuple(
                dict.fromkeys(
                    link_id
                    for reservation in task_transfers
                    for link_id in reservation.path_link_ids
                )
            )
            assignments.append(
                Assignment(
                    task_id=candidate.task_id,
                    target_node_id=candidate.node_id,
                    execution_mode=(
                        ExecutionMode.LOCAL
                        if materialized.is_source
                        else ExecutionMode.EDGE
                    ),
                    estimated_start_ms=materialized.start_ms,
                    estimated_finish_ms=materialized.finish_ms,
                    compute_ms=materialized.compute_ms,
                    communication_ms=materialized.communication_ms,
                    energy_j=materialized.energy_j,
                    reason="exact binary local-or-edge optimum",
                    input_locations=materialized.input_locations,
                    transfer_link_ids=link_ids,
                    optimizer_id=self.optimizer_id,
                    epoch_id=problem.epoch.epoch_id,
                )
            )
            node_reservation = PlannedResourceReservation(
                    reservation_id=(
                        f"binary-node:{problem.epoch.epoch_id}:"
                        f"{candidate.task_id}:{candidate.node_id}"
                    ),
                    epoch_id=problem.epoch.epoch_id,
                    task_id=candidate.task_id,
                    node_id=candidate.node_id,
                    start_ms=compute_start,
                    finish_ms=materialized.finish_ms,
                    demand=materialized.resource_demand,
                )
            node_reservations.append(node_reservation)
            reservations_by_node[candidate.node_id].append(
                node_reservation
            )

        return (
            tuple(assignments),
            tuple(node_reservations),
            tuple(transfer_reservations),
        )


__all__ = ["BinaryOffloadOptimizer"]
