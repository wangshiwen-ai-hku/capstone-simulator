"""Canonical scheduling problem, plan, validation, and optimizer contracts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Protocol, runtime_checkable

from ..models import (
    Assignment,
    ExecutionMode,
    LinkSnapshot,
    LinkSpec,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    TaskInstance,
    TransferEstimate,
    TransferReservation,
)


@dataclass(frozen=True)
class ResourceDemand:
    cpu_units: float
    gpu_units: float
    memory_gb: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.cpu_units,
                self.gpu_units,
                self.memory_gb,
            )
        ):
            raise ValueError("resource demand values must be finite")
        if min(self.cpu_units, self.gpu_units, self.memory_gb) < 0:
            raise ValueError("resource demand values must be non-negative")


@dataclass(frozen=True)
class CandidateEstimate:
    """One feasible or rejected task-to-node placement candidate."""

    task_id: str
    node_id: str
    node_kind: NodeKind
    source_node_id: str
    feasible: bool
    ready_time_ms: float
    start_ms: float
    finish_ms: float
    compute_ms: float
    communication_ms: float
    energy_j: float
    resource_demand: ResourceDemand
    input_locations: tuple[str, ...]
    transfers: tuple[TransferEstimate, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.ready_time_ms):
            raise ValueError("candidate ready_time_ms must be finite")
        if self.ready_time_ms < 0:
            raise ValueError("candidate ready_time_ms must be non-negative")
        if self.finish_ms < self.start_ms:
            raise ValueError("candidate cannot finish before it starts")
        if min(self.compute_ms, self.communication_ms, self.energy_j) < 0:
            raise ValueError("candidate estimates must be non-negative")
        if not all(
            math.isfinite(value)
            for value in (
                self.start_ms,
                self.compute_ms,
                self.communication_ms,
                self.energy_j,
            )
        ):
            raise ValueError("candidate estimates must be finite")
        if self.feasible and not math.isfinite(self.finish_ms):
            raise ValueError("feasible candidate finish_ms must be finite")
        if self.feasible and any(not item.feasible for item in self.transfers):
            raise ValueError(
                "a feasible candidate cannot contain an infeasible transfer"
            )
        if self.feasible and abs(
            self.communication_ms
            - sum(item.transfer_time_ms for item in self.transfers)
        ) > 1e-6:
            raise ValueError(
                "candidate communication estimate must equal its transfers"
            )

    @property
    def is_source(self) -> bool:
        return self.node_id == self.source_node_id


@dataclass(frozen=True)
class SchedulingEpoch:
    """The complete set of ready work considered at one control-plane instant."""

    epoch_id: str
    now_ms: float
    ready_tasks: tuple[TaskInstance, ...]

    def __post_init__(self) -> None:
        if not self.epoch_id.strip():
            raise ValueError("epoch_id must be non-blank")
        if not math.isfinite(self.now_ms) or self.now_ms < 0:
            raise ValueError("epoch now_ms must be non-negative")
        task_ids = tuple(task.task_id for task in self.ready_tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("an epoch cannot contain duplicate task ids")


@dataclass(frozen=True)
class PlannedResourceReservation:
    """Capacity reserved by a plan for one task on one execution node."""

    reservation_id: str
    epoch_id: str
    task_id: str
    node_id: str
    start_ms: float
    finish_ms: float
    demand: ResourceDemand

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.start_ms, self.finish_ms)
        ):
            raise ValueError(
                "resource reservation times must be finite"
            )
        if self.finish_ms < self.start_ms:
            raise ValueError("resource reservation cannot finish before it starts")


@dataclass(frozen=True)
class SchedulingProblem:
    """Transport-neutral input consumed by every MARS optimizer."""

    epoch: SchedulingEpoch
    node_specs: tuple[NodeSpec, ...]
    node_snapshots: tuple[NodeSnapshot, ...]
    link_specs: tuple[LinkSpec, ...]
    link_snapshots: tuple[LinkSnapshot, ...]
    candidates: Mapping[str, tuple[CandidateEstimate, ...]]
    node_available_ms: Mapping[str, float]
    link_available_ms: Mapping[str, float]
    existing_node_reservations: tuple[
        PlannedResourceReservation, ...
    ] = ()
    critical_tail_ms: Mapping[str, float] = field(default_factory=dict)
    solve_budget_ms: float = 50.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.solve_budget_ms)
            or self.solve_budget_ms <= 0
        ):
            raise ValueError("solve_budget_ms must be positive")
        node_ids = tuple(item.node_id for item in self.node_specs)
        snapshot_ids = tuple(item.node_id for item in self.node_snapshots)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("SchedulingProblem node ids must be unique")
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError("SchedulingProblem snapshot ids must be unique")
        if set(node_ids) != set(snapshot_ids):
            raise ValueError(
                "SchedulingProblem requires one snapshot for every node"
            )
        link_ids = tuple(item.link_id for item in self.link_specs)
        link_snapshot_ids = tuple(
            item.link_id for item in self.link_snapshots
        )
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("SchedulingProblem link ids must be unique")
        if len(link_snapshot_ids) != len(set(link_snapshot_ids)):
            raise ValueError(
                "SchedulingProblem link snapshot ids must be unique"
            )
        if set(link_ids) != set(link_snapshot_ids):
            raise ValueError(
                "SchedulingProblem requires one snapshot for every link"
            )
        for link in self.link_specs:
            if (
                link.source_node_id not in node_ids
                or link.target_node_id not in node_ids
            ):
                raise ValueError(
                    f"link {link.link_id} references an unknown node"
                )
        task_ids = tuple(task.task_id for task in self.epoch.ready_tasks)
        unknown_sources = {
            task.source_node_id
            for task in self.epoch.ready_tasks
            if task.source_node_id not in node_ids
        }
        if unknown_sources:
            raise ValueError(
                "SchedulingProblem tasks reference unknown source nodes: "
                f"{sorted(unknown_sources)}"
            )
        if set(self.candidates) != set(task_ids):
            raise ValueError(
                "SchedulingProblem candidates must cover every epoch task"
            )
        for task_id, estimates in self.candidates.items():
            if any(item.task_id != task_id for item in estimates):
                raise ValueError(
                    f"candidate task id mismatch for problem task {task_id}"
                )
            candidate_nodes = tuple(item.node_id for item in estimates)
            if len(candidate_nodes) != len(set(candidate_nodes)):
                raise ValueError(
                    f"task {task_id} has duplicate node candidates"
                )
            for estimate in estimates:
                if estimate.node_id not in node_ids:
                    raise ValueError(
                        f"candidate for task {task_id} references an "
                        f"unknown node {estimate.node_id}"
                    )
                node = next(
                    item
                    for item in self.node_specs
                    if item.node_id == estimate.node_id
                )
                if estimate.node_kind is not node.kind:
                    raise ValueError(
                        f"candidate node kind mismatch for {estimate.node_id}"
                    )
                for transfer in estimate.transfers:
                    unknown_links = (
                        set(transfer.path_link_ids) - set(link_ids)
                    )
                    if unknown_links:
                        raise ValueError(
                            f"candidate for task {task_id} references "
                            f"unknown links {sorted(unknown_links)}"
                        )
        if set(self.node_available_ms) != set(node_ids):
            raise ValueError(
                "node availability must cover every declared node exactly"
            )
        if any(
            not math.isfinite(value) or value < 0
            for value in self.node_available_ms.values()
        ):
            raise ValueError("node availability values must be non-negative")
        if set(self.link_available_ms) != set(link_ids):
            raise ValueError(
                "link availability must cover every declared link exactly"
            )
        if any(
            not math.isfinite(value) or value < 0
            for value in self.link_available_ms.values()
        ):
            raise ValueError("link availability values must be non-negative")
        existing_ids = tuple(
            reservation.reservation_id
            for reservation in self.existing_node_reservations
        )
        if len(existing_ids) != len(set(existing_ids)):
            raise ValueError(
                "existing resource reservation ids must be unique"
            )
        if any(
            reservation.node_id not in node_ids
            for reservation in self.existing_node_reservations
        ):
            raise ValueError(
                "existing resource reservation references an unknown node"
            )
        if set(self.critical_tail_ms) - set(task_ids):
            raise ValueError(
                "critical-tail estimates reference an unknown task"
            )
        if any(
            not math.isfinite(value) or value < 0
            for value in self.critical_tail_ms.values()
        ):
            raise ValueError(
                "critical-tail estimates must be non-negative"
            )

    @property
    def task_by_id(self) -> dict[str, TaskInstance]:
        return {task.task_id: task for task in self.epoch.ready_tasks}

    @property
    def node_by_id(self) -> dict[str, NodeSpec]:
        return {node.node_id: node for node in self.node_specs}

    @property
    def snapshot_by_id(self) -> dict[str, NodeSnapshot]:
        return {item.node_id: item for item in self.node_snapshots}

    @property
    def link_by_id(self) -> dict[str, LinkSpec]:
        return {item.link_id: item for item in self.link_specs}

    @property
    def link_snapshot_by_id(self) -> dict[str, LinkSnapshot]:
        return {item.link_id: item for item in self.link_snapshots}


@dataclass(frozen=True)
class SchedulingPlan:
    """Validated multi-task result returned by an optimizer."""

    epoch_id: str
    optimizer_id: str
    assignments: tuple[Assignment, ...]
    node_reservations: tuple[PlannedResourceReservation, ...] = ()
    transfer_reservations: tuple[TransferReservation, ...] = ()
    deferred_task_ids: tuple[str, ...] = ()
    objective_value: float = 0.0
    diagnostics: Mapping[str, float | int | str] = field(default_factory=dict)

    @property
    def assignment_by_task(self) -> dict[str, Assignment]:
        return {item.task_id: item for item in self.assignments}


@runtime_checkable
class Optimizer(Protocol):
    """A replaceable solver for the canonical SchedulingProblem."""

    optimizer_id: str

    def solve(self, problem: SchedulingProblem) -> SchedulingPlan: ...


class OptimizerRegistry:
    """Explicit registry used by API aliases and dependency injection."""

    def __init__(self) -> None:
        self._optimizers: dict[str, Optimizer] = {}

    def register(
        self,
        optimizer: Optimizer,
        *,
        aliases: tuple[str, ...] = (),
        replace: bool = False,
    ) -> None:
        names = (optimizer.optimizer_id, *aliases)
        if any(not name.strip() for name in names):
            raise ValueError("optimizer ids and aliases must be non-blank")
        if len(names) != len(set(names)):
            raise ValueError(
                "optimizer id and aliases must be unique"
            )
        collisions = [name for name in names if name in self._optimizers]
        if collisions and not replace:
            raise ValueError(
                f"optimizer ids already registered: {sorted(collisions)}"
            )
        for name in names:
            self._optimizers[name] = optimizer

    def resolve(self, optimizer: str | Optimizer) -> Optimizer:
        if isinstance(optimizer, str):
            try:
                return self._optimizers[optimizer]
            except KeyError as exc:
                raise KeyError(
                    f"unknown optimizer {optimizer!r}; available="
                    f"{sorted(self._optimizers)}"
                ) from exc
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a registered id or Optimizer")
        return optimizer

    def extend(
        self,
        other: OptimizerRegistry,
        *,
        replace: bool = False,
    ) -> None:
        collisions = set(self._optimizers) & set(other._optimizers)
        if collisions and not replace:
            raise ValueError(
                "optimizer registries overlap: "
                f"{sorted(collisions)}"
            )
        self._optimizers.update(other._optimizers)

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._optimizers))


class PlanValidationError(ValueError):
    """An optimizer returned a structurally invalid or infeasible plan."""


def validate_plan(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
) -> SchedulingPlan:
    """Prove task coverage, candidate feasibility, and reservation safety."""

    if plan.epoch_id != problem.epoch.epoch_id:
        raise PlanValidationError("plan epoch_id does not match the problem")
    if not plan.optimizer_id.strip():
        raise PlanValidationError("plan optimizer_id must be non-blank")
    if (
        not math.isfinite(plan.objective_value)
        or plan.objective_value < 0
    ):
        raise PlanValidationError(
            "plan objective_value must be finite and non-negative"
        )

    expected = {task.task_id for task in problem.epoch.ready_tasks}
    assignment_ids = tuple(item.task_id for item in plan.assignments)
    if len(assignment_ids) != len(set(assignment_ids)):
        raise PlanValidationError("plan contains duplicate task assignments")
    if len(plan.deferred_task_ids) != len(set(plan.deferred_task_ids)):
        raise PlanValidationError("plan contains duplicate deferred task ids")
    deferred = set(plan.deferred_task_ids)
    if set(assignment_ids) & deferred:
        raise PlanValidationError(
            "a task cannot be both assigned and deferred"
        )
    if set(assignment_ids) | deferred != expected:
        raise PlanValidationError(
            "plan must assign or explicitly defer every epoch task"
        )

    candidates = {
        task_id: {
            item.node_id: item
            for item in estimates
            if item.feasible
        }
        for task_id, estimates in problem.candidates.items()
    }
    non_drop_ids: set[str] = set()
    selected_candidates: dict[str, CandidateEstimate] = {}
    for assignment in plan.assignments:
        if assignment.epoch_id != plan.epoch_id:
            raise PlanValidationError(
                f"task {assignment.task_id} assignment epoch_id mismatch"
            )
        if assignment.optimizer_id != plan.optimizer_id:
            raise PlanValidationError(
                f"task {assignment.task_id} assignment optimizer_id mismatch"
            )
        if assignment.estimated_finish_ms < assignment.estimated_start_ms:
            raise PlanValidationError(
                f"task {assignment.task_id} finishes before it starts"
            )
        numeric_estimates = (
            assignment.estimated_start_ms,
            assignment.estimated_finish_ms,
            assignment.compute_ms,
            assignment.communication_ms,
            assignment.energy_j,
        )
        if (
            not all(math.isfinite(value) for value in numeric_estimates)
            or min(numeric_estimates) < 0
        ):
            raise PlanValidationError(
                f"task {assignment.task_id} contains an invalid estimate"
            )
        if assignment.execution_mode is ExecutionMode.DROP:
            if assignment.target_node_id:
                raise PlanValidationError(
                    "DROP assignments must not have a target node"
                )
            if (
                assignment.estimated_finish_ms
                != assignment.estimated_start_ms
                or assignment.compute_ms != 0
                or assignment.communication_ms != 0
                or assignment.energy_j != 0
                or assignment.transfer_link_ids
            ):
                raise PlanValidationError(
                    "DROP assignments cannot reserve work or report cost"
                )
            task = problem.task_by_id[assignment.task_id]
            if assignment.estimated_start_ms + 1e-9 < max(
                problem.epoch.now_ms,
                task.arrival_time_ms,
            ):
                raise PlanValidationError(
                    f"DROP task {assignment.task_id} starts before it is ready"
                )
            continue
        non_drop_ids.add(assignment.task_id)
        if not assignment.target_node_id:
            raise PlanValidationError(
                "non-DROP assignments require a target node"
            )
        if assignment.target_node_id not in candidates[assignment.task_id]:
            raise PlanValidationError(
                f"task {assignment.task_id} selected an infeasible candidate"
            )
        candidate = candidates[assignment.task_id][
            assignment.target_node_id
        ]
        selected_candidates[assignment.task_id] = candidate
        if assignment.estimated_start_ms + 1e-9 < candidate.ready_time_ms:
            raise PlanValidationError(
                f"task {assignment.task_id} starts before it is ready"
            )
        if abs(assignment.compute_ms - candidate.compute_ms) > 1e-6:
            raise PlanValidationError(
                f"task {assignment.task_id} compute estimate does not "
                "match its candidate"
            )
        if abs(assignment.energy_j - candidate.energy_j) > 1e-6:
            raise PlanValidationError(
                f"task {assignment.task_id} energy estimate does not "
                "match its candidate"
            )
        allowed_modes = (
            {ExecutionMode.LOCAL, ExecutionMode.FALLBACK_LOCAL}
            if candidate.is_source
            else {ExecutionMode.EDGE}
            if candidate.node_kind is NodeKind.EDGE
            else {ExecutionMode.PEER}
            if candidate.node_kind is NodeKind.ROBOT
            else {ExecutionMode.CLOUD}
        )
        if assignment.execution_mode not in allowed_modes:
            raise PlanValidationError(
                f"task {assignment.task_id} execution mode does not "
                "match its selected node"
            )

    reservations_by_task: dict[str, list[PlannedResourceReservation]] = (
        defaultdict(list)
    )
    reservations_by_node: dict[str, list[PlannedResourceReservation]] = (
        defaultdict(list)
    )
    reservation_ids = tuple(
        item.reservation_id for item in plan.node_reservations
    )
    if len(reservation_ids) != len(set(reservation_ids)):
        raise PlanValidationError(
            "resource reservation ids must be unique"
        )
    for reservation in plan.node_reservations:
        if reservation.epoch_id != plan.epoch_id:
            raise PlanValidationError(
                "resource reservation epoch_id mismatch"
            )
        reservations_by_task[reservation.task_id].append(reservation)
        reservations_by_node[reservation.node_id].append(reservation)
    if set(reservations_by_task) != non_drop_ids:
        raise PlanValidationError(
            "every non-DROP assignment requires one resource reservation"
        )
    if any(len(items) != 1 for items in reservations_by_task.values()):
        raise PlanValidationError(
            "every task must have exactly one resource reservation"
        )
    assignment_by_task = plan.assignment_by_task
    for task_id, items in reservations_by_task.items():
        reservation = items[0]
        assignment = assignment_by_task[task_id]
        if reservation.node_id != assignment.target_node_id:
            raise PlanValidationError(
                f"task {task_id} reservation node does not match assignment"
            )
        candidate = selected_candidates[task_id]
        if reservation.demand != candidate.resource_demand:
            raise PlanValidationError(
                f"task {task_id} resource demand does not match candidate"
            )
        if reservation.start_ms + 1e-9 < assignment.estimated_start_ms:
            raise PlanValidationError(
                f"task {task_id} resource reservation starts before assignment"
            )
        if reservation.start_ms + 1e-9 < problem.node_available_ms[
            reservation.node_id
        ]:
            raise PlanValidationError(
                f"task {task_id} starts before its node is available"
            )
        if abs(
            reservation.finish_ms
            - reservation.start_ms
            - candidate.compute_ms
        ) > 1e-6:
            raise PlanValidationError(
                f"task {task_id} resource reservation duration does not "
                "match compute estimate"
            )
        if abs(reservation.finish_ms - assignment.estimated_finish_ms) > 1e-6:
            raise PlanValidationError(
                f"task {task_id} reservation finish does not match assignment"
            )

    node_by_id = problem.node_by_id
    existing_by_node: dict[
        str, list[PlannedResourceReservation]
    ] = defaultdict(list)
    for reservation in problem.existing_node_reservations:
        existing_by_node[reservation.node_id].append(reservation)
    if set(reservation_ids) & {
        item.reservation_id
        for item in problem.existing_node_reservations
    }:
        raise PlanValidationError(
            "new and existing resource reservation ids must be distinct"
        )
    for node_id in set(existing_by_node) | set(reservations_by_node):
        node = node_by_id.get(node_id)
        if node is None:
            raise PlanValidationError(
                f"reservation references unknown node {node_id}"
            )
        _validate_node_capacity(
            node,
            [
                *existing_by_node[node_id],
                *reservations_by_node[node_id],
            ],
        )

    _validate_transfer_reservations(
        problem,
        plan,
        selected_candidates,
        non_drop_ids,
    )
    return plan


def _validate_node_capacity(
    node: NodeSpec,
    reservations: list[PlannedResourceReservation],
) -> None:
    boundaries = sorted(
        {
            point
            for reservation in reservations
            for point in (reservation.start_ms, reservation.finish_ms)
        }
    )
    for point in boundaries:
        active = [
            item
            for item in reservations
            if item.start_ms <= point < item.finish_ms
        ]
        if len(active) > node.max_concurrency:
            raise PlanValidationError(
                f"node {node.node_id} exceeds max_concurrency at {point}"
            )
        cpu = sum(item.demand.cpu_units for item in active)
        gpu = sum(item.demand.gpu_units for item in active)
        memory = sum(item.demand.memory_gb for item in active)
        if cpu > node.cpu_capacity + 1e-9:
            raise PlanValidationError(
                f"node {node.node_id} exceeds CPU capacity at {point}"
            )
        if gpu > node.gpu_capacity + 1e-9:
            raise PlanValidationError(
                f"node {node.node_id} exceeds GPU capacity at {point}"
            )
        if memory > node.memory_gb + 1e-9:
            raise PlanValidationError(
                f"node {node.node_id} exceeds memory capacity at {point}"
            )


def _validate_transfer_reservations(
    problem: SchedulingProblem,
    plan: SchedulingPlan,
    selected_candidates: Mapping[str, CandidateEstimate],
    non_drop_ids: set[str],
) -> None:
    reservations = plan.transfer_reservations
    reservation_ids = tuple(item.reservation_id for item in reservations)
    if len(reservation_ids) != len(set(reservation_ids)):
        raise PlanValidationError(
            "transfer reservation ids must be unique"
        )
    by_task: dict[str, list[TransferReservation]] = defaultdict(list)
    by_link: dict[str, list[TransferReservation]] = defaultdict(list)
    for reservation in reservations:
        if reservation.epoch_id != plan.epoch_id:
            raise PlanValidationError(
                "transfer reservation epoch_id mismatch"
            )
        if reservation.task_id not in non_drop_ids:
            raise PlanValidationError(
                "transfer reservation must belong to a non-DROP task"
            )
        if not reservation.path_link_ids:
            raise PlanValidationError(
                "transfer reservation requires a non-empty link path"
            )
        unknown_links = (
            set(reservation.path_link_ids) - set(problem.link_by_id)
        )
        if unknown_links:
            raise PlanValidationError(
                f"transfer reservation references unknown links "
                f"{sorted(unknown_links)}"
            )
        if reservation.start_ms + 1e-9 < max(
            problem.link_available_ms[link_id]
            for link_id in reservation.path_link_ids
        ):
            raise PlanValidationError(
                f"task {reservation.task_id} transfer starts before a "
                "link is available"
            )
        by_task[reservation.task_id].append(reservation)
        for link_id in reservation.path_link_ids:
            by_link[link_id].append(reservation)

    assignment_by_task = plan.assignment_by_task
    for task_id in non_drop_ids:
        assignment = assignment_by_task[task_id]
        candidate = selected_candidates[task_id]
        expected = sorted(
            (
                transfer.transfer_id,
                transfer.path_link_ids,
                transfer.size_mb,
            )
            for transfer in candidate.transfers
            if transfer.path_link_ids and transfer.transfer_time_ms > 0
        )
        actual = sorted(
            (
                reservation.transfer_id,
                reservation.path_link_ids,
                reservation.size_mb,
            )
            for reservation in by_task.get(task_id, ())
        )
        if actual != expected:
            raise PlanValidationError(
                f"task {task_id} transfer reservations do not match "
                "the selected candidate"
            )
        expected_by_transfer = {
            (
                transfer.transfer_id,
                transfer.path_link_ids,
                transfer.size_mb,
            ): transfer.transfer_time_ms
            for transfer in candidate.transfers
            if transfer.path_link_ids and transfer.transfer_time_ms > 0
        }
        for reservation in by_task.get(task_id, ()):
            expected_duration = expected_by_transfer[
                (
                    reservation.transfer_id,
                    reservation.path_link_ids,
                    reservation.size_mb,
                )
            ]
            if abs(
                reservation.finish_ms
                - reservation.start_ms
                - expected_duration
            ) > 1e-6:
                raise PlanValidationError(
                    f"task {task_id} transfer duration does not match "
                    "its candidate"
                )
        planned_links = tuple(
            dict.fromkeys(
                link_id
                for reservation in by_task.get(task_id, ())
                for link_id in reservation.path_link_ids
            )
        )
        if assignment.transfer_link_ids != planned_links:
            raise PlanValidationError(
                f"task {task_id} assignment transfer links do not match "
                "its reservations"
            )
        transfer_duration = sum(
            reservation.finish_ms - reservation.start_ms
            for reservation in by_task.get(task_id, ())
        )
        if abs(
            assignment.communication_ms - transfer_duration
        ) > 1e-6:
            raise PlanValidationError(
                f"task {task_id} communication estimate does not match "
                "its transfer reservations"
            )
        start_candidates = [
            reservation.start_ms
            for reservation in by_task.get(task_id, ())
        ]
        resource_start = next(
            reservation.start_ms
            for reservation in plan.node_reservations
            if reservation.task_id == task_id
        )
        if by_task.get(task_id) and resource_start + 1e-9 < max(
            reservation.finish_ms
            for reservation in by_task[task_id]
        ):
            raise PlanValidationError(
                f"task {task_id} compute starts before input transfers finish"
            )
        expected_assignment_start = min(
            [resource_start, *start_candidates]
        )
        if abs(
            assignment.estimated_start_ms - expected_assignment_start
        ) > 1e-6:
            raise PlanValidationError(
                f"task {task_id} assignment start does not match its "
                "reservations"
            )

    for link_id, items in by_link.items():
        ordered = sorted(items, key=lambda item: (item.start_ms, item.finish_ms))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_ms < previous.finish_ms - 1e-9:
                raise PlanValidationError(
                    f"link {link_id} has overlapping transfer reservations"
                )
