"""The single asynchronous boundary between the MARS control plane and agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..domain.artifact import (
    ArtifactRef,
    InputArtifactBinding,
    artifacts_from_bindings,
)
from ..domain.execution import Assignment
from ..domain.task import TaskInstance
from ..domain.topology import (
    NodeSnapshot,
    NodeSpec,
)
from ..domain.transfer import TransferReservation
from ..optimizers.base import PlannedResourceReservation


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Behavior that a runtime adapter can provide to the control plane."""

    discovery: bool
    reliable_control: bool
    feedback: bool
    cancellation: bool
    liveliness: bool
    virtual_time: bool


@dataclass(frozen=True)
class AgentHeartbeat:
    agent_id: str
    sequence: int
    sampled_at_ms: float
    snapshot: NodeSnapshot
    active_reservations: int


@dataclass(frozen=True)
class RuntimeInventory:
    """Registered nodes and their latest dynamic state."""

    nodes: tuple[NodeSpec, ...]
    heartbeats: tuple[AgentHeartbeat, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(
            self,
            "heartbeats",
            tuple(self.heartbeats),
        )
        node_ids = tuple(node.node_id for node in self.nodes)
        heartbeat_ids = tuple(item.agent_id for item in self.heartbeats)
        if not node_ids:
            raise ValueError("runtime inventory must contain at least one node")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("runtime inventory contains duplicate node ids")
        if (
            len(heartbeat_ids) != len(set(heartbeat_ids))
            or len(heartbeat_ids) != len(node_ids)
            or set(node_ids) != set(heartbeat_ids)
        ):
            raise ValueError("every runtime node must have exactly one heartbeat")
        if any(
            heartbeat.snapshot.node_id != heartbeat.agent_id
            for heartbeat in self.heartbeats
        ):
            raise ValueError("heartbeat snapshot must match its agent id")

    @property
    def snapshots(self) -> dict[str, NodeSnapshot]:
        return {item.agent_id: item.snapshot for item in self.heartbeats}


@dataclass(frozen=True)
class DispatchCommand:
    attempt_id: str
    attempt_no: int
    task: TaskInstance
    assignment: Assignment
    resource_reservation: PlannedResourceReservation
    transfer_reservations: tuple[TransferReservation, ...]
    input_artifact_bindings: tuple[InputArtifactBinding, ...]
    problem_id: str
    snapshot_id: str
    policy_id: str
    policy_version: str
    seed: int
    inject_failure: bool = False

    def __post_init__(self) -> None:
        """Validate the exact, already-approved plan fragment being committed."""

        object.__setattr__(
            self,
            "transfer_reservations",
            tuple(self.transfer_reservations),
        )
        object.__setattr__(
            self,
            "input_artifact_bindings",
            tuple(self.input_artifact_bindings),
        )
        assignment = self.assignment
        resource = self.resource_reservation
        if not all(
            value.strip()
            for value in (
                self.problem_id,
                self.snapshot_id,
                self.policy_id,
                self.policy_version,
            )
        ):
            raise ValueError(
                "dispatch requires problem, snapshot, and policy correlation"
            )
        if not assignment.epoch_id or not assignment.optimizer_id:
            raise ValueError(
                "dispatch assignment must come from a validated scheduling plan"
            )
        if assignment.task_id != self.task.task_id:
            raise ValueError("dispatch task and assignment must match")
        if any(
            binding.consumer_task_id != self.task.task_id
            for binding in self.input_artifact_bindings
        ):
            raise ValueError(
                "dispatch input bindings must target the dispatched task"
            )
        consumer_ports = tuple(
            binding.consumer_port
            for binding in self.input_artifact_bindings
        )
        if len(consumer_ports) != len(set(consumer_ports)):
            raise ValueError(
                "dispatch input bindings must use unique consumer ports"
            )
        input_locations = tuple(
            artifact.node_id for artifact in self.input_artifacts
        )
        if assignment.input_locations != input_locations:
            raise ValueError(
                "dispatch input bindings must match assignment inputs"
            )
        if (
            resource.task_id != assignment.task_id
            or resource.node_id != assignment.target_node_id
            or resource.epoch_id != assignment.epoch_id
        ):
            raise ValueError(
                "dispatch resource reservation must match its assignment"
            )
        if abs(
            resource.finish_ms
            - resource.start_ms
            - assignment.compute_ms
        ) > 1e-6:
            raise ValueError(
                "dispatch resource reservation must match planned compute time"
            )
        if abs(
            resource.finish_ms - assignment.estimated_finish_ms
        ) > 1e-6:
            raise ValueError(
                "dispatch resource reservation must match assignment finish"
            )

        transfer_ids = tuple(
            reservation.reservation_id
            for reservation in self.transfer_reservations
        )
        if len(transfer_ids) != len(set(transfer_ids)):
            raise ValueError(
                "dispatch transfer reservation ids must be unique"
            )
        if any(
            reservation.task_id != assignment.task_id
            or reservation.epoch_id != assignment.epoch_id
            for reservation in self.transfer_reservations
        ):
            raise ValueError(
                "dispatch transfer reservations must match their assignment"
            )
        planned_links = tuple(
            dict.fromkeys(
                link_id
                for reservation in self.transfer_reservations
                for link_id in reservation.path_link_ids
            )
        )
        if assignment.transfer_link_ids != planned_links:
            raise ValueError(
                "dispatch transfer reservations must match assignment links"
            )
        communication_ms = sum(
            reservation.finish_ms - reservation.start_ms
            for reservation in self.transfer_reservations
        )
        if abs(communication_ms - assignment.communication_ms) > 1e-6:
            raise ValueError(
                "dispatch transfer reservations must match planned communication"
            )
        if self.transfer_reservations and resource.start_ms + 1e-9 < max(
            reservation.finish_ms
            for reservation in self.transfer_reservations
        ):
            raise ValueError(
                "dispatch compute reservation starts before transfers finish"
            )
        expected_start = min(
            (
                resource.start_ms,
                *(
                    reservation.start_ms
                    for reservation in self.transfer_reservations
                ),
            )
        )
        if abs(expected_start - assignment.estimated_start_ms) > 1e-6:
            raise ValueError(
                "dispatch reservations must match assignment start"
            )

    @property
    def input_artifacts(self) -> tuple[ArtifactRef, ...]:
        """Read-only compatibility view of unique input payloads."""

        return artifacts_from_bindings(self.input_artifact_bindings)


@dataclass(frozen=True)
class DispatchAck:
    dispatch_id: str
    attempt_id: str
    task_id: str
    agent_id: str
    accepted: bool
    error_code: str = ""


@dataclass(frozen=True)
class AttemptCompletion:
    """One terminal result correlated to an exact dispatch and retry attempt."""

    dispatch_id: str
    attempt_id: str
    task_id: str
    agent_id: str
    ok: bool
    started_time_ms: float
    finished_time_ms: float
    compute_time_ms: float
    energy_j: float
    outputs: tuple[ArtifactRef, ...]
    error_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if self.started_time_ms < 0:
            raise ValueError(
                "attempt completion start time must be non-negative"
            )
        if self.finished_time_ms < self.started_time_ms:
            raise ValueError(
                "attempt completion cannot finish before it starts"
            )
        if self.compute_time_ms < 0 or self.energy_j < 0:
            raise ValueError(
                "attempt completion compute time and energy "
                "must be non-negative"
            )


@runtime_checkable
class RuntimePort(Protocol):
    """Coordinator-facing runtime contract implemented by every adapter.

    An adapter may execute agents in process or translate these operations to
    gRPC, DDS, or a deployment-specific runtime. Workflow submission remains
    an application concern and is intentionally outside this boundary.
    """

    capabilities: RuntimeCapabilities

    async def start(self, now_ms: float) -> RuntimeInventory: ...

    async def inventory(self, now_ms: float) -> RuntimeInventory: ...

    async def dispatch(self, command: DispatchCommand) -> DispatchAck: ...

    async def receive_completion(self, dispatch_id: str) -> AttemptCompletion: ...

    async def cancel(
        self,
        attempt_id: str,
        reason: str,
        now_ms: float,
    ) -> bool: ...

    async def describe(self, makespan_ms: float) -> tuple[dict[str, object], ...]: ...
