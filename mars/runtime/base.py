"""The single asynchronous boundary between the MARS control plane and agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..models import (
    ArtifactRef,
    Assignment,
    NodeSnapshot,
    NodeSpec,
    TaskInstance,
)


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
    input_artifacts: tuple[ArtifactRef, ...]
    seed: int
    inject_failure: bool = False


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
    finished_time_ms: float
    compute_time_ms: float
    energy_j: float
    outputs: tuple[ArtifactRef, ...]
    error_code: str = ""


@runtime_checkable
class RuntimePort(Protocol):
    """Coordinator-facing runtime contract implemented by every adapter.

    An adapter may execute agents in process or translate these operations to
    gRPC, DDS, or a partner runtime. Workflow submission remains an application
    concern and is intentionally outside this boundary.
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
