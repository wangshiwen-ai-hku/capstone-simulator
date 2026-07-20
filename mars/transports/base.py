"""Transport protocol implemented by MARS communication adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from ..models import Assignment, NodeKind, NodeSnapshot, TaskCompletion, TaskInstance, WorkflowSpec


@dataclass(frozen=True)
class NodeRegistration:
    node_id: str
    kind: NodeKind
    architecture: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class TransportCapabilities:
    discovery: bool
    reliable_control: bool
    best_effort_telemetry: bool
    feedback: bool
    cancellation: bool
    liveliness: bool


class SchedulerTransport(Protocol):
    """The only I/O surface the live control plane is allowed to call."""

    capabilities: TransportCapabilities

    async def register(self, registration: NodeRegistration) -> bool: ...

    async def publish_node_state(self, snapshot: NodeSnapshot) -> None: ...

    async def submit_workflow(self, workflow: WorkflowSpec) -> str: ...

    async def dispatch(self, task: TaskInstance, assignment: Assignment) -> str: ...

    async def cancel(self, task_id: str, reason: str) -> bool: ...

    def completions(self) -> AsyncIterator[TaskCompletion]: ...
