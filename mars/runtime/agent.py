"""Transport-neutral contract for one task execution node."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..domain.topology import NodeSpec
from .base import (
    AgentHeartbeat,
    AttemptCompletion,
    DispatchAck,
    DispatchCommand,
)


@runtime_checkable
class ExecutionAgent(Protocol):
    """Asynchronous lifecycle exposed by one local or remote execution node.

    The contract deliberately contains no gRPC, DDS, or simulator-specific
    types.  An in-process simulator, a gRPC client proxy, and a DDS client
    proxy can therefore all sit behind the same runtime-adapter boundary.
    """

    @property
    def node_spec(self) -> NodeSpec:
        """Return the static capabilities advertised by this node."""

        ...

    @property
    def registered(self) -> bool:
        """Report whether the node completed its registration lifecycle."""

        ...

    async def register(self, now_ms: float) -> bool:
        """Register the node, returning whether this call changed its state."""

        ...

    async def heartbeat(self, now_ms: float) -> AgentHeartbeat:
        """Return one fresh dynamic-state sample for this node."""

        ...

    async def dispatch(self, command: DispatchCommand) -> DispatchAck:
        """Accept or reject one validated task-dispatch command."""

        ...

    async def receive_completion(self, dispatch_id: str) -> AttemptCompletion:
        """Return the terminal result associated with ``dispatch_id``."""

        ...

    async def cancel(
        self,
        attempt_id: str,
        reason: str,
        now_ms: float,
    ) -> bool:
        """Cancel an active attempt if this node owns it."""

        ...
