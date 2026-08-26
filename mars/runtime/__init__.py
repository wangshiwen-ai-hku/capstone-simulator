"""Runtime contract and adapters for the MARS control plane."""

from .base import (
    AgentHeartbeat,
    AttemptCompletion,
    DispatchAck,
    DispatchCommand,
    RuntimeCapabilities,
    RuntimeInventory,
    RuntimePort,
)
from .inprocess import ExecutionInvocation, InProcessRuntime
from .grpc import GrpcRuntimeAdapter

__all__ = [
    "AgentHeartbeat",
    "AttemptCompletion",
    "DispatchAck",
    "DispatchCommand",
    "ExecutionInvocation",
    "GrpcRuntimeAdapter",
    "InProcessRuntime",
    "RuntimeCapabilities",
    "RuntimeInventory",
    "RuntimePort",
]
