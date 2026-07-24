"""Workflow topology, lifecycle policy, and progress state."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .task import TaskInstance


class WorkflowState(str, enum.Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FailurePolicy(str, enum.Enum):
    SKIP_DESCENDANTS = "skip_descendants"
    FAIL_FAST = "fail_fast"


@dataclass(frozen=True)
class DataEdge:
    """Bind one producer output port to one consumer input port."""

    producer_task: str
    producer_port: str
    consumer_task: str
    consumer_port: str
    message_type: str


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    tasks: tuple[TaskInstance, ...]
    deadline_time_ms: float = 0.0
    failure_policy: FailurePolicy = FailurePolicy.SKIP_DESCENDANTS
    metadata: Mapping[str, str] = field(default_factory=dict)
    data_edges: tuple[DataEdge, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "data_edges", tuple(self.data_edges))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )


@dataclass(frozen=True)
class WorkflowProgress:
    workflow_id: str
    state: WorkflowState
    total_tasks: int
    state_counts: Mapping[str, int]
    ready_task_ids: tuple[str, ...]
    critical_path: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_counts",
            MappingProxyType(dict(self.state_counts)),
        )
        object.__setattr__(
            self,
            "ready_task_ids",
            tuple(self.ready_task_ids),
        )
        object.__setattr__(
            self,
            "critical_path",
            tuple(self.critical_path),
        )


__all__ = [
    "DataEdge",
    "FailurePolicy",
    "WorkflowProgress",
    "WorkflowSpec",
    "WorkflowState",
]
