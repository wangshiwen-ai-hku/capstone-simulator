"""Scheduling assignments, resource demand, and execution completion."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .artifact import ArtifactRef
from .task import TaskInstance, TaskState
from .topology import NodeSpec


class ExecutionMode(str, enum.Enum):
    LOCAL = "local"
    PEER = "peer"
    EDGE = "edge"
    CLOUD = "cloud"
    FALLBACK_LOCAL = "fallback_local"
    DROP = "drop"


def task_resource_demand(
    task: TaskInstance,
    node: NodeSpec,
) -> tuple[float, float, float]:
    """Return the canonical CPU, GPU, and memory reservation for a task."""

    return (
        min(
            node.cpu_capacity,
            max(0.05, task.spec.compute_demand * 0.15),
        ),
        max(0.0, task.spec.gpu_demand),
        min(
            node.memory_gb,
            max(0.05, task.spec.compute_demand * 0.08),
        ),
    )


@dataclass(frozen=True)
class Assignment:
    task_id: str
    target_node_id: str
    execution_mode: ExecutionMode
    estimated_start_ms: float
    estimated_finish_ms: float
    compute_ms: float
    communication_ms: float
    energy_j: float
    reason: str
    input_locations: tuple[str, ...] = ()
    transfer_link_ids: tuple[str, ...] = ()
    optimizer_id: str = ""
    epoch_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_locations",
            tuple(self.input_locations),
        )
        object.__setattr__(
            self,
            "transfer_link_ids",
            tuple(self.transfer_link_ids),
        )


@dataclass(frozen=True)
class TaskCompletion:
    task_id: str
    ok: bool
    state: TaskState
    finished_time_ms: float
    artifact: ArtifactRef | None = None
    error_code: str = ""
    outputs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the legacy singular artifact into the multi-output form."""

        object.__setattr__(self, "outputs", tuple(self.outputs))
        if self.artifact is not None and self.outputs:
            if len(self.outputs) != 1 or self.outputs[0] != self.artifact:
                raise ValueError("artifact and outputs describe different task outputs")
        elif self.artifact is not None:
            object.__setattr__(self, "outputs", (self.artifact,))
        elif len(self.outputs) == 1:
            object.__setattr__(self, "artifact", self.outputs[0])


__all__ = [
    "Assignment",
    "ExecutionMode",
    "TaskCompletion",
    "task_resource_demand",
]
