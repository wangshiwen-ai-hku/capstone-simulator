"""Scheduling assignments, resource demand, and execution completion."""

from __future__ import annotations

import enum
from dataclasses import dataclass
import math

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
        # ``compute_demand`` is expressed in physical CPU cores. Do not scale
        # or clamp it to the target capacity: candidate generation must be
        # able to reject a four-core task on a one-core node.
        task.spec.compute_demand,
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
    output_size_mb: float = 0.0
    success_probability: float = 1.0

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
        if (
            not math.isfinite(self.output_size_mb)
            or self.output_size_mb < 0.0
        ):
            raise ValueError("assignment output_size_mb must be non-negative")
        if (
            not math.isfinite(self.success_probability)
            or not 0.0 <= self.success_probability <= 1.0
        ):
            raise ValueError(
                "assignment success_probability must be in [0, 1]"
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
