"""Transport-neutral domain model for MARS.

Framework-specific request and wire objects are adapted into these types so
that task constraints and DAG semantics have one authoritative definition.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping


class TaskClass(str, enum.Enum):
    """Placement classes used by the scheduler.

    LOCAL_SAFETY
        Hard real-time control/safety work which must execute on the source
        safety-capable robot (for example obstacle avoidance).
    REALTIME_OFFLOADABLE
        Latency-sensitive inference which may run locally or at the edge (for
        example YOLO object detection).
    EDGE_HEAVY
        Compute/data-heavy, softer-deadline work which prefers the edge but can
        use an explicitly configured local fallback.
    """

    LOCAL_SAFETY = "local_safety"
    REALTIME_OFFLOADABLE = "realtime_offloadable"
    EDGE_HEAVY = "edge_heavy"


class TaskState(str, enum.Enum):
    BLOCKED = "blocked"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    DROPPED = "dropped"


TERMINAL_STATES = frozenset(
    {
        TaskState.SUCCEEDED,
        TaskState.FAILED,
        TaskState.TIMEOUT,
        TaskState.SKIPPED,
        TaskState.DROPPED,
    }
)


class WorkflowState(str, enum.Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FailurePolicy(str, enum.Enum):
    SKIP_DESCENDANTS = "skip_descendants"
    FAIL_FAST = "fail_fast"


class NodeKind(str, enum.Enum):
    ROBOT = "robot"
    EDGE = "edge"
    CLOUD = "cloud"


class ExecutionMode(str, enum.Enum):
    LOCAL = "local"
    EDGE = "edge"
    CLOUD = "cloud"
    FALLBACK_LOCAL = "fallback_local"
    DROP = "drop"


class ResourceClass(str, enum.Enum):
    CPU = "cpu"
    GPU = "gpu"
    IO = "io"


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    task_class: TaskClass
    compute_demand: float = 1.0
    gpu_demand: float = 0.0
    latency_budget_ms: float = 1000.0
    model_requirement: str = ""
    input_size_mb: float = 0.0
    output_size_mb: float = 0.1
    bandwidth_requirement_mbps: float = 0.0
    energy_budget_j: float = 0.0
    dominant_resource: ResourceClass = ResourceClass.GPU
    allow_local_fallback: bool = True


@dataclass(frozen=True)
class TaskInstance:
    task_id: str
    workflow_id: str
    name: str
    source_node_id: str
    spec: TaskSpec
    dependency_task_ids: tuple[str, ...] = ()
    priority: int = 3
    stage_index: int = 0
    arrival_time_ms: float = 0.0
    deadline_time_ms: float = 1000.0
    expected_accuracy: float = 0.95
    input_ref: str = ""


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    tasks: tuple[TaskInstance, ...]
    deadline_time_ms: float = 0.0
    failure_policy: FailurePolicy = FailurePolicy.SKIP_DESCENDANTS
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    producer_task_id: str
    node_id: str
    size_mb: float
    uri: str = ""
    checksum: str = ""


@dataclass(frozen=True)
class NodeSnapshot:
    node_id: str
    kind: NodeKind
    cpu_capacity: float
    gpu_capacity: float
    memory_gb: float
    bandwidth_mbps: float
    base_latency_ms: float
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    memory_util: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    online: bool = True
    safety_capable: bool = True


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


@dataclass(frozen=True)
class TaskCompletion:
    task_id: str
    ok: bool
    state: TaskState
    finished_time_ms: float
    artifact: ArtifactRef | None = None
    error_code: str = ""


@dataclass(frozen=True)
class WorkflowProgress:
    workflow_id: str
    state: WorkflowState
    total_tasks: int
    state_counts: Mapping[str, int]
    ready_task_ids: tuple[str, ...]
    critical_path: tuple[str, ...]


TASK_CLASS_LABELS: dict[TaskClass, str] = {
    TaskClass.LOCAL_SAFETY: "端侧安全关键任务",
    TaskClass.REALTIME_OFFLOADABLE: "可卸载实时推理任务",
    TaskClass.EDGE_HEAVY: "边缘优先重计算任务",
}


def infer_task_class(task_type: str) -> TaskClass:
    """Infer a placement class when an input trace omits the explicit class."""
    normalized = task_type.lower()
    if normalized in {"obstacle_avoidance", "emergency_stop", "local_control"}:
        return TaskClass.LOCAL_SAFETY
    if normalized in {
        "object_detection",
        "image_classification",
        "segmentation",
        "path_planning",
        "result_verification",
        "yolo_inference",
    }:
        return TaskClass.REALTIME_OFFLOADABLE
    return TaskClass.EDGE_HEAVY
