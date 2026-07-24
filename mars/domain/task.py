"""Task declarations, instances, placement constraints, and task states."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .topology import NodeKind


class TaskClass(str, enum.Enum):
    """Reporting cohorts with compatibility placement mappings.

    Placement behavior is declared through ``PlacementConstraints``.
    ``TaskClass`` supports aggregate metrics and scenes that omit an explicit
    placement contract.
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


class ResourceClass(str, enum.Enum):
    CPU = "cpu"
    GPU = "gpu"
    IO = "io"


@dataclass(frozen=True)
class PlacementConstraints:
    """Declarative placement contract independent of business task labels.

    ``TaskClass`` provides reporting and compatibility metadata. Scheduling
    decisions use this contract, so business categories do not require
    scheduler-specific branches.
    """

    pinned_node_id: str = ""
    allowed_node_kinds: tuple[NodeKind, ...] = ()
    preferred_node_kinds: tuple[NodeKind, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    allow_source_node: bool = True
    allow_other_robots: bool = False
    safety_required: bool = False
    allow_fallback: bool = True
    stateful: bool = False
    idempotent: bool = True
    splittable: bool = False
    replicable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_node_kinds",
            tuple(self.allowed_node_kinds),
        )
        object.__setattr__(
            self,
            "preferred_node_kinds",
            tuple(self.preferred_node_kinds),
        )
        object.__setattr__(
            self,
            "required_capabilities",
            tuple(self.required_capabilities),
        )
        if self.pinned_node_id and not self.pinned_node_id.strip():
            raise ValueError("pinned_node_id must be empty or non-blank")
        if len(self.allowed_node_kinds) != len(set(self.allowed_node_kinds)):
            raise ValueError("allowed_node_kinds must not contain duplicates")
        if len(self.preferred_node_kinds) != len(set(self.preferred_node_kinds)):
            raise ValueError("preferred_node_kinds must not contain duplicates")
        if not set(self.preferred_node_kinds).issubset(self.allowed_node_kinds):
            raise ValueError(
                "preferred_node_kinds must be a subset of allowed_node_kinds"
            )
        normalized_capabilities = tuple(
            capability.strip() for capability in self.required_capabilities
        )
        if any(not capability for capability in normalized_capabilities):
            raise ValueError("required_capabilities must be non-blank")
        if len(normalized_capabilities) != len(set(normalized_capabilities)):
            raise ValueError("required_capabilities must not contain duplicates")
        object.__setattr__(
            self,
            "required_capabilities",
            normalized_capabilities,
        )
        if self.replicable and (self.stateful or not self.idempotent):
            raise ValueError("replicable tasks must be stateless and idempotent")


@dataclass(frozen=True)
class DataPort:
    """A middleware-neutral, typed input or output of a task."""

    name: str
    message_type: str


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
    input_ports: tuple[DataPort, ...] = ()
    output_ports: tuple[DataPort, ...] = ()
    placement_constraints: PlacementConstraints | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_ports", tuple(self.input_ports))
        object.__setattr__(self, "output_ports", tuple(self.output_ports))


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dependency_task_ids",
            tuple(self.dependency_task_ids),
        )


TASK_CLASS_LABELS: dict[TaskClass, str] = {
    TaskClass.LOCAL_SAFETY: "Local safety reporting cohort",
    TaskClass.REALTIME_OFFLOADABLE: "Real-time offloadable reporting cohort",
    TaskClass.EDGE_HEAVY: "Edge-heavy reporting cohort",
}


def resolved_placement_constraints(task: TaskInstance) -> PlacementConstraints:
    """Return explicit constraints or a backwards-compatible legacy contract."""

    explicit = task.spec.placement_constraints
    if explicit is not None:
        return explicit
    if task.spec.task_class is TaskClass.LOCAL_SAFETY:
        return PlacementConstraints(
            pinned_node_id=task.source_node_id,
            allowed_node_kinds=(NodeKind.ROBOT,),
            preferred_node_kinds=(NodeKind.ROBOT,),
            required_capabilities=("local_safety",),
            allow_source_node=True,
            allow_other_robots=False,
            safety_required=True,
            allow_fallback=False,
            stateful=True,
            idempotent=False,
        )
    if task.spec.task_class is TaskClass.REALTIME_OFFLOADABLE:
        return PlacementConstraints(
            allowed_node_kinds=(NodeKind.EDGE,),
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=task.spec.allow_local_fallback,
        )
    return PlacementConstraints(
        allowed_node_kinds=(NodeKind.EDGE,),
        preferred_node_kinds=(NodeKind.EDGE,),
        allow_source_node=task.spec.allow_local_fallback,
        allow_other_robots=False,
        allow_fallback=task.spec.allow_local_fallback,
    )


def infer_task_class(task_type: str) -> TaskClass:
    """Infer a placement class when an input trace omits the explicit class."""

    normalized = task_type.lower()
    if normalized in {
        "obstacle_avoidance",
        "emergency_stop",
        "local_control",
    }:
        return TaskClass.LOCAL_SAFETY
    if normalized in {
        "localization",
        "environment_understanding",
        "object_detection",
        "image_classification",
        "segmentation",
        "semantic_segmentation",
        "path_planning",
        "local_planning",
        "result_verification",
        "yolo_inference",
    }:
        return TaskClass.REALTIME_OFFLOADABLE
    return TaskClass.EDGE_HEAVY


__all__ = [
    "DataPort",
    "PlacementConstraints",
    "ResourceClass",
    "TASK_CLASS_LABELS",
    "TERMINAL_STATES",
    "TaskClass",
    "TaskInstance",
    "TaskSpec",
    "TaskState",
    "infer_task_class",
    "resolved_placement_constraints",
]
