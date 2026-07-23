"""Transport-neutral domain model for MARS.

Framework-specific request and wire objects are converted into these domain
types. Task constraints and DAG semantics are defined in this module.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Mapping


class TaskClass(str, enum.Enum):
    """Business/reporting categories with a legacy placement mapping.

    New workloads declare placement through ``PlacementConstraints``. These
    labels remain stable for API compatibility, metrics, and old scenes that
    do not yet provide an explicit placement contract.
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
    PEER = "peer"
    EDGE = "edge"
    CLOUD = "cloud"
    FALLBACK_LOCAL = "fallback_local"
    DROP = "drop"


class ResourceClass(str, enum.Enum):
    CPU = "cpu"
    GPU = "gpu"
    IO = "io"


@dataclass(frozen=True)
class PlacementConstraints:
    """Declarative placement contract independent of business task labels.

    ``TaskClass`` remains useful for reporting and legacy inputs. New scheduling
    decisions are governed by this contract so that adding a business category
    does not require another scheduler branch.
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
            raise ValueError(
                "replicable tasks must be stateless and idempotent"
            )


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


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    producer_task_id: str
    node_id: str
    size_mb: float
    uri: str = ""
    checksum: str = ""
    producer_port: str = "result"
    message_type: str = ""


@dataclass(frozen=True)
class NodeSpec:
    """Static node identity and declared execution capacity."""

    node_id: str
    kind: NodeKind
    cpu_capacity: float
    gpu_capacity: float
    memory_gb: float
    bandwidth_mbps: float
    base_latency_ms: float
    architecture: str = "generic"
    battery_capacity_wh: float | None = None
    safety_capable: bool = True
    capabilities: tuple[str, ...] = ()
    supported_models: tuple[str, ...] = ()
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id must be non-blank")
        capacities = (
            self.cpu_capacity,
            self.gpu_capacity,
            self.memory_gb,
            self.bandwidth_mbps,
            self.base_latency_ms,
        )
        if not all(math.isfinite(value) for value in capacities):
            raise ValueError("node capacity values must be finite")
        if (
            self.cpu_capacity <= 0
            or self.gpu_capacity < 0
            or self.memory_gb <= 0
            or self.bandwidth_mbps <= 0
            or self.base_latency_ms < 0
        ):
            raise ValueError("node capacities are outside valid ranges")
        if self.battery_capacity_wh is not None and (
            not math.isfinite(self.battery_capacity_wh)
            or self.battery_capacity_wh <= 0
        ):
            raise ValueError(
                "battery_capacity_wh must be positive when provided"
            )
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")


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
class NodeSnapshot:
    """Dynamic resource and health state reported for a registered node."""

    node_id: str
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    memory_util: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    network_latency_ms: float = 0.0
    online: bool = True

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("snapshot node_id must be non-blank")
        values = (
            self.cpu_util,
            self.gpu_util,
            self.memory_util,
            self.temperature_c,
            self.power_w,
            self.network_latency_ms,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("node snapshot values must be finite")
        if not all(
            0.0 <= value <= 1.0
            for value in (
                self.cpu_util,
                self.gpu_util,
                self.memory_util,
            )
        ):
            raise ValueError("node utilization values must be in [0, 1]")
        if self.power_w < 0 or self.network_latency_ms < 0:
            raise ValueError(
                "node power and network latency must be non-negative"
            )


@dataclass(frozen=True)
class LinkSpec:
    """Static declaration for one directed network link."""

    link_id: str
    source_node_id: str
    target_node_id: str
    bandwidth_mbps: float
    base_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.link_id.strip():
            raise ValueError("link_id must be non-blank")
        if not self.source_node_id.strip() or not self.target_node_id.strip():
            raise ValueError("link endpoints must be non-blank")
        if self.source_node_id == self.target_node_id:
            raise ValueError("network links must connect two different nodes")
        if not math.isfinite(self.bandwidth_mbps):
            raise ValueError("link bandwidth_mbps must be finite")
        if self.bandwidth_mbps <= 0:
            raise ValueError("link bandwidth_mbps must be positive")
        if (
            not math.isfinite(self.base_latency_ms)
            or self.base_latency_ms < 0
        ):
            raise ValueError("link base_latency_ms must be non-negative")


@dataclass(frozen=True)
class LinkSnapshot:
    """Dynamic state for a directed link."""

    link_id: str
    available_bandwidth_mbps: float
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    online: bool = True

    def __post_init__(self) -> None:
        if not self.link_id.strip():
            raise ValueError("link_id must be non-blank")
        values = (
            self.available_bandwidth_mbps,
            self.latency_ms,
            self.jitter_ms,
            self.packet_loss_rate,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("link snapshot values must be finite")
        if self.available_bandwidth_mbps < 0:
            raise ValueError(
                "link available_bandwidth_mbps must be non-negative"
            )
        if self.latency_ms < 0 or self.jitter_ms < 0:
            raise ValueError("link latency and jitter must be non-negative")
        if not 0.0 <= self.packet_loss_rate < 1.0:
            raise ValueError("link packet_loss_rate must be in [0, 1)")


@dataclass(frozen=True)
class TransferEstimate:
    """Estimated movement of one input artifact across a directed path."""

    transfer_id: str
    source_node_id: str
    target_node_id: str
    size_mb: float
    path_link_ids: tuple[str, ...]
    bottleneck_bandwidth_mbps: float
    transfer_time_ms: float
    feasible: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.size_mb,
                self.transfer_time_ms,
            )
        ):
            raise ValueError("transfer estimates must be finite")
        if self.size_mb < 0:
            raise ValueError("transfer size_mb must be non-negative")
        if self.transfer_time_ms < 0:
            raise ValueError("transfer_time_ms must be non-negative")
        if (
            self.feasible
            and self.size_mb > 0
            and self.source_node_id != self.target_node_id
        ):
            if not self.path_link_ids:
                raise ValueError("remote feasible transfers require a link path")
            if self.bottleneck_bandwidth_mbps <= 0:
                raise ValueError(
                    "remote feasible transfers require positive bandwidth"
                )


@dataclass(frozen=True)
class TransferReservation:
    """A planned interval during which a task owns its transfer path."""

    reservation_id: str
    epoch_id: str
    task_id: str
    transfer_id: str
    path_link_ids: tuple[str, ...]
    start_ms: float
    finish_ms: float
    size_mb: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.start_ms,
                self.finish_ms,
                self.size_mb,
            )
        ):
            raise ValueError(
                "transfer reservation values must be finite"
            )
        if self.finish_ms < self.start_ms:
            raise ValueError("transfer reservation cannot finish before it starts")
        if self.size_mb < 0:
            raise ValueError("transfer reservation size_mb must be non-negative")


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
        if self.artifact is not None and self.outputs:
            if len(self.outputs) != 1 or self.outputs[0] != self.artifact:
                raise ValueError("artifact and outputs describe different task outputs")
        elif self.artifact is not None:
            object.__setattr__(self, "outputs", (self.artifact,))
        elif len(self.outputs) == 1:
            object.__setattr__(self, "artifact", self.outputs[0])


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
