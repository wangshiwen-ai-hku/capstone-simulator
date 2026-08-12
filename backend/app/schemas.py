from enum import Enum
import math
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from mars.domain.task import TaskClass, infer_task_class
from mars.domain.workflow import FailurePolicy


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"
    stress = "stress"


class ScenarioType(str, Enum):
    warehouse = "warehouse"
    hospital = "hospital"
    campus = "campus"
    factory = "factory"
    disaster = "disaster"
    custom = "custom"


class TaskCategory(str, Enum):
    obstacle_avoidance = "obstacle_avoidance"
    emergency_stop = "emergency_stop"
    local_control = "local_control"
    localization = "localization"
    environment_understanding = "environment_understanding"
    object_detection = "object_detection"
    segmentation = "segmentation"
    semantic_segmentation = "semantic_segmentation"
    path_planning = "path_planning"
    local_planning = "local_planning"
    data_compression = "data_compression"
    vla_inference = "vla_inference"
    llm_planning = "llm_planning"
    local_llm_7b = "local_llm_7b"
    local_llm_10b = "local_llm_10b"
    result_verification = "result_verification"
    map_fusion = "map_fusion"


class GenerateSceneRequest(BaseModel):
    scenario_type: ScenarioType = ScenarioType.warehouse
    custom_scene: Optional[str] = None
    robot_count: int = Field(default=2, ge=1, le=50)
    edge_count: int = Field(default=1, ge=0, le=8)
    task_categories: List[TaskCategory] = Field(default_factory=lambda: [
        TaskCategory.localization,
        TaskCategory.environment_understanding,
        TaskCategory.object_detection,
        TaskCategory.semantic_segmentation,
        TaskCategory.local_planning,
        TaskCategory.obstacle_avoidance,
        TaskCategory.local_control,
        TaskCategory.local_llm_7b,
    ])
    difficulty: Difficulty = Difficulty.medium
    seed: int = Field(default=7, ge=0)
    use_llm: bool = False

    @field_validator("task_categories")
    @classmethod
    def task_categories_non_empty(cls, categories: List[TaskCategory]) -> List[TaskCategory]:
        if not categories:
            raise ValueError("at least one task category is required")
        return categories


class NodeSpec(BaseModel):
    id: str
    kind: Literal["robot", "edge", "cloud"]
    display_name: str
    architecture: str = "generic"
    cpu_capacity: float = Field(gt=0)
    gpu_capacity: float = Field(ge=0)
    memory_gb: float = Field(gt=0)
    bandwidth_mbps: float = Field(gt=0)
    base_latency_ms: float = Field(ge=0)
    battery_wh: Optional[float] = None
    safety_capable: bool = True
    capabilities: List[str] = Field(default_factory=list)
    supported_models: List[str] = Field(default_factory=list)
    max_concurrency: int = Field(default=1, ge=1)


class ResourceSnapshot(BaseModel):
    node_id: str
    cpu_util: float = Field(ge=0, le=1)
    gpu_util: float = Field(ge=0, le=1)
    memory_util: float = Field(ge=0, le=1)
    temperature_c: float
    power_w: float
    network_latency_ms: float
    online: bool = True
    remaining_energy_j: Optional[float] = Field(default=None, ge=0)


class PlacementConstraintsSpec(BaseModel):
    """Web-facing declarative placement contract.

    ``pin_to_source`` is resolved to a concrete node id by the MARS adapter so
    reusable workload templates do not need to know a robot id in advance.
    """

    pinned_node_id: str = ""
    pin_to_source: bool = False
    allowed_node_kinds: List[Literal["robot", "edge", "cloud"]] = Field(
        default_factory=list
    )
    preferred_node_kinds: List[Literal["robot", "edge", "cloud"]] = Field(
        default_factory=list
    )
    required_capabilities: List[str] = Field(default_factory=list)
    allow_source_node: bool = True
    allow_other_robots: bool = False
    safety_required: bool = False
    allow_fallback: bool = True
    stateful: bool = False
    idempotent: bool = True
    splittable: bool = False
    replicable: bool = False

    @field_validator("pinned_node_id", mode="before")
    @classmethod
    def normalize_empty_pin(cls, value):
        """Accept JSON null as the wire representation of no explicit pin."""
        return "" if value is None else value

    @model_validator(mode="after")
    def validate_contract(self):
        if self.pinned_node_id and self.pin_to_source:
            raise ValueError(
                "pinned_node_id and pin_to_source are mutually exclusive"
            )
        if len(self.allowed_node_kinds) != len(set(self.allowed_node_kinds)):
            raise ValueError("allowed_node_kinds must not contain duplicates")
        if len(self.preferred_node_kinds) != len(
            set(self.preferred_node_kinds)
        ):
            raise ValueError(
                "preferred_node_kinds must not contain duplicates"
            )
        if not set(self.preferred_node_kinds).issubset(
            self.allowed_node_kinds
        ):
            raise ValueError(
                "preferred_node_kinds must be a subset of allowed_node_kinds"
            )
        normalized = [item.strip() for item in self.required_capabilities]
        if any(not item for item in normalized):
            raise ValueError("required_capabilities must be non-blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "required_capabilities must not contain duplicates"
            )
        self.required_capabilities = normalized
        if self.replicable and (self.stateful or not self.idempotent):
            raise ValueError(
                "replicable tasks must be stateless and idempotent"
            )
        return self


class LinkSpec(BaseModel):
    id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    bandwidth_mbps: float = Field(gt=0)
    base_latency_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_endpoints(self):
        if self.source_node_id == self.target_node_id:
            raise ValueError("network links must connect different nodes")
        return self


class LinkSnapshot(BaseModel):
    link_id: str = Field(min_length=1)
    available_bandwidth_mbps: float = Field(ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    jitter_ms: float = Field(default=0.0, ge=0)
    packet_loss_rate: float = Field(default=0.0, ge=0, lt=1)
    online: bool = True


class PortSpec(BaseModel):
    name: str = Field(min_length=1)
    message_type: str = Field(min_length=1)


class DataEdgeSpec(BaseModel):
    producer_task: str = Field(min_length=1)
    producer_port: str = Field(min_length=1)
    consumer_task: str = Field(min_length=1)
    consumer_port: str = Field(min_length=1)
    message_type: str = Field(min_length=1)


class Workload(BaseModel):
    id: str
    name: str
    source_robot_id: str
    task_type: str
    task_class: Optional[TaskClass] = Field(
        default=None,
        description=(
            "Optional reporting cohort. PlacementConstraintsSpec is the "
            "authoritative scheduling contract."
        ),
    )
    priority: int = Field(default=3, ge=1, le=5)
    compute_demand: float = Field(gt=0, description="Normalized compute units.")
    gpu_demand: float = Field(default=0.0, ge=0)
    latency_budget_ms: float = Field(gt=0)
    safety_level: int = Field(default=2, ge=1, le=5)
    model_requirement: str
    data_size_mb: float = Field(ge=0)
    output_size_mb: float = Field(default=0.1, ge=0)
    bandwidth_requirement_mbps: float = Field(ge=0)
    energy_budget_j: float = Field(gt=0)
    allow_local_fallback: Optional[bool] = None
    placement_constraints: Optional[PlacementConstraintsSpec] = Field(
        default=None,
        description=(
            "Authoritative node eligibility, preference, safety, and "
            "execution-semantics contract."
        ),
    )
    result_verification: str
    arrival_time_ms: float = Field(default=0, ge=0)
    deadline_ms: float = Field(gt=0)
    dependencies: List[str] = Field(default_factory=list)
    stage_index: int = Field(default=0, ge=0)
    expected_accuracy: float = Field(default=0.95, ge=0, le=1)
    input_ports: List[PortSpec] = Field(default_factory=list)
    output_ports: List[PortSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def apply_compatibility_defaults(self):
        """Normalize legacy scenes without overriding explicit placement."""
        if self.task_class is None:
            self.task_class = infer_task_class(self.task_type)

        if self.placement_constraints is not None:
            # Keep the legacy field as a derived compatibility projection so
            # it cannot contradict the authoritative placement contract.
            self.allow_local_fallback = (
                self.placement_constraints.allow_source_node
                and self.placement_constraints.allow_fallback
            )
            if self.placement_constraints.safety_required:
                self.safety_level = 5
            return self

        # Compatibility path for imported scenes that predate explicit
        # placement contracts.
        if self.allow_local_fallback is None:
            self.allow_local_fallback = (
                self.task_class is not TaskClass.LOCAL_SAFETY
            )
        if self.task_class is TaskClass.LOCAL_SAFETY:
            self.safety_level = 5
            if self.placement_constraints is None:
                self.allow_local_fallback = False
        return self


class BenchmarkScene(BaseModel):
    id: str
    title: str
    natural_language_description: str
    scenario_type: str
    difficulty: Difficulty
    nodes: List[NodeSpec]
    initial_resources: List[ResourceSnapshot]
    links: Optional[List[LinkSpec]] = None
    link_snapshots: Optional[List[LinkSnapshot]] = None
    tasks: List[Workload]
    data_edges: List[DataEdgeSpec] = Field(default_factory=list)
    workflow_id: str = ""
    workflow_deadline_ms: float = Field(default=0.0, ge=0)
    failure_policy: FailurePolicy = FailurePolicy.SKIP_DESCENDANTS
    stressors: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    generation_source: Literal[
        "deterministic",
        "llm",
        "deterministic_fallback",
    ] = "deterministic"
    generation_note: str = ""
    trace_id: Optional[str] = Field(
        default=None,
        description=(
            "Opaque scene-generation trace identifier used to associate "
            "later simulation and runtime calls."
        ),
    )

    @field_validator("tasks")
    @classmethod
    def task_ids_unique(cls, tasks: List[Workload]) -> List[Workload]:
        ids = [t.id for t in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        return tasks

    @model_validator(mode="after")
    def apply_workflow_defaults(self):
        if not self.workflow_id:
            self.workflow_id = f"workflow_{self.id}"
        if self.workflow_deadline_ms <= 0:
            self.workflow_deadline_ms = max((task.deadline_ms for task in self.tasks), default=0.0)
        if (self.links is None) != (self.link_snapshots is None):
            raise ValueError(
                "links and link_snapshots must both be provided or omitted"
            )
        node_ids = {node.id for node in self.nodes}
        if self.links is not None and self.link_snapshots is not None:
            link_ids = [link.id for link in self.links]
            snapshot_ids = [
                snapshot.link_id for snapshot in self.link_snapshots
            ]
            if len(link_ids) != len(set(link_ids)):
                raise ValueError("link ids must be unique")
            if len(snapshot_ids) != len(set(snapshot_ids)):
                raise ValueError("link snapshot ids must be unique")
            if set(link_ids) != set(snapshot_ids):
                raise ValueError(
                    "every link requires exactly one link snapshot"
                )
            endpoints: set[tuple[str, str]] = set()
            for link in self.links:
                if (
                    link.source_node_id not in node_ids
                    or link.target_node_id not in node_ids
                ):
                    raise ValueError(
                        f"link {link.id} references an unknown node"
                    )
                endpoint = (
                    link.source_node_id,
                    link.target_node_id,
                )
                if endpoint in endpoints:
                    raise ValueError(
                        "parallel links for the same directed endpoints are "
                        "not supported"
                    )
                endpoints.add(endpoint)
        for task in self.tasks:
            constraints = task.placement_constraints
            if (
                constraints is not None
                and constraints.pinned_node_id
                and constraints.pinned_node_id not in node_ids
            ):
                raise ValueError(
                    f"task {task.id} pins to unknown node "
                    f"{constraints.pinned_node_id}"
                )
        return self


class SimulateRequest(BaseModel):
    scene: BenchmarkScene
    algorithm: Literal["binary_offload", "dag_deadline", "rule_based", "local_first", "edge_first", "greedy_cost"] = "dag_deadline"
    optimizer_options: Dict[str, float] = Field(default_factory=dict)
    beta: Optional[float] = Field(
        default=None,
        ge=0,
        deprecated=True,
        description=(
            "Deprecated normalized alias for binary_offload "
            "optimizer_options.communication_weight; ignored by other "
            "algorithms."
        ),
    )
    network_jitter: float = Field(default=0.1, ge=0, le=1)
    resource_noise: float = Field(default=0.05, ge=0, le=0.5)
    seed: int = Field(default=7, ge=0)

    @field_validator("optimizer_options")
    @classmethod
    def validate_optimizer_options(
        cls,
        options: Dict[str, float],
    ) -> Dict[str, float]:
        return _validate_optimizer_options(options)


class RuntimeWorkflowRequest(BaseModel):
    scene: BenchmarkScene
    algorithm: Literal["binary_offload", "dag_deadline", "rule_based", "local_first", "edge_first", "greedy_cost"] = "dag_deadline"
    optimizer_options: Dict[str, float] = Field(default_factory=dict)
    beta: Optional[float] = Field(
        default=None,
        ge=0,
        deprecated=True,
        description=(
            "Deprecated normalized alias for binary_offload "
            "optimizer_options.communication_weight; ignored by other "
            "algorithms."
        ),
    )
    seed: int = Field(default=7, ge=0)
    max_attempts: int = Field(default=2, ge=1, le=5)
    inject_first_failure: bool = False
    failure_task_type: str = "local_llm_7b"
    deterministic: bool = True

    @field_validator("optimizer_options")
    @classmethod
    def validate_optimizer_options(
        cls,
        options: Dict[str, float],
    ) -> Dict[str, float]:
        return _validate_optimizer_options(options)


def _validate_optimizer_options(
    options: Dict[str, float],
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in options.items():
        name = key.strip()
        if not name:
            raise ValueError("optimizer option names must be non-blank")
        resolved = float(value)
        if not math.isfinite(resolved) or resolved < 0:
            raise ValueError(
                "optimizer option values must be finite and non-negative"
            )
        normalized[name] = resolved
    return normalized


class TaskRunResult(BaseModel):
    task_id: str
    workflow_id: str
    task_name: str
    task_class: str
    stage_index: int
    dependencies: List[str]
    source_robot_id: str
    target_node_id: str
    mode: str
    priority: int
    start_time_ms: float
    finish_time_ms: float
    queue_delay_ms: float
    compute_time_ms: float
    communication_time_ms: float
    total_latency_ms: float
    energy_j: float
    deadline_missed: bool
    success: bool
    state: str
    reason: str
    input_locations: List[str] = Field(default_factory=list)
    output_ref: str = ""


class SimulationMetrics(BaseModel):
    task_count: int
    success_rate: float
    deadline_miss_rate: float
    executed_deadline_miss_rate: float = 0.0
    required_task_on_time_rate: float = 0.0
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_energy_j: float
    total_energy_j: float
    bandwidth_mb: float
    makespan_ms: float
    edge_offload_ratio: float
    safety_violation_count: int
    skipped_task_count: int
    workflow_success_rate: float
    critical_path_ms: float
    dag_depth: int
    total_solver_time_ms: float = 0.0
    max_solver_time_ms: float = 0.0
    scheduling_epoch_count: int = 0
    expected_success_reward: float = 0.0
    expected_success_ratio: float = 0.0
    communication_time_ms: float = 0.0
    normalized_communication: float = 0.0
    peak_cpu_utilization: float = 0.0
    peak_gpu_utilization: float = 0.0
    peak_memory_utilization: float = 0.0
    maximum_resource_utilization: float = 0.0
    workflow_evaluation_objective: float = 0.0
    fallback_count: int = 0


class WorkflowSummary(BaseModel):
    workflow_id: str
    state: str
    failure_policy: str
    deadline_time_ms: float
    deadline_missed: bool
    state_counts: Dict[str, int]
    critical_path: List[str]
    scheduling: Dict[str, object] = Field(default_factory=dict)
    requested_algorithm: Optional[str] = None
    optimizer_options: Dict[str, float] = Field(default_factory=dict)
    metric_schema_version: Optional[str] = None


class TaskClassMetrics(BaseModel):
    task_count: int
    success_rate: float
    avg_latency_ms: float
    edge_offload_ratio: float


class DagView(BaseModel):
    valid: bool
    topological_order: List[str]
    levels: Dict[str, int]
    edges: List[Dict[str, str]]


class SimulationResponse(BaseModel):
    algorithm: str
    metrics: SimulationMetrics
    task_results: List[TaskRunResult]
    node_utilization: Dict[str, float]
    logs: List[str]
    workflow: WorkflowSummary
    task_class_summary: Dict[str, TaskClassMetrics]
    dag: DagView
    transport: Dict[str, object]
