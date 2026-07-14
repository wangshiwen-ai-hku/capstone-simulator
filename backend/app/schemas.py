from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


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
    object_detection = "object_detection"
    segmentation = "segmentation"
    path_planning = "path_planning"
    data_compression = "data_compression"
    vla_inference = "vla_inference"
    llm_planning = "llm_planning"
    result_verification = "result_verification"
    map_fusion = "map_fusion"


class GenerateSceneRequest(BaseModel):
    scenario_type: ScenarioType = ScenarioType.warehouse
    custom_scene: Optional[str] = None
    robot_count: int = Field(default=4, ge=1, le=50)
    edge_count: int = Field(default=1, ge=1, le=8)
    task_categories: List[TaskCategory] = Field(default_factory=lambda: [
        TaskCategory.object_detection,
        TaskCategory.path_planning,
        TaskCategory.vla_inference,
    ])
    difficulty: Difficulty = Difficulty.medium
    seed: int = Field(default=7, ge=0)
    use_llm: bool = True


class NodeSpec(BaseModel):
    id: str
    kind: Literal["robot", "edge", "cloud"]
    display_name: str
    cpu_capacity: float = Field(gt=0)
    gpu_capacity: float = Field(ge=0)
    memory_gb: float = Field(gt=0)
    bandwidth_mbps: float = Field(gt=0)
    base_latency_ms: float = Field(ge=0)
    battery_wh: Optional[float] = None
    safety_capable: bool = True


class ResourceSnapshot(BaseModel):
    node_id: str
    cpu_util: float = Field(ge=0, le=1)
    gpu_util: float = Field(ge=0, le=1)
    memory_util: float = Field(ge=0, le=1)
    temperature_c: float
    power_w: float
    network_latency_ms: float


class Workload(BaseModel):
    id: str
    name: str
    source_robot_id: str
    task_type: str
    priority: int = Field(default=3, ge=1, le=5)
    compute_demand: float = Field(gt=0, description="Normalized compute units.")
    gpu_demand: float = Field(default=0.0, ge=0)
    latency_budget_ms: float = Field(gt=0)
    safety_level: int = Field(default=2, ge=1, le=5)
    model_requirement: str
    data_size_mb: float = Field(ge=0)
    bandwidth_requirement_mbps: float = Field(ge=0)
    energy_budget_j: float = Field(gt=0)
    fallback_policy: Literal["local_only", "edge_preferred", "local_preferred", "any"] = "any"
    result_verification: str
    arrival_time_ms: float = Field(default=0, ge=0)
    deadline_ms: float = Field(gt=0)
    dependencies: List[str] = Field(default_factory=list)
    expected_accuracy: float = Field(default=0.95, ge=0, le=1)


class BenchmarkScene(BaseModel):
    id: str
    title: str
    natural_language_description: str
    scenario_type: str
    difficulty: Difficulty
    nodes: List[NodeSpec]
    initial_resources: List[ResourceSnapshot]
    tasks: List[Workload]
    stressors: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)

    @field_validator("tasks")
    @classmethod
    def task_ids_unique(cls, tasks: List[Workload]) -> List[Workload]:
        ids = [t.id for t in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        return tasks


class SimulateRequest(BaseModel):
    scene: BenchmarkScene
    algorithm: Literal["rule_based", "local_first", "edge_first", "greedy_cost", "external"] = "greedy_cost"
    external_scheduler_url: Optional[str] = None
    network_jitter: float = Field(default=0.1, ge=0, le=1)
    resource_noise: float = Field(default=0.05, ge=0, le=0.5)
    seed: int = Field(default=7, ge=0)


class TaskRunResult(BaseModel):
    task_id: str
    task_name: str
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
    reason: str


class SimulationMetrics(BaseModel):
    task_count: int
    success_rate: float
    deadline_miss_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_energy_j: float
    total_energy_j: float
    bandwidth_mb: float
    makespan_ms: float
    edge_offload_ratio: float
    safety_violation_count: int


class SimulationResponse(BaseModel):
    algorithm: str
    metrics: SimulationMetrics
    task_results: List[TaskRunResult]
    node_utilization: Dict[str, float]
    logs: List[str]
