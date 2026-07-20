from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from edgesched.models import infer_task_class


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


class TaskClass(str, Enum):
    local_safety = "local_safety"
    realtime_offloadable = "realtime_offloadable"
    edge_heavy = "edge_heavy"


class FailurePolicy(str, Enum):
    skip_descendants = "skip_descendants"
    fail_fast = "fail_fast"


class GenerateSceneRequest(BaseModel):
    scenario_type: ScenarioType = ScenarioType.warehouse
    custom_scene: Optional[str] = None
    robot_count: int = Field(default=4, ge=1, le=50)
    edge_count: int = Field(default=1, ge=1, le=8)
    task_categories: List[TaskCategory] = Field(default_factory=lambda: [
        TaskCategory.obstacle_avoidance,
        TaskCategory.object_detection,
        TaskCategory.path_planning,
        TaskCategory.vla_inference,
    ])
    difficulty: Difficulty = Difficulty.medium
    seed: int = Field(default=7, ge=0)
    use_llm: bool = True

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
    task_class: Optional[TaskClass] = None
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
    fallback_policy: Literal["local_only", "edge_preferred", "local_preferred", "any"] = "any"
    result_verification: str
    arrival_time_ms: float = Field(default=0, ge=0)
    deadline_ms: float = Field(gt=0)
    dependencies: List[str] = Field(default_factory=list)
    stage_index: int = Field(default=0, ge=0)
    expected_accuracy: float = Field(default=0.95, ge=0, le=1)

    @model_validator(mode="after")
    def infer_legacy_task_class(self):
        if self.task_class is None:
            self.task_class = TaskClass(infer_task_class(self.task_type).value)
        if self.task_class == TaskClass.local_safety:
            self.fallback_policy = "local_only"
            self.safety_level = 5
        return self


class BenchmarkScene(BaseModel):
    id: str
    title: str
    natural_language_description: str
    scenario_type: str
    difficulty: Difficulty
    nodes: List[NodeSpec]
    initial_resources: List[ResourceSnapshot]
    tasks: List[Workload]
    workflow_id: str = ""
    workflow_deadline_ms: float = Field(default=0.0, ge=0)
    failure_policy: FailurePolicy = FailurePolicy.skip_descendants
    stressors: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)

    @field_validator("tasks")
    @classmethod
    def task_ids_unique(cls, tasks: List[Workload]) -> List[Workload]:
        ids = [t.id for t in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        return tasks

    @model_validator(mode="after")
    def dag_is_valid(self):
        if not self.workflow_id:
            self.workflow_id = f"workflow_{self.id}"
        task_ids = {task.id for task in self.tasks}
        indegree = {task.id: len(task.dependencies) for task in self.tasks}
        children = {task.id: [] for task in self.tasks}
        for task in self.tasks:
            if len(task.dependencies) != len(set(task.dependencies)):
                raise ValueError(f"task {task.id} has duplicate dependencies")
            if task.id in task.dependencies:
                raise ValueError(f"task {task.id} depends on itself")
            missing = [dep for dep in task.dependencies if dep not in task_ids]
            if missing:
                raise ValueError(f"task {task.id} has missing dependencies: {missing}")
            for parent in task.dependencies:
                children[parent].append(task.id)
        queue = [task.id for task in self.tasks if indegree[task.id] == 0]
        visited = 0
        while queue:
            task_id = queue.pop(0)
            visited += 1
            for child in children[task_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if visited != len(self.tasks):
            raise ValueError("task dependencies must form a DAG (cycle detected)")
        if self.workflow_deadline_ms <= 0:
            self.workflow_deadline_ms = max((task.deadline_ms for task in self.tasks), default=0.0)
        return self


class SimulateRequest(BaseModel):
    scene: BenchmarkScene
    algorithm: Literal["dag_deadline", "rule_based", "local_first", "edge_first", "greedy_cost", "external"] = "dag_deadline"
    external_scheduler_url: Optional[str] = None
    network_jitter: float = Field(default=0.1, ge=0, le=1)
    resource_noise: float = Field(default=0.05, ge=0, le=0.5)
    seed: int = Field(default=7, ge=0)


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


class WorkflowSummary(BaseModel):
    workflow_id: str
    state: str
    failure_policy: str
    deadline_time_ms: float
    deadline_missed: bool
    state_counts: Dict[str, int]
    critical_path: List[str]


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
