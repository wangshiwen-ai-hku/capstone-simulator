"""Deterministic benchmark-scene generation."""

from dataclasses import dataclass
import logging
import random
from typing import List

from mars.domain.task import TaskClass
from mars.synthetic_workloads import ExecutionTarget, load_default_synthetic_workloads

from .schemas import (
    BenchmarkScene,
    DataEdgeSpec,
    Difficulty,
    GenerateSceneRequest,
    LinkSnapshot,
    LinkSpec,
    NodeSpec,
    PlacementConstraintsSpec,
    PortSpec,
    RESOURCE_CONTRACT_VERSION,
    ResourceSnapshot,
    TaskCategory,
    Workload,
)


logger = logging.getLogger(__name__)


DIFFICULTY_FACTOR = {
    Difficulty.easy: 0.75,
    Difficulty.medium: 1.0,
    Difficulty.hard: 1.45,
    Difficulty.stress: 2.1,
}

ROBOT_HARDWARE = {
    "orin_nano": {
        "architecture": "jetson-orin-nano",
        "cpu_capacity": 6.0,
        "gpu_capacity": 67.0,
        "memory_gb": 8.0,
    },
    "orin_nx": {
        "architecture": "jetson-orin-nx",
        "cpu_capacity": 8.0,
        "gpu_capacity": 157.0,
        "memory_gb": 16.0,
    },
    "orin_agx": {
        "architecture": "jetson-agx-orin",
        "cpu_capacity": 12.0,
        "gpu_capacity": 275.0,
        "memory_gb": 32.0,
    },
}

SCENE_TEXT = {
    "warehouse": (
        "Warehouse mobile robots perform inspection, recognition, path "
        "planning, and material transport."
    ),
    "hospital": (
        "Hospital delivery robots navigate between wards and pharmacies "
        "while performing obstacle avoidance, recognition, and task planning."
    ),
    "campus": (
        "Campus robots perform delivery, inspection, map updates, and anomaly "
        "detection."
    ),
    "factory": (
        "Factory robots perform part recognition, quality inspection, path "
        "planning, and cooperative transport."
    ),
    "disaster": (
        "Rescue robots perform search, detection, and status reporting while "
        "network conditions and edge load vary."
    ),
    "custom": "Custom runtime scenario.",
}

CATEGORY_ALIASES = {
    TaskCategory.segmentation: "semantic_segmentation",
    TaskCategory.path_planning: "local_planning",
    TaskCategory.vla_inference: "local_llm_10b",
    TaskCategory.llm_planning: "local_llm_7b",
}


@dataclass(frozen=True)
class TaskTypeTemplate:
    """Web-scene semantics for one concrete task type."""

    reporting_class: TaskClass
    safety_level: int
    placement: PlacementConstraintsSpec

    @property
    def allow_local_fallback(self) -> bool:
        return (
            self.placement.allow_source_node
            and self.placement.allow_fallback
        )


def _placement(**values) -> PlacementConstraintsSpec:
    return PlacementConstraintsSpec(**values)


# Placement is declared per concrete task type. TaskClass remains a reporting
# cohort and is not used to construct these contracts.
TASK_TYPE_TEMPLATES: dict[str, TaskTypeTemplate] = {
    "obstacle_avoidance": TaskTypeTemplate(
        reporting_class=TaskClass.LOCAL_SAFETY,
        safety_level=5,
        placement=_placement(
            pin_to_source=True,
            allowed_node_kinds=["robot"],
            preferred_node_kinds=["robot"],
            required_capabilities=["local_safety"],
            allow_source_node=True,
            allow_other_robots=False,
            safety_required=True,
            allow_fallback=False,
            stateful=True,
            idempotent=False,
        ),
    ),
    "emergency_stop": TaskTypeTemplate(
        reporting_class=TaskClass.LOCAL_SAFETY,
        safety_level=5,
        placement=_placement(
            pin_to_source=True,
            allowed_node_kinds=["robot"],
            preferred_node_kinds=["robot"],
            required_capabilities=["local_safety"],
            allow_source_node=True,
            allow_other_robots=False,
            safety_required=True,
            allow_fallback=False,
            stateful=False,
            idempotent=True,
        ),
    ),
    "local_control": TaskTypeTemplate(
        reporting_class=TaskClass.LOCAL_SAFETY,
        safety_level=5,
        placement=_placement(
            pin_to_source=True,
            allowed_node_kinds=["robot"],
            preferred_node_kinds=["robot"],
            required_capabilities=["local_safety"],
            allow_source_node=True,
            allow_other_robots=False,
            safety_required=True,
            allow_fallback=False,
            stateful=True,
            idempotent=False,
        ),
    ),
    "localization": TaskTypeTemplate(
        reporting_class=TaskClass.REALTIME_OFFLOADABLE,
        safety_level=4,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["robot"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
            stateful=True,
            idempotent=False,
        ),
    ),
    "environment_understanding": TaskTypeTemplate(
        reporting_class=TaskClass.REALTIME_OFFLOADABLE,
        safety_level=3,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["edge"],
            required_capabilities=["cuda"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
        ),
    ),
    "object_detection": TaskTypeTemplate(
        reporting_class=TaskClass.REALTIME_OFFLOADABLE,
        safety_level=3,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["edge"],
            required_capabilities=["cuda"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
        ),
    ),
    "semantic_segmentation": TaskTypeTemplate(
        reporting_class=TaskClass.REALTIME_OFFLOADABLE,
        safety_level=3,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["edge"],
            required_capabilities=["cuda"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
        ),
    ),
    "local_planning": TaskTypeTemplate(
        reporting_class=TaskClass.REALTIME_OFFLOADABLE,
        safety_level=4,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["robot"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
            stateful=True,
            idempotent=False,
        ),
    ),
    "data_compression": TaskTypeTemplate(
        reporting_class=TaskClass.EDGE_HEAVY,
        safety_level=1,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["edge"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
        ),
    ),
    "local_llm_7b": TaskTypeTemplate(
        reporting_class=TaskClass.EDGE_HEAVY,
        safety_level=2,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["edge"],
            required_capabilities=["cuda"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
        ),
    ),
    "local_llm_10b": TaskTypeTemplate(
        reporting_class=TaskClass.EDGE_HEAVY,
        safety_level=2,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["edge"],
            required_capabilities=["cuda"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
        ),
    ),
    "result_verification": TaskTypeTemplate(
        reporting_class=TaskClass.REALTIME_OFFLOADABLE,
        safety_level=3,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["edge"],
            required_capabilities=["cuda"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
        ),
    ),
    "map_fusion": TaskTypeTemplate(
        reporting_class=TaskClass.EDGE_HEAVY,
        safety_level=2,
        placement=_placement(
            allowed_node_kinds=["robot", "edge"],
            preferred_node_kinds=["edge"],
            required_capabilities=["cuda"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=True,
            stateful=True,
            idempotent=False,
        ),
    ),
}


PROFILE_DEFAULTS = {
    "data_compression": dict(
        compute=0.5,
        gpu=0.0,
        latency=500,
        data=8.0,
        output=2.5,
        model="codec",
    ),
    "result_verification": dict(
        compute=1.4,
        gpu=0.5,
        latency=700,
        data=2.0,
        output=0.02,
        model="VLM verifier",
    ),
}

SYNTHETIC_CATALOG = load_default_synthetic_workloads()


def _workload_name(category: TaskCategory) -> str:
    return CATEGORY_ALIASES.get(category, category.value)


def placement_constraints_for(
    task_type: str,
) -> PlacementConstraintsSpec:
    """Return an independent placement contract for a known task type."""
    try:
        template = TASK_TYPE_TEMPLATES[task_type]
    except KeyError as exc:
        raise KeyError(
            f"no placement template for task type: {task_type}"
        ) from exc
    return template.placement.model_copy(deep=True)


def apply_absolute_resource_contract(
    scene: BenchmarkScene,
    robot_hardware: str,
) -> BenchmarkScene:
    """Normalize generated facts to hardware TOPS and fixed task demand."""

    hardware = ROBOT_HARDWARE[robot_hardware]
    for node in scene.nodes:
        if node.kind == "robot":
            node.architecture = str(hardware["architecture"])
            node.cpu_capacity = float(hardware["cpu_capacity"])
            node.gpu_capacity = float(hardware["gpu_capacity"])
            node.memory_gb = float(hardware["memory_gb"])
        elif node.kind == "edge":
            node.gpu_capacity = 500.0
    for task in scene.tasks:
        try:
            workload = SYNTHETIC_CATALOG.get(task.task_type)
        except KeyError:
            continue
        task.gpu_demand = workload.accelerator_demand_tops
    return scene


def _category_definition(
    category: TaskCategory,
) -> tuple[
    dict,
    list[PortSpec],
    list[PortSpec],
    PlacementConstraintsSpec,
]:
    """Resolve one UI category to a scheduler profile and typed ports."""
    task_type = _workload_name(category)
    template = TASK_TYPE_TEMPLATES[task_type]
    placement = placement_constraints_for(task_type)
    try:
        workload = SYNTHETIC_CATALOG.get(task_type)
    except KeyError:
        definition = {
            **PROFILE_DEFAULTS[task_type],
            "task_class": template.reporting_class,
            "safety": template.safety_level,
            "local_fallback": template.allow_local_fallback,
        }
        return definition, [], [], placement

    profile = workload.profile_for(ExecutionTarget.ORIN)
    definition = {
        "task_class": template.reporting_class,
        "compute": profile.resources.cpu_cores,
        "gpu": workload.accelerator_demand_tops,
        "latency": profile.latency.p95_ms * 1.25,
        "data": profile.input_size_mb.typical,
        "output": profile.output_size_mb.typical,
        "safety": template.safety_level,
        "model": workload.model_variant,
        "local_fallback": template.allow_local_fallback,
    }
    inputs = [PortSpec(name=port.name, message_type=port.semantic_type) for port in workload.inputs]
    outputs = [PortSpec(name=port.name, message_type=port.semantic_type) for port in workload.outputs]
    return definition, inputs, outputs, placement


def build_deterministic_scene(
    req: GenerateSceneRequest,
    *,
    preflight: bool = True,
) -> BenchmarkScene:
    rng = random.Random(req.seed)
    factor = DIFFICULTY_FACTOR[req.difficulty]
    robot_hardware = ROBOT_HARDWARE[req.robot_hardware]
    scene_id = f"scene_{req.scenario_type.value}_{req.seed:04d}"
    scene_name = req.custom_scene or SCENE_TEXT[req.scenario_type.value]

    logger.info(
        "Building deterministic scene %s (type=%s difficulty=%s robots=%d "
        "edges=%d)",
        scene_id,
        req.scenario_type.value,
        req.difficulty.value,
        req.robot_count,
        req.edge_count,
    )

    nodes: List[NodeSpec] = []
    resources: List[ResourceSnapshot] = []

    for i in range(req.robot_count):
        rid = f"robot_{i+1}"
        nodes.append(NodeSpec(
            id=rid,
            kind="robot",
            display_name=f"Jetson Orin Robot {i+1}",
            architecture=robot_hardware["architecture"],
            # Resource profiles express CPU demand in cores.  Keep node
            # capacity in the same unit so feasibility is target-specific.
            cpu_capacity=robot_hardware["cpu_capacity"],
            gpu_capacity=robot_hardware["gpu_capacity"],
            memory_gb=robot_hardware["memory_gb"],
            bandwidth_mbps=rng.uniform(40, 120) / factor,
            base_latency_ms=rng.uniform(6, 22) * factor,
            battery_wh=rng.uniform(50, 120),
            safety_capable=True,
            capabilities=["cpu", "cuda", "tensorrt", "local_safety"],
            max_concurrency=1,
        ))
        resources.append(ResourceSnapshot(
            node_id=rid,
            cpu_util=min(0.88, rng.uniform(0.10, 0.45) * factor),
            gpu_util=min(0.92, rng.uniform(0.05, 0.50) * factor),
            memory_util=min(0.85, rng.uniform(0.15, 0.55) * factor),
            temperature_c=rng.uniform(45, 68) + 5 * (factor - 1),
            power_w=rng.uniform(9, 28) * factor,
            network_latency_ms=rng.uniform(6, 25) * factor,
        ))

    for i in range(req.edge_count):
        eid = "edge_pc" if req.edge_count == 1 else f"edge_pc_{i+1}"
        nodes.append(NodeSpec(
            id=eid,
            kind="edge",
            display_name=f"Edge GPU Agent {i+1}",
            architecture="x86_64-cuda",
            cpu_capacity=16.0,
            # Synthetic edge accelerator capacity, expressed in the same
            # sparse INT8 TOPS unit as task demand.
            gpu_capacity=500.0,
            memory_gb=64,
            bandwidth_mbps=rng.uniform(500, 1200) / (0.7 + factor * 0.3),
            base_latency_ms=rng.uniform(12, 35) * factor,
            battery_wh=None,
            safety_capable=False,
            capabilities=["cpu", "cuda", "high_memory"],
            max_concurrency=2,
        ))
        resources.append(ResourceSnapshot(
            node_id=eid,
            cpu_util=min(0.95, rng.uniform(0.20, 0.55) * factor),
            gpu_util=min(0.95, rng.uniform(0.25, 0.70) * factor),
            memory_util=min(0.90, rng.uniform(0.20, 0.65) * factor),
            temperature_c=rng.uniform(52, 76) + 4 * (factor - 1),
            power_w=rng.uniform(90, 240) * factor,
            network_latency_ms=rng.uniform(12, 40) * factor,
        ))

    base_tasks = max(req.robot_count * 2, len(req.task_categories) * req.robot_count)
    if req.difficulty == Difficulty.stress:
        base_tasks = int(base_tasks * 1.5)
    elif req.difficulty == Difficulty.hard:
        base_tasks = int(base_tasks * 1.25)

    tasks: List[Workload] = []
    data_edges: List[DataEdgeSpec] = []
    available_outputs: dict[str, dict[str, tuple[str, str]]] = {}
    for idx in range(base_tasks):
        source_robot_id = f"robot_{idx % req.robot_count + 1}"
        stage_index = idx // req.robot_count
        category = req.task_categories[stage_index % len(req.task_categories)]
        (
            d,
            input_ports,
            output_ports,
            placement_constraints,
        ) = _category_definition(category)
        jitter = rng.uniform(0.75, 1.35)
        compute = d["compute"] * factor * jitter
        data_size = d["data"] * factor * rng.uniform(0.7, 1.5)
        latency_budget = max(40, d["latency"] / (0.9 if req.difficulty in [Difficulty.hard, Difficulty.stress] else 1.0))
        priority = 5 if d["safety"] >= 5 else rng.randint(1, 5)
        arrival = stage_index * rng.uniform(60, 220) / factor
        deadline = arrival + latency_budget * rng.uniform(1.0, 1.8)
        task_id = f"task_{idx+1:03d}"
        dependencies: List[str] = []
        robot_outputs = available_outputs.setdefault(source_robot_id, {})
        for port in input_ports:
            producer = robot_outputs.get(port.message_type)
            if producer is None:
                continue
            producer_task, producer_port = producer
            data_edges.append(DataEdgeSpec(
                producer_task=producer_task,
                producer_port=producer_port,
                consumer_task=task_id,
                consumer_port=port.name,
                message_type=port.message_type,
            ))
            if producer_task not in dependencies:
                dependencies.append(producer_task)
        tasks.append(Workload(
            id=task_id,
            name=f"{_workload_name(category).replace('_', ' ')} #{idx+1}",
            source_robot_id=source_robot_id,
            task_type=_workload_name(category),
            task_class=d["task_class"],
            priority=priority,
            compute_demand=round(compute, 3),
            # A workload has fixed absolute accelerator demand. Difficulty
            # and RNG must not resize the work to match a selected board.
            gpu_demand=round(d["gpu"], 3),
            latency_budget_ms=round(latency_budget, 1),
            safety_level=d["safety"],
            model_requirement=d["model"],
            data_size_mb=round(data_size, 3),
            output_size_mb=round(d["output"] * factor * rng.uniform(0.85, 1.15), 3),
            bandwidth_requirement_mbps=round(
                min(
                    80.0,
                    max(
                        1.0,
                        data_size
                        * 8
                        / max(0.1, latency_budget / 1000)
                        * 0.25,
                    ),
                ),
                2,
            ),
            energy_budget_j=round(20 + compute * 90 + data_size * 2.0, 1),
            allow_local_fallback=d["local_fallback"],
            placement_constraints=placement_constraints,
            result_verification=(
                "Validate the task result, latency budget, deadline, and "
                "task-specific completion status."
            ),
            arrival_time_ms=round(arrival, 1),
            deadline_ms=round(deadline, 1),
            dependencies=dependencies,
            stage_index=stage_index,
            expected_accuracy=round(max(0.72, 0.97 - 0.03 * factor - rng.random() * 0.04), 3),
            input_ports=input_ports,
            output_ports=output_ports,
        ))
        for port in output_ports:
            robot_outputs[port.message_type] = (task_id, port.name)

    stressors = []
    if req.difficulty in [Difficulty.hard, Difficulty.stress]:
        stressors.extend(["edge GPU contention", "network jitter", "priority preemption", "deadline pressure"])
    if req.difficulty == Difficulty.stress:
        stressors.extend(["bursty VLA requests", "robot temperature throttling", "temporary bandwidth degradation"])

    logger.info(
        "Generated %d tasks and %d nodes for scene %s",
        len(tasks),
        len(nodes),
        scene_id,
    )

    links, link_snapshots = _directed_links(nodes, resources)
    scene = BenchmarkScene(
        id=scene_id,
        resource_contract_version=RESOURCE_CONTRACT_VERSION,
        title=f"{req.scenario_type.value.title()} multi-robot scheduling scenario",
        natural_language_description=scene_name,
        scenario_type=req.scenario_type.value,
        difficulty=req.difficulty,
        nodes=nodes,
        initial_resources=resources,
        links=links,
        link_snapshots=link_snapshots,
        tasks=tasks,
        data_edges=data_edges,
        workflow_id=f"workflow_{scene_id}",
        workflow_deadline_ms=round(max((task.deadline_ms for task in tasks), default=0.0) * 1.1, 2),
        stressors=stressors,
        success_criteria=[
            "success_rate >= 0.95 for easy/medium and >= 0.85 for hard/stress",
            "P95 latency remains within declared task budgets",
            "source-pinned safety tasks execute only on their source robots",
            "the run reports deadline, latency, energy, and placement metrics",
        ],
    )
    apply_absolute_resource_contract(scene, req.robot_hardware)
    if preflight:
        # Import lazily to keep scene schemas independent from scheduler wiring.
        from .schedulability import ensure_generated_scene_schedulable

        ensure_generated_scene_schedulable(scene)
    return scene


def _directed_links(
    nodes: List[NodeSpec],
    resources: List[ResourceSnapshot],
) -> tuple[List[LinkSpec], List[LinkSnapshot]]:
    resource_by_id = {item.node_id: item for item in resources}
    links: List[LinkSpec] = []
    snapshots: List[LinkSnapshot] = []
    for source in nodes:
        source_state = resource_by_id[source.id]
        for target in nodes:
            if source.id == target.id:
                continue
            target_state = resource_by_id[target.id]
            link_id = f"link:{source.id}->{target.id}"
            bandwidth = min(
                source.bandwidth_mbps,
                target.bandwidth_mbps,
            )
            links.append(
                LinkSpec(
                    id=link_id,
                    source_node_id=source.id,
                    target_node_id=target.id,
                    bandwidth_mbps=bandwidth,
                    base_latency_ms=(
                        source.base_latency_ms + target.base_latency_ms
                    ),
                )
            )
            snapshots.append(
                LinkSnapshot(
                    link_id=link_id,
                    available_bandwidth_mbps=bandwidth,
                    latency_ms=(
                        source_state.network_latency_ms
                        + target_state.network_latency_ms
                    ),
                    online=source_state.online and target_state.online,
                )
            )
    return links, snapshots
