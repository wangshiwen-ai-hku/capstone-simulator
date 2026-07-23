"""Deterministic benchmark-scene generation."""

import logging
import random
from typing import List

from mars.models import TaskClass
from mars.synthetic_workloads import ExecutionTarget, load_default_synthetic_workloads

logger = logging.getLogger(__name__)

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
    ResourceSnapshot,
    TaskCategory,
    Workload,
)


DIFFICULTY_FACTOR = {
    Difficulty.easy: 0.75,
    Difficulty.medium: 1.0,
    Difficulty.hard: 1.45,
    Difficulty.stress: 2.1,
}

SCENE_TEXT = {
    "warehouse": "同一仓库内，多台移动机器人执行货架巡检、包裹识别、路径规划和抓取/搬运任务。",
    "hospital": "医院楼层内，配送机器人需要在病区和药房之间移动，同时执行避障、物品识别和任务规划。",
    "campus": "校园环境中，多台机器人在楼宇之间执行递送、巡检、地图更新和异常检测。",
    "factory": "工厂产线附近，机器人进行零件识别、质量检测、路径规划和协同搬运。",
    "disaster": "灾害救援模拟中，机器人在网络不稳定和边缘节点负载较高的情况下进行搜索、检测和汇报。",
    "custom": "用户自定义场景。",
}

CATEGORY_DEFAULTS = {
    TaskCategory.obstacle_avoidance: dict(task_class=TaskClass.LOCAL_SAFETY, compute=0.6, gpu=0.1, latency=60, data=0.3, output=0.02, safety=5, model="tiny-safety-cnn", local_fallback=False),
    TaskCategory.object_detection: dict(task_class=TaskClass.REALTIME_OFFLOADABLE, compute=1.2, gpu=0.6, latency=300, data=2.0, output=0.08, safety=3, model="yolo/rt-detr", local_fallback=True),
    TaskCategory.segmentation: dict(task_class=TaskClass.REALTIME_OFFLOADABLE, compute=1.6, gpu=0.8, latency=450, data=3.0, output=0.6, safety=3, model="segformer/sam-lite", local_fallback=True),
    TaskCategory.path_planning: dict(task_class=TaskClass.REALTIME_OFFLOADABLE, compute=1.0, gpu=0.1, latency=250, data=0.5, output=0.03, safety=4, model="astar/mpc", local_fallback=True),
    TaskCategory.data_compression: dict(task_class=TaskClass.EDGE_HEAVY, compute=0.5, gpu=0.0, latency=500, data=8.0, output=2.5, safety=1, model="codec", local_fallback=True),
    TaskCategory.vla_inference: dict(task_class=TaskClass.EDGE_HEAVY, compute=4.6, gpu=2.4, latency=1400, data=6.0, output=0.05, safety=3, model="7B-10B VLA", local_fallback=True),
    TaskCategory.llm_planning: dict(task_class=TaskClass.EDGE_HEAVY, compute=3.2, gpu=1.2, latency=2000, data=1.2, output=0.03, safety=2, model="LLM planner", local_fallback=True),
    TaskCategory.result_verification: dict(task_class=TaskClass.REALTIME_OFFLOADABLE, compute=1.4, gpu=0.5, latency=700, data=2.0, output=0.02, safety=3, model="VLM verifier", local_fallback=True),
    TaskCategory.map_fusion: dict(task_class=TaskClass.EDGE_HEAVY, compute=2.2, gpu=0.6, latency=1200, data=12.0, output=4.0, safety=2, model="SLAM/map fusion", local_fallback=True),
}

CATEGORY_ALIASES = {
    TaskCategory.segmentation: "semantic_segmentation",
    TaskCategory.path_planning: "local_planning",
    TaskCategory.vla_inference: "local_llm_10b",
    TaskCategory.llm_planning: "local_llm_7b",
}

SYNTHETIC_CATALOG = load_default_synthetic_workloads()


def _workload_name(category: TaskCategory) -> str:
    return CATEGORY_ALIASES.get(category, category.value)


def _category_definition(category: TaskCategory) -> tuple[dict, list[PortSpec], list[PortSpec]]:
    """Resolve one UI category to a scheduler profile and typed ports."""
    task_type = _workload_name(category)
    try:
        workload = SYNTHETIC_CATALOG.get(task_type)
    except KeyError:
        return CATEGORY_DEFAULTS[category], [], []

    profile = workload.profile_for(ExecutionTarget.ORIN)
    definition = {
        "task_class": workload.task_class,
        "compute": profile.resources.cpu_cores + 1.5 * profile.resources.gpu_units,
        "gpu": profile.resources.gpu_units,
        "latency": profile.latency.p95_ms * 1.25,
        "data": profile.input_size_mb.typical,
        "output": profile.output_size_mb.typical,
        "safety": 5 if workload.task_class is TaskClass.LOCAL_SAFETY else 3,
        "model": workload.model_variant,
        "local_fallback": workload.task_class is not TaskClass.LOCAL_SAFETY,
    }
    inputs = [PortSpec(name=port.name, message_type=port.semantic_type) for port in workload.inputs]
    outputs = [PortSpec(name=port.name, message_type=port.semantic_type) for port in workload.outputs]
    return definition, inputs, outputs


def build_deterministic_scene(req: GenerateSceneRequest) -> BenchmarkScene:
    rng = random.Random(req.seed)
    factor = DIFFICULTY_FACTOR[req.difficulty]
    scene_id = f"scene_{req.scenario_type.value}_{req.seed:04d}"
    scene_name = req.custom_scene or SCENE_TEXT[req.scenario_type.value]
    
    logger.info(f"Building deterministic scene '{scene_id}' (type={req.scenario_type.value}, difficulty={req.difficulty.value}, robots={req.robot_count}, edges={req.edge_count})")

    nodes: List[NodeSpec] = []
    resources: List[ResourceSnapshot] = []

    for i in range(req.robot_count):
        rid = f"robot_{i+1}"
        nodes.append(NodeSpec(
            id=rid,
            kind="robot",
            display_name=f"Jetson Orin Robot {i+1}",
            architecture="jetson-orin",
            cpu_capacity=1.0,
            gpu_capacity=1.0,
            memory_gb=16,
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
        eid = f"edge_pc" if req.edge_count == 1 else f"edge_pc_{i+1}"
        nodes.append(NodeSpec(
            id=eid,
            kind="edge",
            display_name=f"Edge GPU Agent {i+1}",
            architecture="x86_64-cuda",
            cpu_capacity=5.0,
            gpu_capacity=4.0,
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
        d, input_ports, output_ports = _category_definition(category)
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
            gpu_demand=round(min(1.0, d["gpu"] * factor * jitter), 3),
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
            placement_constraints=_placement_contract(
                d["task_class"],
                d["local_fallback"],
            ),
            result_verification="Check returned result, latency budget, deadline and task-specific success flag.",
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

    logger.info(f"Generated {len(tasks)} tasks and {len(nodes)} nodes for scene '{scene_id}'")

    links, link_snapshots = _directed_links(nodes, resources)
    return BenchmarkScene(
        id=scene_id,
        title=f"{req.scenario_type.value.title()} multi-robot scheduling benchmark",
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
            "success_rate >= 0.95 for easy/medium, >= 0.85 for hard/stress",
            "P95 latency below workload latency budgets where possible",
            "no local-safety task is offloaded from its source robot",
            "deadline, latency, energy and placement metrics are reported for the run",
        ],
    )


def _placement_contract(
    task_class: TaskClass,
    allow_local_fallback: bool,
) -> PlacementConstraintsSpec:
    if task_class is TaskClass.LOCAL_SAFETY:
        return PlacementConstraintsSpec(
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
        )
    if task_class is TaskClass.REALTIME_OFFLOADABLE:
        return PlacementConstraintsSpec(
            allowed_node_kinds=["edge"],
            allow_source_node=True,
            allow_other_robots=False,
            allow_fallback=allow_local_fallback,
        )
    return PlacementConstraintsSpec(
        allowed_node_kinds=["edge"],
        preferred_node_kinds=["edge"],
        allow_source_node=allow_local_fallback,
        allow_other_robots=False,
        allow_fallback=allow_local_fallback,
    )


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
