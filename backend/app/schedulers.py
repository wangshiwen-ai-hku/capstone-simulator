import logging
from typing import Dict, List, Tuple
import httpx

logger = logging.getLogger(__name__)

from .schemas import BenchmarkScene, NodeSpec, ResourceSnapshot, Workload


def node_map(scene: BenchmarkScene) -> Dict[str, NodeSpec]:
    return {n.id: n for n in scene.nodes}


def resource_map(scene: BenchmarkScene) -> Dict[str, ResourceSnapshot]:
    return {r.node_id: r for r in scene.initial_resources}


def candidates_for_task(scene: BenchmarkScene, task: Workload) -> List[NodeSpec]:
    nodes = node_map(scene)
    source = nodes[task.source_robot_id]
    if task.fallback_policy == "local_only" or task.safety_level >= 5:
        return [source]
    if task.fallback_policy == "local_preferred":
        return [source] + [n for n in scene.nodes if n.kind == "edge"]
    if task.fallback_policy == "edge_preferred":
        return [n for n in scene.nodes if n.kind == "edge"] + [source]
    return [source] + [n for n in scene.nodes if n.kind == "edge"]


def estimate_cost(node: NodeSpec, task: Workload, res: ResourceSnapshot, source_node: NodeSpec) -> Tuple[float, str]:
    util_penalty = 1.0 + 2.2 * max(res.cpu_util, res.gpu_util, res.memory_util)
    capacity = max(0.15, node.cpu_capacity + 1.5 * node.gpu_capacity)
    gpu_pressure = 1.0 + max(0.0, task.gpu_demand - node.gpu_capacity) * 3.0
    compute_ms = 100.0 * task.compute_demand / capacity * util_penalty * gpu_pressure

    if node.id == source_node.id:
        communication_ms = 0.0
        mode = "local"
    else:
        bandwidth = max(1.0, min(source_node.bandwidth_mbps, node.bandwidth_mbps))
        transfer_ms = task.data_size_mb * 8.0 / bandwidth * 1000.0
        communication_ms = source_node.base_latency_ms + node.base_latency_ms + res.network_latency_ms + transfer_ms
        mode = "edge"

    priority_bonus = (6 - task.priority) * 20.0
    safety_penalty = 10000.0 if task.safety_level >= 5 and node.id != source_node.id else 0.0
    thermal_penalty = max(0.0, res.temperature_c - 75.0) * 8.0
    cost = compute_ms + communication_ms + priority_bonus + safety_penalty + thermal_penalty
    return cost, mode


def choose_rule_based(scene: BenchmarkScene, task: Workload, resources: Dict[str, ResourceSnapshot]) -> Tuple[str, str, str]:
    candidates = candidates_for_task(scene, task)
    nodes = node_map(scene)
    source = nodes[task.source_robot_id]
    if task.safety_level >= 5 or task.fallback_policy == "local_only":
        return source.id, "local", "safety/local_only task must stay on source robot"
    source_res = resources[source.id]
    if source_res.cpu_util > 0.80 or source_res.gpu_util > 0.80 or task.compute_demand > 2.5:
        edge_candidates = [n for n in candidates if n.kind == "edge"]
        if edge_candidates:
            edge = min(edge_candidates, key=lambda n: max(resources[n.id].cpu_util, resources[n.id].gpu_util))
            return edge.id, "edge", "rule: offload when local load is high or workload is heavy"
    return source.id, "local", "rule: local execution acceptable"


def choose_local_first(scene: BenchmarkScene, task: Workload, resources: Dict[str, ResourceSnapshot]) -> Tuple[str, str, str]:
    return task.source_robot_id, "local", "local-first baseline"


def choose_edge_first(scene: BenchmarkScene, task: Workload, resources: Dict[str, ResourceSnapshot]) -> Tuple[str, str, str]:
    if task.fallback_policy == "local_only" or task.safety_level >= 5:
        return task.source_robot_id, "local", "edge-first constrained by local_only/safety"
    edges = [n for n in scene.nodes if n.kind == "edge"]
    if not edges:
        return task.source_robot_id, "local", "no edge nodes available"
    edge = min(edges, key=lambda n: max(resources[n.id].cpu_util, resources[n.id].gpu_util))
    return edge.id, "edge", "edge-first baseline"


def choose_greedy_cost(scene: BenchmarkScene, task: Workload, resources: Dict[str, ResourceSnapshot]) -> Tuple[str, str, str]:
    nodes = node_map(scene)
    source = nodes[task.source_robot_id]
    best = None
    for cand in candidates_for_task(scene, task):
        cost, mode = estimate_cost(cand, task, resources[cand.id], source)
        if best is None or cost < best[0]:
            best = (cost, cand, mode)
    assert best is not None
    return best[1].id, best[2], f"greedy min estimated cost={best[0]:.1f}"


async def choose_external(scene: BenchmarkScene, task: Workload, resources: Dict[str, ResourceSnapshot], url: str | None) -> Tuple[str, str, str]:
    if not url:
        tid, mode, reason = choose_greedy_cost(scene, task, resources)
        return tid, mode, "external URL missing; fallback to " + reason
    candidates = candidates_for_task(scene, task)
    payload = {
        "scene": scene.model_dump(),
        "task": task.model_dump(),
        "resources": {k: v.model_dump() for k, v in resources.items()},
        "candidates": [c.id for c in candidates],
    }
    try:
        logger.info(f"[External Scheduler] Calling URL={url} for task={task.id}")
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        target = data.get("target_node_id")
        logger.info(f"[External Scheduler] task={task.id} returned target={target}")
        if target in [c.id for c in candidates]:
            mode = data.get("mode") or ("local" if target == task.source_robot_id else "edge")
            return target, mode, "external: " + str(data.get("reason", "accepted"))
    except Exception as exc:
        logger.warning(f"[External Scheduler] Failed for task={task.id} with exception: {type(exc).__name__} - {str(exc)}")
        tid, mode, reason = choose_greedy_cost(scene, task, resources)
        return tid, mode, f"external failed ({type(exc).__name__}); fallback to {reason}"
    tid, mode, reason = choose_greedy_cost(scene, task, resources)
    return tid, mode, "external returned invalid target; fallback to " + reason
