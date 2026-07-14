import logging
import random
from statistics import mean
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

import numpy as np

from .schemas import BenchmarkScene, ResourceSnapshot, SimulateRequest, SimulationMetrics, SimulationResponse, TaskRunResult
from .schedulers import choose_edge_first, choose_external, choose_greedy_cost, choose_local_first, choose_rule_based, estimate_cost, node_map


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _clone_resources(scene: BenchmarkScene) -> Dict[str, ResourceSnapshot]:
    return {r.node_id: ResourceSnapshot.model_validate(r.model_dump()) for r in scene.initial_resources}


def _execution_numbers(scene: BenchmarkScene, task, target_id: str, resources: Dict[str, ResourceSnapshot], rng: random.Random, network_jitter: float) -> Tuple[float, float, float, float]:
    nodes = node_map(scene)
    target = nodes[target_id]
    source = nodes[task.source_robot_id]
    res = resources[target_id]

    capacity = max(0.15, target.cpu_capacity + 1.5 * target.gpu_capacity)
    util_penalty = 1.0 + 2.2 * max(res.cpu_util, res.gpu_util, res.memory_util)
    gpu_pressure = 1.0 + max(0.0, task.gpu_demand - target.gpu_capacity) * 2.0
    compute_ms = 100.0 * task.compute_demand / capacity * util_penalty * gpu_pressure
    compute_ms *= rng.uniform(0.88, 1.20)

    if target.id == source.id:
        communication_ms = 0.0
        bandwidth_mb = 0.0
    else:
        bandwidth = max(1.0, min(source.bandwidth_mbps, target.bandwidth_mbps))
        jitter = max(0.0, rng.gauss(1.0, network_jitter))
        transfer_ms = task.data_size_mb * 8.0 / bandwidth * 1000.0 * jitter
        communication_ms = source.base_latency_ms + target.base_latency_ms + res.network_latency_ms * jitter + transfer_ms
        bandwidth_mb = task.data_size_mb

    power_w = max(1.0, res.power_w)
    energy_j = (compute_ms / 1000.0) * power_w + communication_ms * 0.015
    logger.info(f"[Calculation] task={task.id} target={target_id} compute_ms={compute_ms:.2f} communication_ms={communication_ms:.2f} energy_j={energy_j:.2f} bandwidth_mb={bandwidth_mb:.2f}")
    return compute_ms, communication_ms, energy_j, bandwidth_mb


def _update_resources(resources: Dict[str, ResourceSnapshot], target_id: str, task, resource_noise: float, rng: random.Random) -> None:
    r = resources[target_id]
    inc_cpu = min(0.25, 0.015 * task.compute_demand + rng.random() * resource_noise)
    inc_gpu = min(0.30, 0.020 * task.gpu_demand + rng.random() * resource_noise)
    r.cpu_util = min(0.99, max(0.0, r.cpu_util * 0.95 + inc_cpu))
    r.gpu_util = min(0.99, max(0.0, r.gpu_util * 0.96 + inc_gpu))
    r.memory_util = min(0.98, max(0.0, r.memory_util * 0.98 + 0.005 * task.compute_demand))
    r.temperature_c = min(96.0, r.temperature_c * 0.995 + 0.12 * task.compute_demand + rng.uniform(-0.5, 0.8))
    r.power_w = max(5.0, r.power_w * 0.99 + 2.0 * task.compute_demand + rng.uniform(-1.5, 2.5))
    r.network_latency_ms = max(1.0, r.network_latency_ms * rng.uniform(0.96, 1.07))
    logger.info(f"[Resource Update] target={target_id} after task={task.id} -> cpu={r.cpu_util:.2f} gpu={r.gpu_util:.2f} temp={r.temperature_c:.2f} power={r.power_w:.2f}")


async def run_simulation(req: SimulateRequest) -> SimulationResponse:
    logger.info(f"Starting simulation run with algorithm='{req.algorithm}', scene='{req.scene.id}', total_tasks={len(req.scene.tasks)}")
    rng = random.Random(req.seed)
    scene = req.scene
    resources = _clone_resources(scene)
    node_available_at: Dict[str, float] = {n.id: 0.0 for n in scene.nodes}
    task_finish: Dict[str, float] = {}
    results: List[TaskRunResult] = []
    logs: List[str] = []
    safety_violations = 0
    total_bandwidth_mb = 0.0
    offloaded = 0

    tasks = sorted(scene.tasks, key=lambda t: (t.arrival_time_ms, -t.priority, t.id))
    for task in tasks:
        if req.algorithm == "rule_based":
            target_id, mode, reason = choose_rule_based(scene, task, resources)
        elif req.algorithm == "local_first":
            target_id, mode, reason = choose_local_first(scene, task, resources)
        elif req.algorithm == "edge_first":
            target_id, mode, reason = choose_edge_first(scene, task, resources)
        elif req.algorithm == "external":
            target_id, mode, reason = await choose_external(scene, task, resources, req.external_scheduler_url)
        else:
            target_id, mode, reason = choose_greedy_cost(scene, task, resources)

        if task.safety_level >= 5 and target_id != task.source_robot_id:
            safety_violations += 1

        dependency_ready = 0.0
        for dep in task.dependencies:
            dependency_ready = max(dependency_ready, task_finish.get(dep, task.arrival_time_ms))
        ready_time = max(task.arrival_time_ms, dependency_ready)
        start_time = max(ready_time, node_available_at[target_id])
        queue_delay = max(0.0, start_time - task.arrival_time_ms)

        compute_ms, communication_ms, energy_j, bandwidth_mb = _execution_numbers(scene, task, target_id, resources, rng, req.network_jitter)
        finish = start_time + compute_ms + communication_ms
        node_available_at[target_id] = finish
        task_finish[task.id] = finish
        total_bandwidth_mb += bandwidth_mb
        if target_id != task.source_robot_id:
            offloaded += 1

        latency = finish - task.arrival_time_ms
        deadline_missed = finish > task.deadline_ms or latency > task.latency_budget_ms * 1.25
        thermal_fail = resources[target_id].temperature_c > 93
        accuracy_drop = 0.04 if target_id != task.source_robot_id and task.data_size_mb > 8 else 0.0
        success_probability = max(0.05, task.expected_accuracy - accuracy_drop - (0.25 if deadline_missed else 0.0) - (0.20 if thermal_fail else 0.0))
        success = rng.random() < success_probability
        if task.safety_level >= 5 and target_id != task.source_robot_id:
            success = False

        _update_resources(resources, target_id, task, req.resource_noise, rng)

        results.append(TaskRunResult(
            task_id=task.id,
            task_name=task.name,
            source_robot_id=task.source_robot_id,
            target_node_id=target_id,
            mode=mode,
            priority=task.priority,
            start_time_ms=round(start_time, 2),
            finish_time_ms=round(finish, 2),
            queue_delay_ms=round(queue_delay, 2),
            compute_time_ms=round(compute_ms, 2),
            communication_time_ms=round(communication_ms, 2),
            total_latency_ms=round(latency, 2),
            energy_j=round(energy_j, 2),
            deadline_missed=deadline_missed,
            success=success,
            reason=reason,
        ))
        logs.append(f"{task.id} -> {target_id} ({mode}); latency={latency:.1f}ms; {reason}")
        logger.info(f"[Task Scheduled] id={task.id} target={target_id} mode={mode} latency={latency:.1f}ms success={success} deadline_missed={deadline_missed} reason='{reason}'")

    latencies = [r.total_latency_ms for r in results]
    energies = [r.energy_j for r in results]
    success_rate = sum(1 for r in results if r.success) / max(1, len(results))
    miss_rate = sum(1 for r in results if r.deadline_missed) / max(1, len(results))
    makespan = max([r.finish_time_ms for r in results], default=0.0)

    busy_by_node = {n.id: 0.0 for n in scene.nodes}
    for r in results:
        busy_by_node[r.target_node_id] += r.compute_time_ms
    node_utilization = {k: round(v / max(1.0, makespan), 4) for k, v in busy_by_node.items()}

    metrics = SimulationMetrics(
        task_count=len(results),
        success_rate=round(success_rate, 4),
        deadline_miss_rate=round(miss_rate, 4),
        avg_latency_ms=round(mean(latencies) if latencies else 0.0, 2),
        p95_latency_ms=round(percentile(latencies, 95), 2),
        p99_latency_ms=round(percentile(latencies, 99), 2),
        avg_energy_j=round(mean(energies) if energies else 0.0, 2),
        total_energy_j=round(sum(energies), 2),
        bandwidth_mb=round(total_bandwidth_mb, 2),
        makespan_ms=round(makespan, 2),
        edge_offload_ratio=round(offloaded / max(1, len(results)), 4),
        safety_violation_count=safety_violations,
    )
    logger.info(f"Simulation completed. task_count={len(results)} success_rate={success_rate:.4f} makespan_ms={makespan:.2f} avg_latency_ms={metrics.avg_latency_ms} total_energy_j={metrics.total_energy_j}")
    return SimulationResponse(
        algorithm=req.algorithm,
        metrics=metrics,
        task_results=results,
        node_utilization=node_utilization,
        logs=logs,
    )
