"""DAG-aware placement, data-locality costing, and hard task-class constraints."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .dag import DagIndex
from .models import (
    ArtifactRef,
    Assignment,
    ExecutionMode,
    NodeKind,
    NodeSnapshot,
    TaskClass,
    TaskInstance,
)
from .profiling import ProfileCatalog


@dataclass(frozen=True)
class CandidateEstimate:
    node_id: str
    feasible: bool
    start_ms: float
    finish_ms: float
    compute_ms: float
    communication_ms: float
    energy_j: float
    input_locations: tuple[str, ...]
    reason: str = ""


def allowed_nodes(task: TaskInstance, nodes: Iterable[NodeSnapshot]) -> list[NodeSnapshot]:
    online = [node for node in nodes if node.online]
    source = next((node for node in online if node.node_id == task.source_node_id), None)
    if source is None:
        return []
    if task.spec.task_class is TaskClass.LOCAL_SAFETY:
        return [source] if source.kind is NodeKind.ROBOT and source.safety_capable else []
    edges = [node for node in online if node.kind is NodeKind.EDGE]
    if task.spec.task_class is TaskClass.REALTIME_OFFLOADABLE:
        return [source, *edges]
    # The current data plane contains source robots and on-premise edge nodes.
    # Cloud nodes are excluded from placement candidates.
    return [*edges, source] if task.spec.allow_local_fallback else edges


def estimate_candidate(
    task: TaskInstance,
    node: NodeSnapshot,
    *,
    ready_time_ms: float,
    node_available_ms: float,
    nodes: dict[str, NodeSnapshot],
    parent_artifacts: Iterable[ArtifactRef],
    profiles: ProfileCatalog | None = None,
) -> CandidateEstimate:
    if not node.online:
        return _infeasible(node, "node_offline")
    if task.spec.task_class is TaskClass.LOCAL_SAFETY and (
        node.node_id != task.source_node_id
        or node.kind is not NodeKind.ROBOT
        or not node.safety_capable
    ):
        return _infeasible(node, "local_safety_requires_safety_capable_source_robot")

    util_penalty = 1.0 + 2.2 * max(node.cpu_util, node.gpu_util, node.memory_util)
    profile = profiles.lookup(task.spec.task_type, node.kind) if profiles is not None else None
    if profile is not None and not profile.supported:
        return _infeasible(node, "profile_marks_task_unsupported")
    if profile is not None:
        compute_ms = profile.p95_ms * util_penalty
    else:
        capacity = max(0.15, node.cpu_capacity + 1.5 * node.gpu_capacity)
        gpu_pressure = 1.0 + max(0.0, task.spec.gpu_demand - node.gpu_capacity) * 2.5
        compute_ms = 100.0 * task.spec.compute_demand / capacity * util_penalty * gpu_pressure

    artifacts = tuple(parent_artifacts)
    transfers: list[tuple[str, float]]
    if artifacts:
        transfers = [(artifact.node_id, artifact.size_mb) for artifact in artifacts]
    else:
        transfers = [(task.source_node_id, task.spec.input_size_mb)]

    communication_ms = 0.0
    locations: list[str] = []
    for source_id, size_mb in transfers:
        locations.append(source_id)
        if source_id == node.node_id:
            continue
        source = nodes.get(source_id)
        if source is None or not source.online:
            return _infeasible(node, f"input_source_unavailable:{source_id}")
        bandwidth = max(1e-6, min(source.bandwidth_mbps, node.bandwidth_mbps))
        if task.spec.bandwidth_requirement_mbps and bandwidth < task.spec.bandwidth_requirement_mbps:
            return _infeasible(node, "bandwidth_below_requirement")
        communication_ms += size_mb * 8.0 / bandwidth * 1000.0
        communication_ms += source.base_latency_ms + node.base_latency_ms

    start_ms = max(ready_time_ms, node_available_ms)
    finish_ms = start_ms + communication_ms + compute_ms
    power = max(1.0, node.power_w)
    profiled_energy = profile.energy_j * util_penalty if profile is not None else compute_ms / 1000.0 * power
    energy_j = profiled_energy + communication_ms * 0.015
    return CandidateEstimate(
        node_id=node.node_id,
        feasible=True,
        start_ms=start_ms,
        finish_ms=finish_ms,
        compute_ms=compute_ms,
        communication_ms=communication_ms,
        energy_j=energy_j,
        input_locations=tuple(locations),
    )


def choose_assignment(
    task: TaskInstance,
    *,
    algorithm: str,
    ready_time_ms: float,
    node_available: dict[str, float],
    nodes: dict[str, NodeSnapshot],
    parent_artifacts: Iterable[ArtifactRef],
    critical_tail_ms: float = 0.0,
    profiles: ProfileCatalog | None = None,
) -> Assignment:
    candidates = allowed_nodes(task, nodes.values())
    estimates = [
        estimate_candidate(
            task,
            node,
            ready_time_ms=ready_time_ms,
            node_available_ms=node_available.get(node.node_id, 0.0),
            nodes=nodes,
            parent_artifacts=parent_artifacts,
            profiles=profiles,
        )
        for node in candidates
    ]
    feasible = [estimate for estimate in estimates if estimate.feasible]
    if not feasible:
        return Assignment(
            task_id=task.task_id,
            target_node_id="",
            execution_mode=ExecutionMode.DROP,
            estimated_start_ms=ready_time_ms,
            estimated_finish_ms=ready_time_ms,
            compute_ms=0.0,
            communication_ms=0.0,
            energy_j=0.0,
            reason="no feasible node under task-class and network constraints",
        )

    source_id = task.source_node_id
    if algorithm == "local_first":
        chosen = next((item for item in feasible if item.node_id == source_id), min(feasible, key=lambda x: x.finish_ms))
        reason = "local-first baseline constrained by the three-class contract"
    elif algorithm == "edge_first":
        chosen = next(
            (item for item in feasible if nodes[item.node_id].kind is NodeKind.EDGE),
            min(feasible, key=lambda x: x.finish_ms),
        )
        reason = "edge-first baseline constrained by safety and fallback rules"
    elif algorithm == "rule_based":
        source = nodes[source_id]
        edge = [item for item in feasible if nodes[item.node_id].kind is NodeKind.EDGE]
        should_offload = (
            task.spec.task_class is TaskClass.EDGE_HEAVY
            or source.cpu_util > 0.8
            or source.gpu_util > 0.8
            or task.spec.compute_demand > 2.5
        )
        chosen = min(edge, key=lambda x: x.finish_ms) if should_offload and edge else next(
            (item for item in feasible if item.node_id == source_id), min(feasible, key=lambda x: x.finish_ms)
        )
        reason = "three-class placement rule"
    elif algorithm == "dag_deadline":
        def dag_score(item: CandidateEstimate) -> tuple[float, float, float]:
            projected_workflow_finish = item.finish_ms + critical_tail_ms
            lateness = max(0.0, projected_workflow_finish - task.deadline_time_ms)
            locality_penalty = len(set(item.input_locations) - {item.node_id}) * 2.0
            return (lateness, projected_workflow_finish + locality_penalty, item.energy_j)

        chosen = min(feasible, key=dag_score)
        reason = "DAG deadline/critical-tail/data-locality minimum"
    else:
        chosen = min(feasible, key=lambda item: (item.finish_ms, item.energy_j))
        reason = "minimum estimated finish time and energy"

    node = nodes[chosen.node_id]
    mode = ExecutionMode.LOCAL if chosen.node_id == source_id else (
        ExecutionMode.EDGE if node.kind is NodeKind.EDGE else ExecutionMode.CLOUD
    )
    if mode is ExecutionMode.LOCAL and task.spec.task_class is TaskClass.EDGE_HEAVY:
        mode = ExecutionMode.FALLBACK_LOCAL
    return Assignment(
        task_id=task.task_id,
        target_node_id=chosen.node_id,
        execution_mode=mode,
        estimated_start_ms=chosen.start_ms,
        estimated_finish_ms=chosen.finish_ms,
        compute_ms=chosen.compute_ms,
        communication_ms=chosen.communication_ms,
        energy_j=chosen.energy_j,
        reason=reason,
        input_locations=chosen.input_locations,
    )


def critical_path(
    tasks: Iterable[TaskInstance], index: DagIndex
) -> tuple[tuple[str, ...], float, dict[str, float]]:
    """Return longest compute-weighted path, its cost and per-task tail cost."""
    task_by_id = {task.task_id: task for task in tasks}
    own_cost = {
        task_id: max(1.0, task_by_id[task_id].spec.compute_demand * 100.0)
        for task_id in index.topological_order
    }
    tail: dict[str, float] = {}
    successor: dict[str, str | None] = {}
    for task_id in reversed(index.topological_order):
        children = index.children[task_id]
        if not children:
            tail[task_id] = own_cost[task_id]
            successor[task_id] = None
        else:
            best = max(children, key=lambda child: tail[child])
            tail[task_id] = own_cost[task_id] + tail[best]
            successor[task_id] = best
    root = max(index.topological_order, key=lambda task_id: tail[task_id])
    path: list[str] = []
    cursor: str | None = root
    while cursor is not None:
        path.append(cursor)
        cursor = successor[cursor]
    # Scheduler wants the cost after the current task, not including it.
    critical_tail = {task_id: tail[task_id] - own_cost[task_id] for task_id in tail}
    return tuple(path), tail[root], critical_tail


def apply_load(node: NodeSnapshot, task: TaskInstance) -> NodeSnapshot:
    """Small deterministic utilization update used by simulation snapshots."""
    return replace(
        node,
        cpu_util=min(0.99, node.cpu_util * 0.96 + min(0.22, task.spec.compute_demand * 0.015)),
        gpu_util=min(0.99, node.gpu_util * 0.96 + min(0.28, task.spec.gpu_demand * 0.025)),
        memory_util=min(0.98, node.memory_util * 0.985 + task.spec.compute_demand * 0.004),
        temperature_c=min(96.0, node.temperature_c * 0.997 + task.spec.compute_demand * 0.09),
    )


def _infeasible(node: NodeSnapshot, reason: str) -> CandidateEstimate:
    return CandidateEstimate(
        node_id=node.node_id,
        feasible=False,
        start_ms=0.0,
        finish_ms=float("inf"),
        compute_ms=0.0,
        communication_ms=0.0,
        energy_j=0.0,
        input_locations=(),
        reason=reason,
    )
