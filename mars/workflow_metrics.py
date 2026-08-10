"""Solver-independent workflow evaluation metrics.

The optimizer, runtime API, and benchmark all consume the same execution
profiles, but they operate on different report representations.  This module
keeps the post-run projection small and dependency-free: callers provide the
coordinator task rows and receive normalized, auditable metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from .domain.execution import task_resource_demand
from .domain.task import TaskInstance
from .domain.topology import NodeSnapshot, NodeSpec
from .domain.workflow import WorkflowSpec
from .profiling import ProfileCatalog


@dataclass(frozen=True)
class WorkflowEvaluationWeights:
    """Dimensionless secondary-objective weights used for reporting."""

    success: float = 1.0
    communication: float = 1.0
    utilization: float = 2.0

    def __post_init__(self) -> None:
        values = (self.success, self.communication, self.utilization)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("workflow evaluation weights must be finite and non-negative")


def evaluate_workflow_metrics(
    task_results: Iterable[Mapping[str, object]],
    workflow: WorkflowSpec,
    node_specs: Iterable[NodeSpec],
    node_snapshots: Iterable[NodeSnapshot],
    profiles: ProfileCatalog | None,
    *,
    weights: WorkflowEvaluationWeights = WorkflowEvaluationWeights(),
) -> dict[str, float]:
    """Evaluate selected placements and realized execution intervals.

    Snapshot utilization is treated as background load at the start of the
    run.  Task reservations are added to it; this matches the Web scene
    contract where ``initial_resources`` excludes work submitted by the run.
    """

    rows = tuple(task_results)
    task_by_id = {task.task_id: task for task in workflow.tasks}
    node_by_id = {node.node_id: node for node in node_specs}
    snapshot_by_id = {snapshot.node_id: snapshot for snapshot in node_snapshots}

    total_priority = float(sum(max(0, task.priority) for task in workflow.tasks))
    expected_reward = 0.0
    communication_ms = 0.0
    intervals_by_node: dict[
        str,
        list[tuple[float, float, tuple[float, float, float]]],
    ] = {node_id: [] for node_id in node_by_id}

    for row in rows:
        task_id = str(row.get("task_id", ""))
        task = task_by_id.get(task_id)
        target_node_id = str(row.get("target_node_id", ""))
        node = node_by_id.get(target_node_id)
        if task is None:
            continue

        if node is not None:
            profile = (
                profiles.lookup(task.spec.task_type, node.kind)
                if profiles is not None
                else None
            )
            success_probability = (
                1.0 - profile.failure_rate if profile is not None else 1.0
            )
            expected_reward += max(0, task.priority) * success_probability

        attempts = row.get("attempts", ())
        if not isinstance(attempts, (list, tuple)):
            continue
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            communication_ms += _number(attempt.get("communication_time_ms"))
            attempt_node_id = str(attempt.get("target_node_id", target_node_id))
            attempt_node = node_by_id.get(attempt_node_id)
            if attempt_node is None:
                continue
            start_ms = _number(attempt.get("start_time_ms"))
            finish_ms = _number(attempt.get("finish_time_ms"))
            compute_ms = _number(attempt.get("compute_time_ms"))
            compute_start_ms = max(start_ms, finish_ms - compute_ms)
            if finish_ms <= compute_start_ms:
                continue
            intervals_by_node[attempt_node_id].append(
                (
                    compute_start_ms,
                    finish_ms,
                    _resource_demand(task, attempt_node, profiles),
                )
            )

    peaks = {"cpu": 0.0, "gpu": 0.0, "memory": 0.0}
    for node_id, node in node_by_id.items():
        snapshot = snapshot_by_id.get(node_id, NodeSnapshot(node_id))
        intervals = intervals_by_node[node_id]
        boundaries = {
            0.0,
            *(point for start, finish, _ in intervals for point in (start, finish)),
        }
        for point in sorted(boundaries):
            active = [
                demand
                for start, finish, demand in intervals
                if start <= point < finish
            ]
            cpu = snapshot.cpu_util + sum(item[0] for item in active) / node.cpu_capacity
            gpu = snapshot.gpu_util
            if node.gpu_capacity > 0:
                gpu += sum(item[1] for item in active) / node.gpu_capacity
            elif any(item[1] > 0 for item in active):
                gpu = math.inf
            memory = (
                snapshot.memory_util
                + sum(item[2] for item in active) / node.memory_gb
            )
            peaks["cpu"] = max(peaks["cpu"], cpu)
            peaks["gpu"] = max(peaks["gpu"], gpu)
            peaks["memory"] = max(peaks["memory"], memory)

    expected_success_ratio = expected_reward / max(1.0, total_priority)
    communication_scale_ms = max(
        1.0,
        sum(max(1.0, task.spec.latency_budget_ms) for task in workflow.tasks),
    )
    normalized_communication = communication_ms / communication_scale_ms
    maximum_utilization = max(peaks.values(), default=0.0)
    objective = (
        -weights.success * expected_success_ratio
        + weights.communication * normalized_communication
        + weights.utilization * maximum_utilization
    )
    return {
        "expected_success_reward": round(expected_reward, 6),
        "expected_success_ratio": round(expected_success_ratio, 6),
        "communication_time_ms": round(communication_ms, 6),
        "normalized_communication": round(normalized_communication, 6),
        "peak_cpu_utilization": round(peaks["cpu"], 6),
        "peak_gpu_utilization": round(peaks["gpu"], 6),
        "peak_memory_utilization": round(peaks["memory"], 6),
        "maximum_resource_utilization": round(maximum_utilization, 6),
        "workflow_evaluation_objective": round(objective, 6),
    }


def _resource_demand(
    task: TaskInstance,
    node: NodeSpec,
    profiles: ProfileCatalog | None,
) -> tuple[float, float, float]:
    profile = (
        profiles.lookup(task.spec.task_type, node.kind)
        if profiles is not None
        else None
    )
    if profile is not None:
        default_cpu, default_gpu, _ = task_resource_demand(task, node)
        return (
            float(
                default_cpu
                if profile.cpu_units is None
                else profile.cpu_units
            ),
            float(
                default_gpu
                if profile.gpu_units is None
                else profile.gpu_units
            ),
            float(profile.peak_memory_mb) / 1024.0,
        )
    return task_resource_demand(task, node)


def _number(value: object) -> float:
    try:
        resolved = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return resolved if math.isfinite(resolved) else 0.0


__all__ = [
    "WorkflowEvaluationWeights",
    "evaluate_workflow_metrics",
]
