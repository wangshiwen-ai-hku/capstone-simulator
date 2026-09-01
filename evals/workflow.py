"""Canonical post-run evaluation for a complete MARS run artifact."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean
from typing import Iterable, Mapping

from mars.coordinator import CoordinatorReport
from mars.domain.execution import task_resource_demand
from mars.domain.task import (
    TaskClass,
    TaskInstance,
    TaskState,
    resolved_placement_constraints,
)
from mars.domain.topology import NodeSnapshot, NodeSpec
from mars.domain.workflow import WorkflowSpec
from mars.profiling import ProfileCatalog
from mars.run_artifact import RunArtifact

from .contracts import (
    AggregationRule,
    EvaluationResult,
    MetricDefinition,
    MetricObservation,
)


@dataclass(frozen=True)
class WorkflowEvaluationWeights:
    """Dimensionless weights for the balanced workflow reporting score."""

    success: float = 1.0
    communication: float = 1.0
    utilization: float = 2.0

    def __post_init__(self) -> None:
        values = (self.success, self.communication, self.utilization)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(
                "workflow evaluation weights must be finite and non-negative"
            )


def _definition(
    metric_id: str,
    unit: str,
    aggregation: AggregationRule = AggregationRule.MEAN,
) -> MetricDefinition:
    return MetricDefinition(metric_id, unit, aggregation)


WORKFLOW_METRIC_DEFINITIONS = {
    item.metric_id: item
    for item in (
        _definition("task_count", "count", AggregationRule.SUM),
        _definition("success_rate", "ratio", AggregationRule.RATIO_OF_SUMS),
        _definition(
            "deadline_miss_rate",
            "ratio",
            AggregationRule.RATIO_OF_SUMS,
        ),
        _definition(
            "executed_deadline_miss_rate",
            "ratio",
            AggregationRule.RATIO_OF_SUMS,
        ),
        _definition(
            "required_task_on_time_rate",
            "ratio",
            AggregationRule.RATIO_OF_SUMS,
        ),
        _definition("avg_latency_ms", "ms", AggregationRule.RATIO_OF_SUMS),
        _definition("p95_latency_ms", "ms"),
        _definition("p99_latency_ms", "ms"),
        _definition("avg_energy_j", "joule", AggregationRule.RATIO_OF_SUMS),
        _definition("total_energy_j", "joule", AggregationRule.SUM),
        _definition("total_solver_time_ms", "ms", AggregationRule.SUM),
        _definition("max_solver_time_ms", "ms", AggregationRule.MAX),
        _definition("scheduling_epoch_count", "count", AggregationRule.SUM),
        _definition("bandwidth_mb", "megabyte", AggregationRule.SUM),
        _definition("makespan_ms", "ms"),
        _definition(
            "edge_offload_ratio",
            "ratio",
            AggregationRule.RATIO_OF_SUMS,
        ),
        _definition(
            "safety_violation_count",
            "count",
            AggregationRule.SUM,
        ),
        _definition("skipped_task_count", "count", AggregationRule.SUM),
        _definition(
            "workflow_success_rate",
            "ratio",
            AggregationRule.RATIO_OF_SUMS,
        ),
        _definition("critical_path_ms", "ms"),
        _definition("dag_depth", "count"),
        _definition(
            "expected_success_reward",
            "priority_weight",
            AggregationRule.SUM,
        ),
        _definition(
            "expected_success_ratio",
            "ratio",
            AggregationRule.RATIO_OF_SUMS,
        ),
        _definition(
            "communication_time_ms",
            "ms",
            AggregationRule.SUM,
        ),
        _definition(
            "normalized_communication",
            "ratio",
            AggregationRule.RATIO_OF_SUMS,
        ),
        _definition("peak_cpu_utilization", "ratio"),
        _definition("peak_gpu_utilization", "ratio"),
        _definition("peak_memory_utilization", "ratio"),
        _definition("maximum_resource_utilization", "ratio"),
        _definition("workflow_evaluation_objective", "dimensionless"),
        _definition("fallback_count", "count", AggregationRule.SUM),
    )
}


def evaluate_run_artifact(
    artifact: RunArtifact,
    *,
    weights: WorkflowEvaluationWeights = WorkflowEvaluationWeights(),
) -> EvaluationResult:
    """Compute all supported workflow metrics from immutable run evidence."""

    report = artifact.raw_report
    workflow = artifact.workflow
    task_by_id = {task.task_id: task for task in workflow.tasks}
    node_by_id = {node.node_id: node for node in artifact.node_specs}
    snapshot_by_id = {
        snapshot.node_id: snapshot for snapshot in artifact.node_snapshots
    }
    profiles = ProfileCatalog(list(artifact.profiles))

    task_count = len(workflow.tasks)
    succeeded = 0
    missed = 0
    executed_missed = 0
    required_on_time = 0
    skipped = 0
    executed_count = 0
    edge_count = 0
    safety_violations = 0
    latencies: list[float] = []
    energies: list[float] = []
    expected_reward = 0.0
    communication_ms = 0.0
    finish_times: list[float] = []
    intervals_by_node: dict[
        str,
        list[tuple[float, float, tuple[float, float, float]]],
    ] = {node_id: [] for node_id in node_by_id}

    for raw_row in report.task_results:
        row = _mapping(raw_row)
        task_id = str(row.get("task_id", ""))
        task = task_by_id.get(task_id)
        if task is None:
            continue
        state = str(row.get("state", TaskState.BLOCKED.value))
        target_node_id = str(row.get("target_node_id", ""))
        target_node = node_by_id.get(target_node_id)
        attempts = tuple(
            _mapping(item)
            for item in _sequence(row.get("attempts", ()))
        )
        finishes = [_number(item.get("finish_time_ms")) for item in attempts]
        finish_ms = max(finishes, default=0.0)
        compute_ms = sum(
            _number(item.get("compute_time_ms")) for item in attempts
        )
        energy_j = sum(_number(item.get("energy_j")) for item in attempts)
        projected_finish_ms = round(finish_ms, 2)
        projected_compute_ms = round(compute_ms, 2)
        projected_energy_j = round(energy_j, 2)
        projected_latency_ms = round(
            max(0.0, finish_ms - task.arrival_time_ms)
            if attempts
            else 0.0,
            2,
        )
        finish_times.append(projected_finish_ms)
        deadline_missed = bool(
            attempts
            and (
                finish_ms > task.deadline_time_ms
                or state == TaskState.TIMEOUT.value
            )
        )
        success = state == TaskState.SUCCEEDED.value
        executed = projected_compute_ms > 0.0
        succeeded += int(success)
        missed += int(deadline_missed)
        executed_count += int(executed)
        executed_missed += int(executed and deadline_missed)
        required_on_time += int(success and not deadline_missed)
        skipped += int(state == TaskState.SKIPPED.value)
        edge_count += int(str(row.get("mode", "")) == "edge")
        safety_violations += int(
            _violates_safety_contract(task, target_node_id, node_by_id)
        )

        if projected_finish_ms > 0.0 or state in {
            TaskState.SUCCEEDED.value,
            TaskState.FAILED.value,
            TaskState.TIMEOUT.value,
            TaskState.DROPPED.value,
        }:
            latencies.append(projected_latency_ms)
            energies.append(projected_energy_j)

        if target_node is not None:
            profile = profiles.lookup(task.spec.task_type, target_node.kind)
            success_probability = (
                1.0 - profile.failure_rate if profile is not None else 1.0
            )
            expected_reward += max(0, task.priority) * success_probability

        for attempt in attempts:
            communication_ms += _number(
                attempt.get("communication_time_ms")
            )
            attempt_node_id = str(
                attempt.get("target_node_id", target_node_id)
            )
            attempt_node = node_by_id.get(attempt_node_id)
            if attempt_node is None:
                continue
            attempt_start_ms = _number(attempt.get("start_time_ms"))
            attempt_finish_ms = _number(attempt.get("finish_time_ms"))
            attempt_compute_ms = _number(attempt.get("compute_time_ms"))
            compute_start_ms = max(
                attempt_start_ms,
                attempt_finish_ms - attempt_compute_ms,
            )
            if attempt_finish_ms <= compute_start_ms:
                continue
            intervals_by_node[attempt_node_id].append(
                (
                    compute_start_ms,
                    attempt_finish_ms,
                    _resource_demand(task, attempt_node, profiles),
                )
            )

    peaks = _peak_utilization(
        node_by_id,
        snapshot_by_id,
        intervals_by_node,
    )
    maximum_utilization = max(peaks.values(), default=0.0)
    total_priority = float(
        sum(max(0, task.priority) for task in workflow.tasks)
    )
    priority_denominator = max(1.0, total_priority)
    expected_success_ratio = expected_reward / priority_denominator
    communication_scale_ms = max(
        1.0,
        sum(
            max(1.0, task.spec.latency_budget_ms)
            for task in workflow.tasks
        ),
    )
    normalized_communication = communication_ms / communication_scale_ms
    objective = (
        -weights.success * expected_success_ratio
        + weights.communication * normalized_communication
        + weights.utilization * maximum_utilization
    )
    makespan_ms = max(finish_times, default=0.0)
    levels = _mapping(report.workflow.get("levels", {}))
    workflow_success = int(report.workflow.get("state") == "succeeded")
    raw_metrics = report.metrics

    values: dict[str, float | int] = {
        "task_count": task_count,
        "success_rate": round(succeeded / max(1, task_count), 4),
        "deadline_miss_rate": round(missed / max(1, task_count), 4),
        "executed_deadline_miss_rate": round(
            executed_missed / max(1, executed_count),
            4,
        ),
        "required_task_on_time_rate": round(
            required_on_time / max(1, task_count),
            4,
        ),
        "avg_latency_ms": round(mean(latencies) if latencies else 0.0, 2),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 2),
        "p99_latency_ms": round(_percentile(latencies, 0.99), 2),
        "avg_energy_j": round(mean(energies) if energies else 0.0, 2),
        "total_energy_j": round(sum(energies), 2),
        "total_solver_time_ms": round(
            _number(raw_metrics.get("total_solver_time_ms")),
            6,
        ),
        "max_solver_time_ms": round(
            _number(raw_metrics.get("max_solver_time_ms")),
            6,
        ),
        "scheduling_epoch_count": int(
            _number(raw_metrics.get("scheduling_epoch_count"))
        ),
        "bandwidth_mb": round(
            _number(raw_metrics.get("transferred_mb")),
            2,
        ),
        "makespan_ms": round(makespan_ms, 2),
        "edge_offload_ratio": round(edge_count / max(1, task_count), 4),
        "safety_violation_count": safety_violations,
        "skipped_task_count": skipped,
        "workflow_success_rate": float(workflow_success),
        "critical_path_ms": round(
            _number(raw_metrics.get("critical_path_ms")),
            2,
        ),
        "dag_depth": max(
            (int(value) for value in levels.values()),
            default=-1,
        )
        + 1,
        "expected_success_reward": round(expected_reward, 6),
        "expected_success_ratio": round(expected_success_ratio, 6),
        "communication_time_ms": round(communication_ms, 6),
        "normalized_communication": round(normalized_communication, 6),
        "peak_cpu_utilization": round(peaks["cpu"], 6),
        "peak_gpu_utilization": round(peaks["gpu"], 6),
        "peak_memory_utilization": round(peaks["memory"], 6),
        "maximum_resource_utilization": round(maximum_utilization, 6),
        "workflow_evaluation_objective": round(objective, 6),
        "fallback_count": int(_number(raw_metrics.get("fallback_count"))),
    }
    ratio_terms = {
        "success_rate": (succeeded, task_count),
        "deadline_miss_rate": (missed, task_count),
        "executed_deadline_miss_rate": (
            executed_missed,
            executed_count,
        ),
        "required_task_on_time_rate": (required_on_time, task_count),
        "avg_latency_ms": (sum(latencies), len(latencies)),
        "avg_energy_j": (sum(energies), len(energies)),
        "edge_offload_ratio": (edge_count, task_count),
        "workflow_success_rate": (workflow_success, 1),
        "expected_success_ratio": (expected_reward, priority_denominator),
        "normalized_communication": (
            communication_ms,
            communication_scale_ms,
        ),
    }
    return EvaluationResult(
        observations=tuple(
            MetricObservation(
                WORKFLOW_METRIC_DEFINITIONS[metric_id],
                value,
                *(ratio_terms.get(metric_id, (None, None))),
            )
            for metric_id, value in values.items()
        )
    )


def evaluate_task_class_summary(
    artifact: RunArtifact,
) -> dict[str, dict[str, float | int]]:
    """Compute the existing API cohort summary from the same run artifact."""

    return evaluate_task_class_summary_from_report(
        artifact.workflow,
        artifact.raw_report,
    )


def evaluate_task_class_summary_from_report(
    workflow: WorkflowSpec,
    report: CoordinatorReport,
) -> dict[str, dict[str, float | int]]:
    """Project task-class cohorts for the compatibility report adapter."""

    task_by_id = {task.task_id: task for task in workflow.tasks}
    grouped: dict[TaskClass, list[tuple[bool, float, str]]] = {
        task_class: [] for task_class in TaskClass
    }
    for row in report.task_results:
        task = task_by_id[str(row["task_id"])]
        attempts = tuple(
            _mapping(item)
            for item in _sequence(row.get("attempts", ()))
        )
        finish_ms = max(
            (_number(item.get("finish_time_ms")) for item in attempts),
            default=0.0,
        )
        latency_ms = round(
            max(0.0, finish_ms - task.arrival_time_ms)
            if attempts
            else 0.0,
            2,
        )
        grouped[task.spec.task_class].append(
            (
                str(row.get("state", "")) == TaskState.SUCCEEDED.value,
                latency_ms,
                str(row.get("mode", "")),
            )
        )
    return {
        task_class.value: {
            "task_count": len(items),
            "success_rate": round(
                sum(success for success, _, _ in items) / max(1, len(items)),
                4,
            ),
            "avg_latency_ms": round(
                mean(latency for _, latency, _ in items) if items else 0.0,
                2,
            ),
            "edge_offload_ratio": round(
                sum(mode == "edge" for _, _, mode in items)
                / max(1, len(items)),
                4,
            ),
        }
        for task_class, items in grouped.items()
    }


def _peak_utilization(
    node_by_id: Mapping[str, NodeSpec],
    snapshot_by_id: Mapping[str, NodeSnapshot],
    intervals_by_node: Mapping[
        str,
        list[tuple[float, float, tuple[float, float, float]]],
    ],
) -> dict[str, float]:
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
            cpu = (
                snapshot.cpu_util
                + sum(item[0] for item in active) / node.cpu_capacity
            )
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
    return peaks


def _resource_demand(
    task: TaskInstance,
    node: NodeSpec,
    profiles: ProfileCatalog,
) -> tuple[float, float, float]:
    profile = profiles.lookup(task.spec.task_type, node.kind)
    if profile is None:
        return task_resource_demand(task, node)
    cpu, gpu, _ = task_resource_demand(task, node)
    return (
        float(cpu),
        float(gpu),
        float(profile.peak_memory_mb) / 1024.0,
    )


def _violates_safety_contract(
    task: TaskInstance,
    target_node_id: str,
    node_by_id: Mapping[str, NodeSpec],
) -> bool:
    constraints = resolved_placement_constraints(task)
    if not constraints.safety_required or not target_node_id:
        return False
    target = node_by_id.get(target_node_id)
    return (
        target is None
        or not target.safety_capable
        or bool(constraints.pinned_node_id)
        and target_node_id != constraints.pinned_node_id
    )


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    raise TypeError("run artifact entries must be mappings")


def _sequence(value: object) -> Iterable[object]:
    if isinstance(value, (tuple, list)):
        return value
    return ()


def _number(value: object) -> float:
    try:
        resolved = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return resolved if math.isfinite(resolved) else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = [
    "WORKFLOW_METRIC_DEFINITIONS",
    "WorkflowEvaluationWeights",
    "evaluate_run_artifact",
    "evaluate_task_class_summary",
    "evaluate_task_class_summary_from_report",
]
