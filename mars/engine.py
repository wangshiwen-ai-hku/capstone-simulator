"""Compatibility facade for running MARS workflows in virtual time.

The coordinator owns the scheduling and completion event loop. This module
configures the process-local runtime and projects ``CoordinatorReport`` into
the ``SimulationReport`` representation used by the Web API.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from statistics import mean
from typing import Iterable, Mapping

from .coordinator import CentralCoordinator, CoordinatorReport
from .dag import TaskManager
from .domain.task import (
    TaskClass,
    TaskInstance,
    TaskState,
    resolved_placement_constraints,
)
from .domain.topology import (
    LinkSnapshot,
    LinkSpec,
    NodeSnapshot,
    NodeSpec,
)
from .domain.workflow import WorkflowSpec
from .network import synthesize_legacy_full_mesh
from .optimizers import (
    FormulationRegistry,
    OptimizerRegistry,
    SchedulingFormulation,
)
from .profiling import ProfileCatalog
from .runtime import InProcessRuntime
from .scheduler import critical_path
from .workflow_metrics import (
    WorkflowEvaluationWeights,
    evaluate_workflow_metrics,
)


@dataclass
class SimulationRecord:
    task_id: str
    workflow_id: str
    task_name: str
    task_class: str
    stage_index: int
    dependencies: list[str]
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
    input_locations: list[str]
    output_ref: str


@dataclass
class SimulationReport:
    algorithm: str
    metrics: dict[str, float | int]
    task_results: list[SimulationRecord]
    node_utilization: dict[str, float]
    logs: list[str]
    workflow: dict[str, object]
    task_class_summary: dict[str, dict[str, float | int]]
    dag: dict[str, object]
    transport: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["task_results"] = [
            asdict(record) for record in self.task_results
        ]
        return data


def run_workflow_simulation(
    workflow: WorkflowSpec,
    node_specs: list[NodeSpec],
    node_snapshots: list[NodeSnapshot],
    *,
    algorithm: str = "dag_deadline",
    formulation: str | SchedulingFormulation | None = None,
    seed: int = 7,
    network_jitter: float = 0.1,
    resource_noise: float = 0.05,
    profiles: ProfileCatalog | None = None,
    evaluation_weights: WorkflowEvaluationWeights = (
        WorkflowEvaluationWeights()
    ),
    link_specs: list[LinkSpec] | None = None,
    link_snapshots: list[LinkSnapshot] | None = None,
    optimizer_registry: OptimizerRegistry | None = None,
    formulation_registry: FormulationRegistry | None = None,
    fallback_optimizer: str | None = "heuristic",
    fail_first_task_ids: Iterable[str] = (),
) -> SimulationReport:
    """Execute a simulation through the coordinator and RuntimePort."""

    if network_jitter < 0:
        raise ValueError("network_jitter must be non-negative")
    if not 0.0 <= resource_noise <= 1.0:
        raise ValueError("resource_noise must be in [0, 1]")
    resolved_link_specs, resolved_link_snapshots = _resolve_links(
        node_specs,
        node_snapshots,
        link_specs=link_specs,
        link_snapshots=link_snapshots,
        network_jitter=network_jitter,
    )
    runtime = InProcessRuntime(
        node_specs,
        node_snapshots,
        execution_noise=resource_noise,
        sample_execution_failures=True,
    )
    coordinator = CentralCoordinator(
        runtime,
        link_specs=resolved_link_specs,
        link_snapshots=resolved_link_snapshots,
        optimizer_registry=optimizer_registry,
        formulation_registry=formulation_registry,
        profile_catalog=profiles,
        fallback_optimizer=fallback_optimizer,
    )
    report = coordinator.run(
        workflow,
        algorithm=algorithm,
        formulation=formulation,
        seed=seed,
        max_attempts=1,
        fail_first_task_ids=fail_first_task_ids,
        deterministic=True,
    )
    report = replace(
        report,
        metrics={
            **report.metrics,
            **evaluate_workflow_metrics(
                report.task_results,
                workflow,
                node_specs,
                node_snapshots,
                coordinator.profile_catalog,
                weights=evaluation_weights,
            ),
        },
    )
    return project_coordinator_report(
        report,
        workflow,
        node_specs,
        algorithm=algorithm,
        profiles=coordinator.profile_catalog,
        network_jitter=network_jitter,
        resource_noise=resource_noise,
    )


def project_coordinator_report(
    report: CoordinatorReport,
    workflow: WorkflowSpec,
    node_specs: Iterable[NodeSpec],
    *,
    algorithm: str,
    profiles: ProfileCatalog | None,
    network_jitter: float = 0.0,
    resource_noise: float = 0.0,
) -> SimulationReport:
    """Project a coordinator result onto the simulation contract."""

    task_by_id = {task.task_id: task for task in workflow.tasks}
    node_by_id = {node.node_id: node for node in node_specs}
    runtime_by_id = {
        str(item["task_id"]): item for item in report.task_results
    }
    topological_order = [
        str(task_id)
        for task_id in report.workflow.get(
            "topological_order",
            tuple(task_by_id),
        )
    ]
    finish_by_id: dict[str, float] = {}
    records: list[SimulationRecord] = []

    for task_id in topological_order:
        task = task_by_id[task_id]
        runtime_result = runtime_by_id[task_id]
        attempts = tuple(
            _mapping(item)
            for item in runtime_result.get("attempts", ())
        )
        start_time_ms = min(
            (
                _number(attempt.get("start_time_ms"))
                for attempt in attempts
            ),
            default=0.0,
        )
        finish_time_ms = max(
            (
                _number(attempt.get("finish_time_ms"))
                for attempt in attempts
            ),
            default=0.0,
        )
        finish_by_id[task_id] = finish_time_ms
        parent_ids = [
            str(parent_id)
            for parent_id in runtime_result.get(
                "dependencies",
                task.dependency_task_ids,
            )
        ]
        released_at_ms = max(
            task.arrival_time_ms,
            max(
                (
                    finish_by_id.get(parent_id, 0.0)
                    for parent_id in parent_ids
                ),
                default=0.0,
            ),
        )
        state = str(runtime_result.get("state", TaskState.BLOCKED.value))
        outputs = tuple(
            _mapping(item)
            for item in runtime_result.get("outputs", ())
        )
        records.append(
            SimulationRecord(
                task_id=task_id,
                workflow_id=task.workflow_id,
                task_name=task.name,
                task_class=task.spec.task_class.value,
                stage_index=task.stage_index,
                dependencies=parent_ids,
                source_robot_id=task.source_node_id,
                target_node_id=str(
                    runtime_result.get("target_node_id", "")
                ),
                mode=str(runtime_result.get("mode", "")),
                priority=task.priority,
                start_time_ms=round(start_time_ms, 2),
                finish_time_ms=round(finish_time_ms, 2),
                queue_delay_ms=round(
                    max(0.0, start_time_ms - released_at_ms),
                    2,
                ),
                compute_time_ms=round(
                    sum(
                        _number(attempt.get("compute_time_ms"))
                        for attempt in attempts
                    ),
                    2,
                ),
                communication_time_ms=round(
                    sum(
                        _number(
                            attempt.get("communication_time_ms")
                        )
                        for attempt in attempts
                    ),
                    2,
                ),
                total_latency_ms=round(
                    max(
                        0.0,
                        finish_time_ms - task.arrival_time_ms,
                    )
                    if attempts
                    else 0.0,
                    2,
                ),
                energy_j=round(
                    sum(
                        _number(attempt.get("energy_j"))
                        for attempt in attempts
                    ),
                    2,
                ),
                deadline_missed=bool(
                    attempts
                    and (
                        finish_time_ms > task.deadline_time_ms
                        or state == TaskState.TIMEOUT.value
                    )
                ),
                success=state == TaskState.SUCCEEDED.value,
                state=state,
                reason=_task_result_reason(state, attempts),
                input_locations=_input_locations(
                    task,
                    parent_ids,
                    runtime_by_id,
                ),
                output_ref=(
                    str(outputs[0].get("uri", ""))
                    if outputs
                    else ""
                ),
            )
        )

    manager = TaskManager()
    index = manager.submit(workflow)
    critical_ids, critical_path_ms, _ = critical_path(
        workflow.tasks,
        index,
    )
    metrics = _simulation_metrics(
        report,
        workflow,
        records,
        node_by_id,
        critical_path_ms=critical_path_ms,
    )
    makespan_ms = float(metrics["makespan_ms"])
    state_counts = Counter(record.state for record in records)
    levels = {
        str(task_id): int(level)
        for task_id, level in dict(
            report.workflow.get("levels", index.levels)
        ).items()
    }
    profile_sources = _profile_sources(
        records,
        task_by_id,
        node_by_id,
        profiles,
    )

    return SimulationReport(
        algorithm=algorithm,
        metrics=metrics,
        task_results=records,
        node_utilization={
            str(agent["agent_id"]): round(
                min(
                    1.0,
                    max(0.0, _number(agent.get("utilization"))),
                ),
                4,
            )
            for agent in report.agents
        },
        logs=list(report.logs),
        workflow={
            "workflow_id": workflow.workflow_id,
            "state": str(report.workflow["state"]),
            "failure_policy": workflow.failure_policy.value,
            "deadline_time_ms": workflow.deadline_time_ms,
            "deadline_missed": bool(
                workflow.deadline_time_ms
                and makespan_ms > workflow.deadline_time_ms
            ),
            "state_counts": dict(state_counts),
            "critical_path": list(
                report.workflow.get("critical_path", critical_ids)
            ),
            **{
                key: report.workflow[key]
                for key in (
                    "scheduling",
                    "requested_algorithm",
                    "optimizer_options",
                    "metric_schema_version",
                )
                if key in report.workflow
            },
        },
        task_class_summary=_task_class_summary(records),
        dag={
            "valid": True,
            "topological_order": topological_order,
            "levels": levels,
            "edges": [
                {"from": parent_id, "to": task_id}
                for task_id in topological_order
                for parent_id in runtime_by_id[task_id].get(
                    "dependencies",
                    (),
                )
            ],
        },
        transport={
            "active": "in_process_runtime",
            "execution_path": "central_coordinator_runtime_port",
            "runtime_adapter": "InProcessRuntime",
            "network_jitter": network_jitter,
            "resource_noise": resource_noise,
            "runtime_event_count": len(report.events),
            "profile_source": (
                profile_sources[0]
                if len(profile_sources) == 1
                else "mixed"
                if profile_sources
                else "not_used"
            ),
            "profile_sources": profile_sources,
            "profile_catalog_provenance": (
                profiles.provenance
                if profiles is not None
                else "unavailable"
            ),
        },
    )


def _resolve_links(
    node_specs: Iterable[NodeSpec],
    node_snapshots: Iterable[NodeSnapshot],
    *,
    link_specs: Iterable[LinkSpec] | None,
    link_snapshots: Iterable[LinkSnapshot] | None,
    network_jitter: float,
) -> tuple[tuple[LinkSpec, ...], tuple[LinkSnapshot, ...]]:
    if (link_specs is None) != (link_snapshots is None):
        raise ValueError(
            "link_specs and link_snapshots must both be provided or omitted"
        )
    if link_specs is None:
        resolved_specs, resolved_snapshots = synthesize_legacy_full_mesh(
            node_specs,
            node_snapshots,
        )
    else:
        resolved_specs = tuple(link_specs)
        resolved_snapshots = tuple(link_snapshots or ())
    return resolved_specs, tuple(
        replace(
            snapshot,
            jitter_ms=(
                snapshot.jitter_ms
                + max(1.0, snapshot.latency_ms) * network_jitter
            ),
        )
        for snapshot in resolved_snapshots
    )


def _simulation_metrics(
    report: CoordinatorReport,
    workflow: WorkflowSpec,
    records: list[SimulationRecord],
    node_by_id: Mapping[str, NodeSpec],
    *,
    critical_path_ms: float,
) -> dict[str, float | int]:
    completed = [
        record
        for record in records
        if record.finish_time_ms > 0.0
        or record.state
        in {
            TaskState.SUCCEEDED.value,
            TaskState.FAILED.value,
            TaskState.TIMEOUT.value,
            TaskState.DROPPED.value,
        }
    ]
    latencies = [record.total_latency_ms for record in completed]
    energies = [record.energy_j for record in completed]
    executed = [
        record
        for record in records
        if record.compute_time_ms > 0.0
    ]
    succeeded = sum(record.success for record in records)
    missed = sum(record.deadline_missed for record in records)
    executed_missed = sum(
        record.deadline_missed for record in executed
    )
    required_on_time = sum(
        record.success and not record.deadline_missed
        for record in records
    )
    makespan_ms = max(
        (record.finish_time_ms for record in records),
        default=0.0,
    )
    safety_violations = sum(
        _violates_safety_contract(
            next(
                task
                for task in workflow.tasks
                if task.task_id == record.task_id
            ),
            record.target_node_id,
            node_by_id,
        )
        for record in records
    )
    levels = dict(report.workflow.get("levels", {}))
    metrics: dict[str, float | int] = {
        "task_count": len(records),
        "success_rate": round(
            succeeded / max(1, len(records)),
            4,
        ),
        "deadline_miss_rate": round(
            missed / max(1, len(records)),
            4,
        ),
        "executed_deadline_miss_rate": round(
            executed_missed / max(1, len(executed)),
            4,
        ),
        "required_task_on_time_rate": round(
            required_on_time / max(1, len(records)),
            4,
        ),
        "avg_latency_ms": round(
            mean(latencies) if latencies else 0.0,
            2,
        ),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 2),
        "p99_latency_ms": round(_percentile(latencies, 0.99), 2),
        "avg_energy_j": round(
            mean(energies) if energies else 0.0,
            2,
        ),
        "total_energy_j": round(sum(energies), 2),
        "total_solver_time_ms": round(
            _number(report.metrics.get("total_solver_time_ms")),
            6,
        ),
        "max_solver_time_ms": round(
            _number(report.metrics.get("max_solver_time_ms")),
            6,
        ),
        "scheduling_epoch_count": int(
            _number(report.metrics.get("scheduling_epoch_count"))
        ),
        "bandwidth_mb": round(
            _number(report.metrics.get("transferred_mb")),
            2,
        ),
        "makespan_ms": round(makespan_ms, 2),
        "edge_offload_ratio": round(
            sum(record.mode == "edge" for record in records)
            / max(1, len(records)),
            4,
        ),
        "safety_violation_count": safety_violations,
        "skipped_task_count": sum(
            record.state == TaskState.SKIPPED.value
            for record in records
        ),
        "workflow_success_rate": (
            1.0
            if report.workflow["state"] == "succeeded"
            else 0.0
        ),
        "critical_path_ms": round(
            _number(
                report.metrics.get(
                    "critical_path_ms",
                    critical_path_ms,
                )
            ),
            2,
        ),
        "dag_depth": max(
            (int(level) for level in levels.values()),
            default=-1,
        )
        + 1,
    }
    for key in (
        "expected_success_reward",
        "expected_success_ratio",
        "communication_time_ms",
        "normalized_communication",
        "peak_cpu_utilization",
        "peak_gpu_utilization",
        "peak_memory_utilization",
        "maximum_resource_utilization",
        "workflow_evaluation_objective",
        "fallback_count",
    ):
        if key in report.metrics:
            value = _number(report.metrics[key])
            metrics[key] = int(value) if key == "fallback_count" else value
    return metrics


def _task_class_summary(
    records: Iterable[SimulationRecord],
) -> dict[str, dict[str, float | int]]:
    items = tuple(records)
    summary: dict[str, dict[str, float | int]] = {}
    for task_class in TaskClass:
        class_items = [
            record
            for record in items
            if record.task_class == task_class.value
        ]
        summary[task_class.value] = {
            "task_count": len(class_items),
            "success_rate": round(
                sum(record.success for record in class_items)
                / max(1, len(class_items)),
                4,
            ),
            "avg_latency_ms": round(
                mean(
                    record.total_latency_ms
                    for record in class_items
                )
                if class_items
                else 0.0,
                2,
            ),
            "edge_offload_ratio": round(
                sum(record.mode == "edge" for record in class_items)
                / max(1, len(class_items)),
                4,
            ),
        }
    return summary


def _profile_sources(
    records: Iterable[SimulationRecord],
    task_by_id: Mapping[str, TaskInstance],
    node_by_id: Mapping[str, NodeSpec],
    profiles: ProfileCatalog | None,
) -> list[str]:
    sources: set[str] = set()
    for record in records:
        node = node_by_id.get(record.target_node_id)
        if node is None:
            continue
        task = task_by_id[record.task_id]
        profile = (
            profiles.lookup(task.spec.task_type, node.kind)
            if profiles is not None
            else None
        )
        sources.add(
            profile.provenance
            if profile is not None
            else "demand_formula_fallback"
        )
    return sorted(sources)


def _input_locations(
    task: TaskInstance,
    parent_ids: Iterable[str],
    runtime_by_id: Mapping[str, Mapping[str, object]],
) -> list[str]:
    parents = tuple(parent_ids)
    if not parents:
        return [task.source_node_id]
    return list(
        dict.fromkeys(
            str(runtime_by_id[parent_id].get("target_node_id", ""))
            for parent_id in parents
            if runtime_by_id[parent_id].get("target_node_id")
        )
    )


def _task_result_reason(
    state: str,
    attempts: Iterable[Mapping[str, object]],
) -> str:
    items = tuple(attempts)
    error_code = (
        str(items[-1].get("error_code", "")) if items else ""
    )
    if error_code:
        return error_code
    if state == TaskState.SUCCEEDED.value:
        return "completed"
    if state == TaskState.SKIPPED.value:
        return "upstream_dependency_failed"
    return state


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
    raise TypeError("coordinator report entries must be mappings")


def _number(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
