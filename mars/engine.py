"""Compatibility facade for running MARS workflows in virtual time.

The coordinator owns the scheduling and completion event loop. This module
configures the process-local runtime and projects ``CoordinatorReport`` into
the ``SimulationReport`` representation used by the Web API.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Iterable, Mapping

from evals.contracts import EvaluationResult
from evals.workflow import (
    WorkflowEvaluationWeights,
    evaluate_run_artifact,
    evaluate_task_class_summary_from_report,
)

from .coordinator import CentralCoordinator, CoordinatorReport
from .domain.task import (
    TaskInstance,
    TaskState,
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
    SolveLimits,
)
from .profiling import ProfileCatalog
from .run_artifact import RunArtifact, build_run_artifact
from .runtime import InProcessRuntime


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
    run_id: str | None = None,
    solve_limits: SolveLimits | None = None,
) -> SimulationReport:
    """Execute, evaluate, and project one run for compatibility callers."""

    artifact = run_workflow_artifact(
        workflow,
        node_specs,
        node_snapshots,
        algorithm=algorithm,
        formulation=formulation,
        seed=seed,
        network_jitter=network_jitter,
        resource_noise=resource_noise,
        profiles=profiles,
        link_specs=link_specs,
        link_snapshots=link_snapshots,
        optimizer_registry=optimizer_registry,
        formulation_registry=formulation_registry,
        fallback_optimizer=fallback_optimizer,
        fail_first_task_ids=fail_first_task_ids,
        run_id=run_id,
        solve_limits=solve_limits,
    )
    return project_run_artifact(
        artifact,
        evaluation_weights=evaluation_weights,
    )


def project_run_artifact(
    artifact: RunArtifact,
    *,
    evaluation_weights: WorkflowEvaluationWeights = (
        WorkflowEvaluationWeights()
    ),
) -> SimulationReport:
    """Evaluate and project one factual run through the legacy report API."""

    evaluation = evaluate_run_artifact(
        artifact,
        weights=evaluation_weights,
    )
    report = replace(
        artifact.raw_report,
        metrics={
            **artifact.raw_report.metrics,
            **evaluation.as_dict(),
        },
        workflow={
            **artifact.raw_report.workflow,
            "metric_schema_version": evaluation.schema_version,
        },
    )
    return project_coordinator_report(
        report,
        artifact.workflow,
        artifact.node_specs,
        algorithm=artifact.algorithm,
        profiles=ProfileCatalog(list(artifact.profiles)),
        network_jitter=artifact.network_jitter,
        resource_noise=artifact.resource_noise,
        evaluation=evaluation,
    )


def run_workflow_artifact(
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
    link_specs: list[LinkSpec] | None = None,
    link_snapshots: list[LinkSnapshot] | None = None,
    optimizer_registry: OptimizerRegistry | None = None,
    formulation_registry: FormulationRegistry | None = None,
    fallback_optimizer: str | None = "heuristic",
    fail_first_task_ids: Iterable[str] = (),
    max_attempts: int = 1,
    deterministic: bool = True,
    run_id: str | None = None,
    solve_limits: SolveLimits | None = None,
) -> RunArtifact:
    """Execute one workflow and return its unevaluated factual artifact."""

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
        max_attempts=max_attempts,
        fail_first_task_ids=fail_first_task_ids,
        deterministic=deterministic,
        solve_limits=solve_limits,
    )
    return build_run_artifact(
        run_id=(
            run_id
            or f"run:{workflow.workflow_id}:{algorithm}:{seed}"
        ),
        workflow=workflow,
        node_specs=node_specs,
        node_snapshots=node_snapshots,
        link_specs=resolved_link_specs,
        link_snapshots=resolved_link_snapshots,
        profiles=coordinator.profile_catalog.profiles,
        raw_report=report,
        algorithm=algorithm,
        formulation=formulation,
        seed=seed,
        deterministic=deterministic,
        max_attempts=max_attempts,
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
    evaluation: EvaluationResult | None = None,
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

    metrics = (
        evaluation.as_dict()
        if evaluation is not None
        else dict(report.metrics)
    )
    makespan_ms = float(metrics["makespan_ms"])
    state_counts = Counter(record.state for record in records)
    levels = {
        str(task_id): int(level)
        for task_id, level in dict(report.workflow.get("levels", {})).items()
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
                report.workflow.get("critical_path", ())
            ),
            **{
                key: _plain_data(report.workflow[key])
                for key in (
                    "scheduling",
                    "requested_algorithm",
                    "optimizer_options",
                    "metric_schema_version",
                )
                if key in report.workflow
            },
        },
        task_class_summary=evaluate_task_class_summary_from_report(
            workflow,
            report,
        ),
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


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    raise TypeError("coordinator report entries must be mappings")


def _plain_data(value: object) -> object:
    """Detach immutable run evidence before exposing legacy API data."""

    if isinstance(value, Mapping):
        return {
            str(key): _plain_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain_data(item) for item in value]
    return value


def _number(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
