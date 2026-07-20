"""Deterministic event-driven workflow simulator backed by the MARS DAG state machine."""

from __future__ import annotations

import heapq
import random
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean

from .dag import TaskManager
from .models import (
    ArtifactRef,
    Assignment,
    NodeSnapshot,
    TaskClass,
    TaskInstance,
    TaskState,
    WorkflowSpec,
)
from .scheduler import apply_load, choose_assignment, critical_path
from .profiling import ProfileCatalog, load_default_catalog


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
        data["task_results"] = [asdict(record) for record in self.task_results]
        return data


def run_workflow_simulation(
    workflow: WorkflowSpec,
    nodes: list[NodeSnapshot],
    *,
    algorithm: str = "dag_deadline",
    seed: int = 7,
    network_jitter: float = 0.1,
    resource_noise: float = 0.05,
    profiles: ProfileCatalog | None = None,
) -> SimulationReport:
    manager = TaskManager()
    index = manager.submit(workflow)
    critical_ids, critical_path_ms, critical_tail = critical_path(workflow.tasks, index)
    rng = random.Random(seed)
    if profiles is None:
        profiles = load_default_catalog()
    node_by_id = {node.node_id: node for node in nodes}
    node_available = {node.node_id: 0.0 for node in nodes}
    busy_by_node = {node.node_id: 0.0 for node in nodes}
    completion_time: dict[str, float] = {}
    records: dict[str, SimulationRecord] = {}
    events: list[tuple[float, int, str, bool, Assignment]] = []
    sequence = 0
    logs: list[str] = []
    total_bandwidth_mb = 0.0

    while manager.unresolved():
        ready = manager.ready()
        if ready:
            def ready_at(task: TaskInstance) -> float:
                parents = index.parents[task.task_id]
                parent_time = max((completion_time.get(parent, 0.0) for parent in parents), default=0.0)
                return max(task.arrival_time_ms, parent_time)

            def ready_key(task: TaskInstance) -> tuple[float, float, int, str]:
                at = ready_at(task)
                slack = task.deadline_time_ms - at - critical_tail[task.task_id]
                if algorithm == "dag_deadline":
                    return (at, slack, -task.priority, task.task_id)
                return (at, 0.0, -task.priority, task.task_id)

            for task in sorted(ready, key=ready_key):
                at = ready_at(task)
                artifacts = [
                    artifact
                    for parent in index.parents[task.task_id]
                    if (artifact := manager.artifact_for(parent)) is not None
                ]
                assignment = choose_assignment(
                    task,
                    algorithm=algorithm,
                    ready_time_ms=at,
                    node_available=node_available,
                    nodes=node_by_id,
                    parent_artifacts=artifacts,
                    critical_tail_ms=critical_tail[task.task_id],
                    profiles=profiles,
                )
                manager.mark_running(task.task_id)
                if not assignment.target_node_id:
                    sequence += 1
                    heapq.heappush(events, (at, sequence, task.task_id, False, assignment))
                    continue

                jitter = max(0.75, rng.gauss(1.0, network_jitter * 0.35))
                noise = rng.uniform(1.0 - resource_noise, 1.0 + resource_noise)
                compute_ms = assignment.compute_ms * noise
                communication_ms = assignment.communication_ms * jitter
                start_ms = assignment.estimated_start_ms
                finish_ms = start_ms + compute_ms + communication_ms
                assignment = Assignment(
                    task_id=assignment.task_id,
                    target_node_id=assignment.target_node_id,
                    execution_mode=assignment.execution_mode,
                    estimated_start_ms=start_ms,
                    estimated_finish_ms=finish_ms,
                    compute_ms=compute_ms,
                    communication_ms=communication_ms,
                    energy_j=assignment.energy_j * noise,
                    reason=assignment.reason,
                    input_locations=assignment.input_locations,
                )
                node_available[assignment.target_node_id] = finish_ms
                busy_by_node[assignment.target_node_id] += compute_ms
                node_by_id[assignment.target_node_id] = apply_load(node_by_id[assignment.target_node_id], task)
                if assignment.target_node_id != task.source_node_id:
                    total_bandwidth_mb += sum(a.size_mb for a in artifacts) if artifacts else task.spec.input_size_mb

                deadline_missed = finish_ms > task.deadline_time_ms
                target = node_by_id[assignment.target_node_id]
                thermal_penalty = 0.2 if target.temperature_c > 93.0 else 0.0
                network_penalty = 0.04 if assignment.communication_ms > task.spec.latency_budget_ms * 0.5 else 0.0
                probability = max(
                    0.05,
                    task.expected_accuracy - thermal_penalty - network_penalty - (0.25 if deadline_missed else 0.0),
                )
                ok = rng.random() < probability
                sequence += 1
                heapq.heappush(events, (finish_ms, sequence, task.task_id, ok, assignment))
                records[task.task_id] = SimulationRecord(
                    task_id=task.task_id,
                    workflow_id=task.workflow_id,
                    task_name=task.name,
                    task_class=task.spec.task_class.value,
                    stage_index=task.stage_index,
                    dependencies=list(task.dependency_task_ids),
                    source_robot_id=task.source_node_id,
                    target_node_id=assignment.target_node_id,
                    mode=assignment.execution_mode.value,
                    priority=task.priority,
                    start_time_ms=round(start_ms, 2),
                    finish_time_ms=round(finish_ms, 2),
                    queue_delay_ms=round(max(0.0, start_ms - at), 2),
                    compute_time_ms=round(compute_ms, 2),
                    communication_time_ms=round(communication_ms, 2),
                    total_latency_ms=round(finish_ms - task.arrival_time_ms, 2),
                    energy_j=round(assignment.energy_j, 2),
                    deadline_missed=deadline_missed,
                    success=ok,
                    state=TaskState.RUNNING.value,
                    reason=assignment.reason,
                    input_locations=list(assignment.input_locations),
                    output_ref="",
                )
            continue

        if not events:
            raise RuntimeError("workflow is unresolved but has neither ready tasks nor completion events")

        finish_ms, _, task_id, ok, assignment = heapq.heappop(events)
        if manager.state_of(task_id) is TaskState.SKIPPED:
            continue
        task = manager.get(task_id)
        dropped = not assignment.target_node_id
        artifact = None
        if ok and not dropped:
            artifact = ArtifactRef(
                artifact_id=f"artifact:{workflow.workflow_id}:{task_id}",
                producer_task_id=task_id,
                node_id=assignment.target_node_id,
                size_mb=task.spec.output_size_mb,
                uri=f"node://{assignment.target_node_id}/{workflow.workflow_id}/{task_id}",
            )
        released, skipped = manager.complete(
            task_id,
            ok=ok and not dropped,
            finished_time_ms=finish_ms,
            artifact=artifact,
            timed_out=(finish_ms > task.deadline_time_ms and not ok),
            dropped=dropped,
            error_code="no_feasible_node" if dropped else ("execution_failed" if not ok else ""),
        )
        completion_time[task_id] = finish_ms
        if task_id not in records:
            records[task_id] = _empty_record(task, state=manager.state_of(task_id), reason=assignment.reason)
        record = records[task_id]
        record.state = manager.state_of(task_id).value
        record.success = manager.state_of(task_id) is TaskState.SUCCEEDED
        record.output_ref = artifact.uri if artifact else ""
        record.deadline_missed = finish_ms > task.deadline_time_ms or manager.state_of(task_id) is TaskState.TIMEOUT
        logs.append(
            f"{task_id} [{task.spec.task_class.value}] -> {assignment.target_node_id or 'DROP'} "
            f"state={record.state}; released={released}; skipped={skipped}; {assignment.reason}"
        )

    for task in workflow.tasks:
        if task.task_id not in records or manager.state_of(task.task_id) is TaskState.SKIPPED:
            reason = "skipped because an upstream dependency failed"
            records[task.task_id] = _empty_record(task, state=manager.state_of(task.task_id), reason=reason)
            logs.append(f"{task.task_id} state=skipped; {reason}")

    ordered_records = [records[task_id] for task_id in index.topological_order]
    finished_records = [record for record in ordered_records if record.finish_time_ms > 0]
    latencies = [record.total_latency_ms for record in finished_records]
    energies = [record.energy_j for record in finished_records]
    makespan = max((record.finish_time_ms for record in finished_records), default=0.0)
    succeeded = sum(record.state == TaskState.SUCCEEDED.value for record in ordered_records)
    missed = sum(record.deadline_missed or not record.success for record in ordered_records)
    offloaded = sum(record.mode == "edge" for record in ordered_records)
    safety_violations = sum(
        record.task_class == TaskClass.LOCAL_SAFETY.value
        and record.target_node_id
        and record.target_node_id != record.source_robot_id
        for record in ordered_records
    )
    progress = manager.progress(critical_ids)
    metrics: dict[str, float | int] = {
        "task_count": len(ordered_records),
        "success_rate": round(succeeded / max(1, len(ordered_records)), 4),
        "deadline_miss_rate": round(missed / max(1, len(ordered_records)), 4),
        "avg_latency_ms": round(mean(latencies) if latencies else 0.0, 2),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 2),
        "p99_latency_ms": round(_percentile(latencies, 0.99), 2),
        "avg_energy_j": round(mean(energies) if energies else 0.0, 2),
        "total_energy_j": round(sum(energies), 2),
        "bandwidth_mb": round(total_bandwidth_mb, 2),
        "makespan_ms": round(makespan, 2),
        "edge_offload_ratio": round(offloaded / max(1, len(ordered_records)), 4),
        "safety_violation_count": safety_violations,
        "skipped_task_count": sum(record.state == TaskState.SKIPPED.value for record in ordered_records),
        "workflow_success_rate": 1.0 if progress.state.value == "succeeded" else 0.0,
        "critical_path_ms": round(critical_path_ms, 2),
        "dag_depth": max(index.levels.values(), default=0) + 1,
    }
    class_summary: dict[str, dict[str, float | int]] = {}
    for task_class in TaskClass:
        items = [record for record in ordered_records if record.task_class == task_class.value]
        class_summary[task_class.value] = {
            "task_count": len(items),
            "success_rate": round(sum(item.success for item in items) / max(1, len(items)), 4),
            "avg_latency_ms": round(mean([item.total_latency_ms for item in items]) if items else 0.0, 2),
            "edge_offload_ratio": round(sum(item.mode == "edge" for item in items) / max(1, len(items)), 4),
        }

    return SimulationReport(
        algorithm=algorithm,
        metrics=metrics,
        task_results=ordered_records,
        node_utilization={node_id: round(busy / max(1.0, makespan), 4) for node_id, busy in busy_by_node.items()},
        logs=logs,
        workflow={
            "workflow_id": workflow.workflow_id,
            "state": progress.state.value,
            "failure_policy": workflow.failure_policy.value,
            "deadline_time_ms": workflow.deadline_time_ms,
            "deadline_missed": bool(workflow.deadline_time_ms and makespan > workflow.deadline_time_ms),
            "state_counts": dict(Counter(record.state for record in ordered_records)),
            "critical_path": list(critical_ids),
        },
        task_class_summary=class_summary,
        dag={
            "valid": True,
            "topological_order": list(index.topological_order),
            "levels": index.levels,
            "edges": [
                {"from": parent, "to": child}
                for child in index.topological_order
                for parent in index.parents[child]
            ],
        },
        transport={
            "active": "in_process_simulation",
            "schema_version": "mars.v1",
            "profile_source": "synthetic_placeholder" if profiles is not None else "demand_formula_fallback",
        },
    )


def _empty_record(task: TaskInstance, *, state: TaskState, reason: str) -> SimulationRecord:
    return SimulationRecord(
        task_id=task.task_id,
        workflow_id=task.workflow_id,
        task_name=task.name,
        task_class=task.spec.task_class.value,
        stage_index=task.stage_index,
        dependencies=list(task.dependency_task_ids),
        source_robot_id=task.source_node_id,
        target_node_id="",
        mode="drop" if state is TaskState.DROPPED else "",
        priority=task.priority,
        start_time_ms=0.0,
        finish_time_ms=0.0,
        queue_delay_ms=0.0,
        compute_time_ms=0.0,
        communication_time_ms=0.0,
        total_latency_ms=0.0,
        energy_j=0.0,
        deadline_missed=state in {TaskState.TIMEOUT, TaskState.FAILED, TaskState.SKIPPED, TaskState.DROPPED},
        success=False,
        state=state.value,
        reason=reason,
        input_locations=[],
        output_ref="",
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
