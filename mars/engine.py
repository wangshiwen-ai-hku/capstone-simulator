"""Deterministic event-driven workflow simulator backed by the MARS DAG state machine."""

from __future__ import annotations

import heapq
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from statistics import mean

from .dag import TaskManager, resolve_task_input_bindings
from .models import (
    ArtifactRef,
    Assignment,
    FailurePolicy,
    LinkSnapshot,
    LinkSpec,
    NodeSnapshot,
    NodeSpec,
    TaskClass,
    TaskInstance,
    TaskState,
    TransferReservation,
    WorkflowSpec,
    artifacts_from_bindings,
    resolved_placement_constraints,
)
from .network import synthesize_legacy_full_mesh
from .optimizers import (
    OptimizerRegistry,
    PlannedResourceReservation,
    ResourceDemand,
    SchedulingEpoch,
)
from .scheduler import apply_load, critical_path, plan_scheduling_epoch
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
    node_specs: list[NodeSpec],
    node_snapshots: list[NodeSnapshot],
    *,
    algorithm: str = "dag_deadline",
    seed: int = 7,
    network_jitter: float = 0.1,
    resource_noise: float = 0.05,
    profiles: ProfileCatalog | None = None,
    link_specs: list[LinkSpec] | None = None,
    link_snapshots: list[LinkSnapshot] | None = None,
    optimizer_registry: OptimizerRegistry | None = None,
) -> SimulationReport:
    manager = TaskManager()
    index = manager.submit(workflow)
    critical_ids, critical_path_ms, critical_tail = critical_path(workflow.tasks, index)
    rng = random.Random(seed)
    if profiles is None:
        profiles = load_default_catalog()
    node_by_id = {node.node_id: node for node in node_specs}
    snapshot_by_id = {snapshot.node_id: snapshot for snapshot in node_snapshots}
    if len(node_by_id) != len(node_specs):
        raise ValueError("node spec ids must be unique")
    if len(snapshot_by_id) != len(node_snapshots):
        raise ValueError("node snapshot ids must be unique")
    if node_by_id.keys() != snapshot_by_id.keys():
        missing = sorted(node_by_id.keys() - snapshot_by_id.keys())
        unknown = sorted(snapshot_by_id.keys() - node_by_id.keys())
        raise ValueError(f"node inventory mismatch: missing snapshots={missing}; unknown snapshots={unknown}")
    if (link_specs is None) != (link_snapshots is None):
        raise ValueError(
            "link_specs and link_snapshots must both be provided or omitted"
        )
    if link_specs is None:
        synthesized_specs, synthesized_snapshots = (
            synthesize_legacy_full_mesh(node_specs, node_snapshots)
        )
        resolved_link_specs = list(synthesized_specs)
        resolved_link_snapshots = list(synthesized_snapshots)
    else:
        resolved_link_specs = list(link_specs)
        resolved_link_snapshots = list(link_snapshots or ())
    link_available = {
        link.link_id: 0.0 for link in resolved_link_specs
    }
    actual_reservations_by_node: dict[
        str, list[PlannedResourceReservation]
    ] = {
        node_id: [] for node_id in node_by_id
    }
    busy_by_node = {node_id: 0.0 for node_id in node_by_id}
    completion_time: dict[str, float] = {}
    records: dict[str, SimulationRecord] = {}
    events: list[tuple[float, int, str, bool, Assignment]] = []
    sequence = 0
    logs: list[str] = []
    total_bandwidth_mb = 0.0
    profile_sources_used: set[str] = set()
    current_time_ms = 0.0
    epoch_sequence = 0

    def ready_at(task: TaskInstance) -> float:
        parents = index.parents[task.task_id]
        parent_time = max((completion_time.get(parent, 0.0) for parent in parents), default=0.0)
        return max(task.arrival_time_ms, parent_time)

    while manager.unresolved():
        force_completion_before_replan = False
        ready = manager.ready()
        arrived = [task for task in ready if ready_at(task) <= current_time_ms]
        if arrived:
            epoch_sequence += 1
            epoch_tasks = tuple(
                sorted(arrived, key=lambda task: task.task_id)
            )
            input_bindings_by_task = {}
            ready_times: dict[str, float] = {}
            for task in epoch_tasks:
                input_bindings_by_task[task.task_id] = (
                    resolve_task_input_bindings(
                        manager,
                        task.task_id,
                    )
                )
                ready_times[task.task_id] = ready_at(task)
            carry_in_reservations = tuple(
                reservation
                for reservations in actual_reservations_by_node.values()
                for reservation in reservations
                if reservation.finish_ms > current_time_ms + 1e-9
            )
            epoch = SchedulingEpoch(
                epoch_id=(
                    f"{workflow.workflow_id}:epoch:{epoch_sequence}"
                ),
                now_ms=current_time_ms,
                ready_tasks=epoch_tasks,
            )
            plan = plan_scheduling_epoch(
                epoch,
                optimizer=algorithm,
                node_specs=node_by_id,
                node_snapshots=snapshot_by_id,
                input_artifact_bindings=input_bindings_by_task,
                ready_time_ms=ready_times,
                node_available_ms={
                    node_id: current_time_ms for node_id in node_by_id
                },
                link_specs=resolved_link_specs,
                link_snapshots=resolved_link_snapshots,
                link_available_ms=link_available,
                existing_node_reservations=carry_in_reservations,
                critical_tail_ms=critical_tail,
                profiles=profiles,
                registry=optimizer_registry,
            )
            if plan.deferred_task_ids:
                raise RuntimeError(
                    "the deterministic engine does not commit partial "
                    "plans with deferred ready tasks"
                )
            if not plan.assignments:
                raise RuntimeError(
                    "optimizer returned no committable assignment"
                )
            actual_link_available = dict(link_available)
            task_by_id = {
                task.task_id: task for task in epoch_tasks
            }
            resource_by_task = {
                reservation.task_id: reservation
                for reservation in plan.node_reservations
            }
            transfers_by_task: dict[
                str, list[TransferReservation]
            ] = {
                task.task_id: [] for task in epoch_tasks
            }
            for reservation in plan.transfer_reservations:
                transfers_by_task[reservation.task_id].append(
                    reservation
                )
            ordered_assignments = sorted(
                plan.assignments,
                key=lambda item: (
                    item.estimated_start_ms,
                    item.task_id,
                ),
            )
            if workflow.failure_policy is FailurePolicy.FAIL_FAST:
                drop_assignments = [
                    item
                    for item in ordered_assignments
                    if not item.target_node_id
                ]
                ordered_assignments = [
                    drop_assignments[0]
                    if drop_assignments
                    else ordered_assignments[0]
                ]
                force_completion_before_replan = True
            for assignment in ordered_assignments:
                task = task_by_id[assignment.task_id]
                released_at = ready_times[task.task_id]
                artifacts = artifacts_from_bindings(
                    input_bindings_by_task[task.task_id]
                )
                manager.mark_running(task.task_id)
                if not assignment.target_node_id:
                    sequence += 1
                    heapq.heappush(
                        events,
                        (
                            max(current_time_ms, released_at),
                            sequence,
                            task.task_id,
                            False,
                            assignment,
                        ),
                    )
                    continue

                jitter = max(0.75, rng.gauss(1.0, network_jitter * 0.35))
                noise = rng.uniform(1.0 - resource_noise, 1.0 + resource_noise)
                compute_ms = assignment.compute_ms * noise
                transfer_cursor = released_at
                transfer_starts: list[float] = []
                communication_ms = 0.0
                for reservation in sorted(
                    transfers_by_task[task.task_id],
                    key=lambda item: (
                        item.start_ms,
                        item.reservation_id,
                    ),
                ):
                    duration_ms = (
                        reservation.finish_ms - reservation.start_ms
                    ) * jitter
                    transfer_start = max(
                        transfer_cursor,
                        reservation.start_ms,
                        *(
                            actual_link_available.get(
                                link_id,
                                current_time_ms,
                            )
                            for link_id in reservation.path_link_ids
                        ),
                    )
                    transfer_finish = transfer_start + duration_ms
                    transfer_starts.append(transfer_start)
                    communication_ms += duration_ms
                    transfer_cursor = transfer_finish
                    for link_id in reservation.path_link_ids:
                        actual_link_available[link_id] = transfer_finish

                resource = resource_by_task[task.task_id]
                compute_start = _earliest_resource_start(
                    node_by_id[assignment.target_node_id],
                    resource.demand,
                    max(
                        released_at,
                        resource.start_ms,
                        transfer_cursor,
                    ),
                    compute_ms,
                    actual_reservations_by_node[
                        assignment.target_node_id
                    ],
                )
                finish_ms = compute_start + compute_ms
                start_ms = min(
                    [compute_start, *transfer_starts]
                )
                actual_reservations_by_node[
                    assignment.target_node_id
                ].append(
                    replace(
                        resource,
                        start_ms=compute_start,
                        finish_ms=finish_ms,
                    )
                )
                assignment = replace(
                    assignment,
                    estimated_start_ms=start_ms,
                    estimated_finish_ms=finish_ms,
                    compute_ms=compute_ms,
                    communication_ms=communication_ms,
                    energy_j=assignment.energy_j * noise,
                )
                busy_by_node[assignment.target_node_id] += compute_ms
                snapshot_by_id[assignment.target_node_id] = apply_load(
                    snapshot_by_id[assignment.target_node_id],
                    task,
                )
                target = node_by_id[assignment.target_node_id]
                target_snapshot = snapshot_by_id[assignment.target_node_id]
                selected_profile = profiles.lookup(task.spec.task_type, target.kind) if profiles else None
                profile_sources_used.add(
                    selected_profile.provenance if selected_profile else "demand_formula_fallback"
                )
                if assignment.target_node_id != task.source_node_id:
                    total_bandwidth_mb += sum(a.size_mb for a in artifacts) if artifacts else task.spec.input_size_mb

                deadline_missed = finish_ms > task.deadline_time_ms
                thermal_penalty = 0.2 if target_snapshot.temperature_c > 93.0 else 0.0
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
                    queue_delay_ms=round(max(0.0, start_ms - released_at), 2),
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
            link_available = actual_link_available
            if not force_completion_before_replan:
                continue

        next_arrival_ms = (
            float("inf")
            if force_completion_before_replan
            else min(
                (ready_at(task) for task in ready),
                default=float("inf"),
            )
        )
        next_completion_ms = events[0][0] if events else float("inf")
        if next_arrival_ms < next_completion_ms:
            current_time_ms = max(current_time_ms, next_arrival_ms)
            continue
        if not events:
            raise RuntimeError("workflow is unresolved but has neither arrivals nor completion events")

        finish_ms, _, task_id, ok, assignment = heapq.heappop(events)
        current_time_ms = max(current_time_ms, finish_ms)
        if manager.state_of(task_id) is TaskState.SKIPPED:
            continue
        task = manager.get(task_id)
        dropped = not assignment.target_node_id
        outputs: tuple[ArtifactRef, ...] = ()
        if ok and not dropped:
            declared = task.spec.output_ports or ()
            output_count = max(1, len(declared))
            ports = declared or (None,)
            outputs = tuple(
                ArtifactRef(
                    artifact_id=(
                        f"artifact:{workflow.workflow_id}:{task_id}:"
                        f"{port.name if port is not None else 'result'}"
                    ),
                    producer_task_id=task_id,
                    node_id=assignment.target_node_id,
                    size_mb=task.spec.output_size_mb / output_count,
                    uri=(
                        f"node://{assignment.target_node_id}/{workflow.workflow_id}/"
                        f"{task_id}/{port.name if port is not None else 'result'}"
                    ),
                    producer_port=port.name if port is not None else "result",
                    message_type=port.message_type if port is not None else "",
                )
                for port in ports
            )
        released, skipped = manager.complete(
            task_id,
            ok=ok and not dropped,
            finished_time_ms=finish_ms,
            outputs=outputs,
            timed_out=(finish_ms > task.deadline_time_ms and not ok),
            dropped=dropped,
            error_code="no_feasible_node" if dropped else ("execution_failed" if not ok else ""),
        )
        completion_time[task_id] = finish_ms
        if task_id not in records:
            records[task_id] = _empty_record(task, state=manager.state_of(task_id), reason=assignment.reason)
        record = records[task_id]
        if dropped:
            record.start_time_ms = round(finish_ms, 2)
            record.finish_time_ms = round(finish_ms, 2)
            record.queue_delay_ms = round(
                max(0.0, finish_ms - task.arrival_time_ms),
                2,
            )
            record.total_latency_ms = round(
                max(0.0, finish_ms - task.arrival_time_ms),
                2,
            )
        record.state = manager.state_of(task_id).value
        record.success = manager.state_of(task_id) is TaskState.SUCCEEDED
        record.output_ref = outputs[0].uri if outputs else ""
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
    missed = sum(record.deadline_missed for record in ordered_records)
    offloaded = sum(record.mode == "edge" for record in ordered_records)
    workflow_task_by_id = {
        task.task_id: task for task in workflow.tasks
    }
    safety_violations = sum(
        _violates_safety_contract(
            workflow_task_by_id[record.task_id],
            record.target_node_id,
            node_by_id,
        )
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
        node_utilization={
            node_id: round(
                busy
                / max(
                    1.0,
                    makespan * node_by_id[node_id].max_concurrency,
                ),
                4,
            )
            for node_id, busy in busy_by_node.items()
        },
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
            "active": "deterministic_event_engine",
            "profile_source": (
                next(iter(profile_sources_used))
                if len(profile_sources_used) == 1
                else "mixed"
                if profile_sources_used
                else "not_used"
            ),
            "profile_sources": sorted(profile_sources_used),
            "profile_catalog_provenance": profiles.provenance if profiles is not None else "unavailable",
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
        deadline_missed=state is TaskState.TIMEOUT,
        success=False,
        state=state.value,
        reason=reason,
        input_locations=[],
        output_ref="",
    )


def _earliest_resource_start(
    node: NodeSpec,
    demand: ResourceDemand,
    earliest_ms: float,
    duration_ms: float,
    existing: list[PlannedResourceReservation],
) -> float:
    """Place one noisy compute interval without violating node capacity."""

    cursor = earliest_ms
    while True:
        finish = cursor + duration_ms
        overlapping = [
            reservation
            for reservation in existing
            if reservation.start_ms < finish - 1e-9
            and reservation.finish_ms > cursor + 1e-9
        ]
        boundaries = sorted(
            {
                cursor,
                *(
                    max(cursor, reservation.start_ms)
                    for reservation in overlapping
                ),
                *(
                    min(finish, reservation.finish_ms)
                    for reservation in overlapping
                ),
            }
        )
        feasible = True
        for point in boundaries:
            if point >= finish - 1e-9:
                continue
            active = [
                reservation
                for reservation in overlapping
                if reservation.start_ms <= point + 1e-9
                and reservation.finish_ms > point + 1e-9
            ]
            if (
                len(active) + 1 > node.max_concurrency
                or sum(item.demand.cpu_units for item in active)
                + demand.cpu_units
                > node.cpu_capacity + 1e-9
                or sum(item.demand.gpu_units for item in active)
                + demand.gpu_units
                > node.gpu_capacity + 1e-9
                or sum(item.demand.memory_gb for item in active)
                + demand.memory_gb
                > node.memory_gb + 1e-9
            ):
                feasible = False
                break
        if feasible:
            return cursor
        releases = [
            reservation.finish_ms
            for reservation in overlapping
            if reservation.finish_ms > cursor + 1e-9
        ]
        if not releases:
            return cursor
        cursor = min(releases)


def _violates_safety_contract(
    task: TaskInstance,
    target_node_id: str,
    node_by_id: dict[str, NodeSpec],
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


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
