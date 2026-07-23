"""Candidate generation and optimizer-neutral scheduling orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

from .dag import DagIndex
from .models import (
    ArtifactRef,
    Assignment,
    LinkSnapshot,
    LinkSpec,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    TaskInstance,
    resolved_placement_constraints,
    task_resource_demand,
)
from .network import NetworkTopology, synthesize_legacy_full_mesh
from .optimizers import (
    CandidateEstimate,
    Optimizer,
    OptimizerRegistry,
    PlanValidationError,
    PlannedResourceReservation,
    ResourceDemand,
    SchedulingEpoch,
    SchedulingPlan,
    SchedulingProblem,
    built_in_registry,
    validate_plan,
)
from .profiling import ProfileCatalog


def allowed_nodes(
    task: TaskInstance,
    node_specs: Iterable[NodeSpec],
    node_snapshots: dict[str, NodeSnapshot],
    excluded_node_ids: frozenset[str] = frozenset(),
) -> list[NodeSpec]:
    specs = tuple(node_specs)
    source = next((node for node in specs if node.node_id == task.source_node_id), None)
    if source is None:
        return []
    constraints = resolved_placement_constraints(task)
    candidates: list[NodeSpec] = []
    for spec in specs:
        snapshot = node_snapshots.get(spec.node_id)
        if (
            spec.node_id in excluded_node_ids
            or snapshot is None
            or not snapshot.online
        ):
            continue
        if constraints.pinned_node_id:
            if spec.node_id != constraints.pinned_node_id:
                continue
            if (
                constraints.allowed_node_kinds
                and spec.kind not in constraints.allowed_node_kinds
            ):
                continue
        elif spec.node_id == task.source_node_id:
            if not constraints.allow_source_node:
                continue
        else:
            if spec.kind is NodeKind.ROBOT and not constraints.allow_other_robots:
                continue
            if spec.kind not in constraints.allowed_node_kinds:
                continue
        if (
            not constraints.pinned_node_id
            and constraints.preferred_node_kinds
            and not constraints.allow_fallback
            and spec.kind not in constraints.preferred_node_kinds
        ):
            continue
        if constraints.safety_required and not spec.safety_capable:
            continue
        capabilities = set(spec.capabilities)
        if spec.safety_capable:
            capabilities.add("local_safety")
        if not set(constraints.required_capabilities).issubset(capabilities):
            continue
        candidates.append(spec)
    preferred_rank = {
        kind: rank
        for rank, kind in enumerate(constraints.preferred_node_kinds)
    }
    candidates.sort(
        key=lambda node: (
            preferred_rank.get(node.kind, len(preferred_rank)),
            node.node_id,
        )
    )
    return candidates


def estimate_candidate(
    task: TaskInstance,
    node: NodeSpec,
    *,
    ready_time_ms: float,
    node_available_ms: float,
    node_specs: dict[str, NodeSpec],
    node_snapshots: dict[str, NodeSnapshot],
    parent_artifacts: Iterable[ArtifactRef],
    profiles: ProfileCatalog | None = None,
    topology: NetworkTopology | None = None,
) -> CandidateEstimate:
    snapshot = node_snapshots.get(node.node_id)
    if snapshot is None or not snapshot.online:
        return _infeasible(task, node, "node_offline")
    if node not in allowed_nodes(
        task,
        node_specs.values(),
        node_snapshots,
    ):
        constraints = resolved_placement_constraints(task)
        reason = (
            "local_safety_requires_safety_capable_source_robot"
            if constraints.safety_required
            else "placement_constraints_reject_node"
        )
        return _infeasible(task, node, reason)
    if (
        task.spec.model_requirement
        and node.supported_models
        and task.spec.model_requirement not in node.supported_models
    ):
        return _infeasible(task, node, "model_not_supported")
    if task.spec.gpu_demand > node.gpu_capacity + 1e-9:
        return _infeasible(task, node, "gpu_capacity_insufficient")

    util_penalty = 1.0 + 2.2 * max(
        snapshot.cpu_util,
        snapshot.gpu_util,
        snapshot.memory_util,
    )
    profile = profiles.lookup(task.spec.task_type, node.kind) if profiles is not None else None
    if profile is not None and not profile.supported:
        return _infeasible(task, node, "profile_marks_task_unsupported")
    if profile is not None:
        compute_ms = profile.p95_ms * util_penalty
    else:
        capacity = max(0.15, node.cpu_capacity + 1.5 * node.gpu_capacity)
        gpu_pressure = 1.0 + max(0.0, task.spec.gpu_demand - node.gpu_capacity) * 2.5
        compute_ms = 100.0 * task.spec.compute_demand / capacity * util_penalty * gpu_pressure

    artifacts = tuple(parent_artifacts)
    transfer_inputs: list[tuple[str, str, float]]
    if artifacts:
        transfer_inputs = [
            (artifact.artifact_id, artifact.node_id, artifact.size_mb)
            for artifact in artifacts
        ]
    elif task.dependency_task_ids:
        return _infeasible(
            task,
            node,
            "dependency_artifact_unavailable",
        )
    else:
        transfer_inputs = [
            (
                f"input:{task.workflow_id}:{task.task_id}",
                task.source_node_id,
                task.spec.input_size_mb,
            )
        ]

    if topology is None:
        legacy_specs, legacy_snapshots = synthesize_legacy_full_mesh(
            node_specs.values(),
            node_snapshots.values(),
        )
        topology = NetworkTopology(
            node_specs,
            legacy_specs,
            legacy_snapshots,
            node_online={
                node_id: snapshot.online
                for node_id, snapshot in node_snapshots.items()
            },
        )
    communication_ms = 0.0
    locations: list[str] = []
    transfers = []
    for transfer_id, source_id, size_mb in transfer_inputs:
        locations.append(source_id)
        source_snapshot = node_snapshots.get(source_id)
        if source_snapshot is None or not source_snapshot.online:
            return _infeasible(
                task,
                node,
                f"input_source_unavailable:{source_id}",
            )
        transfer = topology.estimate(
            transfer_id=transfer_id,
            source_node_id=source_id,
            target_node_id=node.node_id,
            size_mb=size_mb,
            minimum_bandwidth_mbps=task.spec.bandwidth_requirement_mbps,
        )
        if not transfer.feasible:
            return _infeasible(task, node, transfer.reason)
        transfers.append(transfer)
        communication_ms += transfer.transfer_time_ms

    start_ms = max(ready_time_ms, node_available_ms)
    finish_ms = start_ms + communication_ms + compute_ms
    power = max(1.0, snapshot.power_w)
    profiled_energy = profile.energy_j * util_penalty if profile is not None else compute_ms / 1000.0 * power
    energy_j = profiled_energy + communication_ms * 0.015
    return CandidateEstimate(
        task_id=task.task_id,
        node_id=node.node_id,
        node_kind=node.kind,
        source_node_id=task.source_node_id,
        feasible=True,
        ready_time_ms=ready_time_ms,
        start_ms=start_ms,
        finish_ms=finish_ms,
        compute_ms=compute_ms,
        communication_ms=communication_ms,
        energy_j=energy_j,
        resource_demand=_resource_demand(task, node),
        input_locations=tuple(locations),
        transfers=tuple(transfers),
    )


def build_scheduling_problem(
    epoch: SchedulingEpoch,
    *,
    node_specs: Mapping[str, NodeSpec],
    node_snapshots: Mapping[str, NodeSnapshot],
    parent_artifacts: Mapping[str, Iterable[ArtifactRef]],
    ready_time_ms: Mapping[str, float],
    node_available_ms: Mapping[str, float] | None = None,
    link_specs: Iterable[LinkSpec] | None = None,
    link_snapshots: Iterable[LinkSnapshot] | None = None,
    link_available_ms: Mapping[str, float] | None = None,
    existing_node_reservations: Iterable[
        PlannedResourceReservation
    ] = (),
    critical_tail_ms: Mapping[str, float] | None = None,
    profiles: ProfileCatalog | None = None,
    excluded_node_ids: Mapping[str, frozenset[str]] | None = None,
    solve_budget_ms: float = 50.0,
) -> SchedulingProblem:
    """Build the one canonical optimization input for a ready-task epoch."""

    if (link_specs is None) != (link_snapshots is None):
        raise ValueError(
            "link_specs and link_snapshots must both be provided or omitted"
        )
    if link_specs is None:
        resolved_link_specs, resolved_link_snapshots = (
            synthesize_legacy_full_mesh(
                node_specs.values(),
                node_snapshots.values(),
            )
        )
    else:
        resolved_link_specs = tuple(link_specs)
        resolved_link_snapshots = tuple(link_snapshots or ())
    topology = NetworkTopology(
        node_specs,
        resolved_link_specs,
        resolved_link_snapshots,
        node_online={
            node_id: snapshot.online
            for node_id, snapshot in node_snapshots.items()
        },
    )
    exclusions = excluded_node_ids or {}
    candidate_map: dict[str, tuple[CandidateEstimate, ...]] = {}
    for task in epoch.ready_tasks:
        task_ready = max(
            epoch.now_ms,
            ready_time_ms.get(task.task_id, task.arrival_time_ms),
        )
        candidates = allowed_nodes(
            task,
            node_specs.values(),
            dict(node_snapshots),
            exclusions.get(task.task_id, frozenset()),
        )
        task_artifacts = tuple(
            parent_artifacts.get(task.task_id, ())
        )
        candidate_map[task.task_id] = tuple(
            estimate_candidate(
                task,
                node,
                ready_time_ms=task_ready,
                node_available_ms=(
                    node_available_ms or {}
                ).get(node.node_id, epoch.now_ms),
                node_specs=dict(node_specs),
                node_snapshots=dict(node_snapshots),
                parent_artifacts=task_artifacts,
                profiles=profiles,
                topology=topology,
            )
            for node in candidates
        )
    return SchedulingProblem(
        epoch=epoch,
        node_specs=tuple(node_specs.values()),
        node_snapshots=tuple(node_snapshots.values()),
        link_specs=resolved_link_specs,
        link_snapshots=resolved_link_snapshots,
        candidates=candidate_map,
        node_available_ms={
            node_id: (node_available_ms or {}).get(node_id, epoch.now_ms)
            for node_id in node_specs
        },
        link_available_ms={
            spec.link_id: (link_available_ms or {}).get(
                spec.link_id,
                epoch.now_ms,
            )
            for spec in resolved_link_specs
        },
        existing_node_reservations=tuple(
            existing_node_reservations
        ),
        critical_tail_ms={
            task.task_id: (critical_tail_ms or {}).get(
                task.task_id,
                0.0,
            )
            for task in epoch.ready_tasks
        },
        solve_budget_ms=solve_budget_ms,
    )


def plan_scheduling_epoch(
    epoch: SchedulingEpoch,
    *,
    optimizer: str | Optimizer,
    node_specs: Mapping[str, NodeSpec],
    node_snapshots: Mapping[str, NodeSnapshot],
    parent_artifacts: Mapping[str, Iterable[ArtifactRef]],
    ready_time_ms: Mapping[str, float],
    node_available_ms: Mapping[str, float] | None = None,
    link_specs: Iterable[LinkSpec] | None = None,
    link_snapshots: Iterable[LinkSnapshot] | None = None,
    link_available_ms: Mapping[str, float] | None = None,
    existing_node_reservations: Iterable[
        PlannedResourceReservation
    ] = (),
    critical_tail_ms: Mapping[str, float] | None = None,
    profiles: ProfileCatalog | None = None,
    excluded_node_ids: Mapping[str, frozenset[str]] | None = None,
    solve_budget_ms: float = 50.0,
    registry: OptimizerRegistry | None = None,
    fallback_optimizer: str | Optimizer | None = "dag_deadline",
) -> SchedulingPlan:
    """Solve and validate one epoch through a replaceable optimizer.

    A structurally invalid plug-in plan is never committed. Unless disabled,
    the epoch is repaired by re-solving it with the declared fallback
    optimizer and recording the rejected optimizer in plan diagnostics.
    """

    problem = build_scheduling_problem(
        epoch,
        node_specs=node_specs,
        node_snapshots=node_snapshots,
        parent_artifacts=parent_artifacts,
        ready_time_ms=ready_time_ms,
        node_available_ms=node_available_ms,
        link_specs=link_specs,
        link_snapshots=link_snapshots,
        link_available_ms=link_available_ms,
        existing_node_reservations=existing_node_reservations,
        critical_tail_ms=critical_tail_ms,
        profiles=profiles,
        excluded_node_ids=excluded_node_ids,
        solve_budget_ms=solve_budget_ms,
    )
    active_registry = built_in_registry()
    if registry is not None:
        active_registry.extend(registry, replace=True)
    selected = active_registry.resolve(optimizer)
    try:
        return _solve_validated(problem, selected)
    except Exception as rejected:
        if fallback_optimizer is None:
            raise
        fallback = _resolve_fallback(
            fallback_optimizer,
            active_registry,
        )
        if fallback is selected:
            raise
        try:
            repaired = _solve_validated(problem, fallback)
        except Exception as fallback_error:
            raise RuntimeError(
                f"optimizer {selected.optimizer_id!r} failed and fallback "
                f"{fallback.optimizer_id!r} also failed"
            ) from fallback_error
        return replace(
            repaired,
            diagnostics={
                **repaired.diagnostics,
                "repaired_from_optimizer": selected.optimizer_id,
                "repair_reason": (
                    f"{type(rejected).__name__}: {rejected}"
                ),
                "fallback_optimizer": fallback.optimizer_id,
            },
        )


def choose_assignment(
    task: TaskInstance,
    *,
    algorithm: str,
    ready_time_ms: float,
    node_available: dict[str, float],
    node_specs: dict[str, NodeSpec],
    node_snapshots: dict[str, NodeSnapshot],
    parent_artifacts: Iterable[ArtifactRef],
    critical_tail_ms: float = 0.0,
    profiles: ProfileCatalog | None = None,
    excluded_node_ids: frozenset[str] = frozenset(),
    link_specs: Iterable[LinkSpec] | None = None,
    link_snapshots: Iterable[LinkSnapshot] | None = None,
) -> Assignment:
    epoch = SchedulingEpoch(
        epoch_id=f"single:{task.workflow_id}:{task.task_id}",
        now_ms=ready_time_ms,
        ready_tasks=(task,),
    )
    plan = plan_scheduling_epoch(
        epoch,
        optimizer=algorithm,
        node_specs=node_specs,
        node_snapshots=node_snapshots,
        parent_artifacts={task.task_id: tuple(parent_artifacts)},
        ready_time_ms={task.task_id: ready_time_ms},
        node_available_ms=node_available,
        link_specs=link_specs,
        link_snapshots=link_snapshots,
        critical_tail_ms={task.task_id: critical_tail_ms},
        profiles=profiles,
        excluded_node_ids={task.task_id: excluded_node_ids},
    )
    return plan.assignments[0]


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
    # Candidate scoring uses successor cost and excludes the task's own cost.
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


def _resource_demand(
    task: TaskInstance,
    node: NodeSpec,
) -> ResourceDemand:
    cpu_units, gpu_units, memory_gb = task_resource_demand(task, node)
    return ResourceDemand(
        cpu_units=cpu_units,
        gpu_units=gpu_units,
        memory_gb=memory_gb,
    )


def _infeasible(
    task: TaskInstance,
    node: NodeSpec,
    reason: str,
) -> CandidateEstimate:
    return CandidateEstimate(
        task_id=task.task_id,
        node_id=node.node_id,
        node_kind=node.kind,
        source_node_id=task.source_node_id,
        feasible=False,
        ready_time_ms=0.0,
        start_ms=0.0,
        finish_ms=float("inf"),
        compute_ms=0.0,
        communication_ms=0.0,
        energy_j=0.0,
        resource_demand=ResourceDemand(0.0, 0.0, 0.0),
        input_locations=(),
        reason=reason,
    )


def _solve_validated(
    problem: SchedulingProblem,
    optimizer: Optimizer,
) -> SchedulingPlan:
    plan = optimizer.solve(problem)
    if plan.optimizer_id != optimizer.optimizer_id:
        raise PlanValidationError(
            "plan optimizer_id does not match the selected optimizer"
        )
    return validate_plan(problem, plan)


def _resolve_fallback(
    optimizer: str | Optimizer,
    registry: OptimizerRegistry,
) -> Optimizer:
    if not isinstance(optimizer, str):
        return registry.resolve(optimizer)
    if optimizer == "dag_deadline":
        return built_in_registry().resolve(optimizer)
    try:
        return registry.resolve(optimizer)
    except KeyError:
        return built_in_registry().resolve(optimizer)
