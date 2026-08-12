"""Candidate generation and optimizer-neutral scheduling orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass, replace
import enum
import hashlib
import json
import math
from time import perf_counter

from .dag import DagIndex
from .domain.artifact import (
    ArtifactRef,
    InputArtifactBinding,
    artifacts_from_bindings,
)
from .domain.execution import Assignment, task_resource_demand
from .domain.task import TaskInstance, resolved_placement_constraints
from .domain.topology import (
    LinkSnapshot,
    LinkSpec,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
)
from .network import NetworkTopology, synthesize_legacy_full_mesh
from .optimizers import (
    CandidateEstimate,
    FormulatedOptimizer,
    FormulationCompatibilityError,
    FormulationRegistry,
    SchedulingFormulation,
    Optimizer,
    OptimizerRegistry,
    OptimizerSolveState,
    ONE_HOT_PLACEMENT_SPEC,
    PlanValidationError,
    PlannedResourceReservation,
    ResourceDemand,
    SchedulingEpoch,
    SchedulingPlan,
    SchedulingPolicy,
    SchedulingProblem,
    SchedulingSnapshot,
    SolveLimits,
    SolveStatus,
    SolveTraceContext,
    SolveTracePhase,
    StatefulOptimizer,
    StatefulFormulatedOptimizer,
    algorithm_aliases,
    built_in_policy,
    built_in_registry,
    built_in_formulation_registry,
    build_solve_request,
    compile_solve_request,
    formulation_failure_status,
    metric_contract_id,
    validate_plan,
)
from .profiling import ExecutionProfile, ProfileCatalog


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
    util_penalty = 1.0 + 2.2 * max(
        snapshot.cpu_util,
        snapshot.gpu_util,
        snapshot.memory_util,
    )
    profile = profiles.lookup(task.spec.task_type, node.kind) if profiles is not None else None
    if profile is not None and not profile.supported:
        return _infeasible(task, node, "profile_marks_task_unsupported")
    resource_demand = _resource_demand(task, node, profile)
    if resource_demand.cpu_units > node.cpu_capacity + 1e-9:
        return _infeasible(task, node, "cpu_capacity_insufficient")
    if resource_demand.gpu_units > node.gpu_capacity + 1e-9:
        return _infeasible(task, node, "gpu_capacity_insufficient")
    if resource_demand.memory_gb > node.memory_gb + 1e-9:
        return _infeasible(task, node, "memory_capacity_insufficient")
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
        resource_demand=resource_demand,
        output_size_mb=(
            profile.output_size_mb
            if profile is not None
            else task.spec.output_size_mb
        ),
        success_probability=(
            1.0 - profile.failure_rate
            if profile is not None
            else 1.0
        ),
        input_locations=tuple(locations),
        transfers=tuple(transfers),
    )


def build_scheduling_problem(
    epoch: SchedulingEpoch,
    *,
    node_specs: Mapping[str, NodeSpec],
    node_snapshots: Mapping[str, NodeSnapshot],
    parent_artifacts: Mapping[str, Iterable[ArtifactRef]] | None = None,
    input_artifact_bindings: Mapping[
        str, Iterable[InputArtifactBinding]
    ] | None = None,
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
    policy: str | SchedulingPolicy | None = None,
    solve_limits: SolveLimits | None = None,
    solve_budget_ms: float | None = None,
) -> SchedulingProblem:
    """Build the optimization input for a ready-task epoch.

    ``input_artifact_bindings`` is the port-specific input. ``parent_artifacts``
    remains a flat compatibility input and is normalized immediately; the
    resulting Snapshot stores only exact port bindings.
    """

    if solve_limits is not None and solve_budget_ms is not None:
        raise ValueError(
            "provide solve_limits or solve_budget_ms, not both"
        )
    resolved_policy = _resolve_policy(policy)
    resolved_input_bindings = _normalize_input_artifact_bindings(
        epoch,
        input_artifact_bindings=input_artifact_bindings,
        parent_artifacts=parent_artifacts,
    )
    resolved_solve_limits = (
        solve_limits
        if solve_limits is not None
        else SolveLimits(
            solve_budget_ms=(
                50.0
                if solve_budget_ms is None
                else solve_budget_ms
            )
        )
    )
    resolved_metric_contract_id = metric_contract_id(resolved_policy)
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
        task_artifacts = artifacts_from_bindings(
            resolved_input_bindings[task.task_id]
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
    resolved_node_specs = tuple(node_specs.values())
    resolved_node_snapshots = tuple(node_snapshots.values())
    resolved_node_available = {
        node_id: (node_available_ms or {}).get(
            node_id,
            epoch.now_ms,
        )
        for node_id in node_specs
    }
    resolved_link_available = {
        spec.link_id: (link_available_ms or {}).get(
            spec.link_id,
            epoch.now_ms,
        )
        for spec in resolved_link_specs
    }
    resolved_existing_reservations = tuple(
        existing_node_reservations
    )
    resolved_critical_tail = {
        task.task_id: (critical_tail_ms or {}).get(
            task.task_id,
            0.0,
        )
        for task in epoch.ready_tasks
    }
    snapshot_digest = _contract_digest(
        {
            "schema_version": "mars.scheduling-snapshot.v1",
            "captured_at_ms": epoch.now_ms,
            "epoch": epoch,
            "node_specs": resolved_node_specs,
            "node_snapshots": resolved_node_snapshots,
            "link_specs": resolved_link_specs,
            "link_snapshots": resolved_link_snapshots,
            "candidates": candidate_map,
            "input_artifact_bindings": resolved_input_bindings,
            "node_available_ms": resolved_node_available,
            "link_available_ms": resolved_link_available,
            "existing_node_reservations": (
                resolved_existing_reservations
            ),
            "critical_tail_ms": resolved_critical_tail,
        }
    )
    snapshot_id = f"{epoch.epoch_id}:snapshot:{snapshot_digest}"
    snapshot = SchedulingSnapshot(
        schema_version="mars.scheduling-snapshot.v1",
        snapshot_id=snapshot_id,
        captured_at_ms=epoch.now_ms,
        epoch=epoch,
        node_specs=resolved_node_specs,
        node_snapshots=resolved_node_snapshots,
        link_specs=resolved_link_specs,
        link_snapshots=resolved_link_snapshots,
        candidates=candidate_map,
        input_artifact_bindings=resolved_input_bindings,
        node_available_ms=resolved_node_available,
        link_available_ms=resolved_link_available,
        existing_node_reservations=resolved_existing_reservations,
        critical_tail_ms=resolved_critical_tail,
    )
    problem_digest = _contract_digest(
        {
            "schema_version": "mars.scheduling-problem.v1",
            "snapshot_id": snapshot_id,
            "policy": resolved_policy,
            "metric_contract_id": resolved_metric_contract_id,
            "solve_limits": resolved_solve_limits,
        }
    )
    return SchedulingProblem(
        problem_id=f"{epoch.epoch_id}:problem:{problem_digest}",
        snapshot=snapshot,
        policy=resolved_policy,
        solve_limits=resolved_solve_limits,
        metric_contract_id=resolved_metric_contract_id,
    )


def plan_scheduling_epoch(
    epoch: SchedulingEpoch,
    *,
    optimizer: str | Optimizer,
    node_specs: Mapping[str, NodeSpec],
    node_snapshots: Mapping[str, NodeSnapshot],
    parent_artifacts: Mapping[str, Iterable[ArtifactRef]] | None = None,
    input_artifact_bindings: Mapping[
        str, Iterable[InputArtifactBinding]
    ] | None = None,
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
    policy: str | SchedulingPolicy | None = None,
    formulation: str | SchedulingFormulation | None = None,
    solve_limits: SolveLimits | None = None,
    solve_budget_ms: float | None = None,
    registry: OptimizerRegistry | None = None,
    formulation_registry: FormulationRegistry | None = None,
    fallback_optimizer: str | Optimizer | None = "heuristic",
    solve_state: OptimizerSolveState | None = None,
) -> SchedulingPlan:
    """Solve and validate one epoch through a replaceable optimizer.

    A structurally invalid plug-in plan is never committed. Unless disabled,
    the epoch is repaired by re-solving it with the declared fallback
    optimizer and recording the rejected optimizer in plan diagnostics.
    """

    active_registry = built_in_registry()
    if registry is not None:
        active_registry.extend(registry, replace=True)
    active_formulation_registry = built_in_formulation_registry()
    if formulation_registry is not None:
        active_formulation_registry.extend(
            formulation_registry,
            replace=True,
        )
    selected, resolved_policy = _resolve_selection(
        optimizer,
        policy,
        active_registry,
    )
    resolved_formulation = _resolve_formulation(
        selected,
        formulation,
        active_formulation_registry,
    )
    problem = build_scheduling_problem(
        epoch,
        node_specs=node_specs,
        node_snapshots=node_snapshots,
        parent_artifacts=parent_artifacts,
        input_artifact_bindings=input_artifact_bindings,
        ready_time_ms=ready_time_ms,
        node_available_ms=node_available_ms,
        link_specs=link_specs,
        link_snapshots=link_snapshots,
        link_available_ms=link_available_ms,
        existing_node_reservations=existing_node_reservations,
        critical_tail_ms=critical_tail_ms,
        profiles=profiles,
        excluded_node_ids=excluded_node_ids,
        policy=resolved_policy,
        solve_limits=solve_limits,
        solve_budget_ms=solve_budget_ms,
    )
    orchestration_deadline = (
        perf_counter() + problem.solve_limits.solve_budget_ms / 1000.0
    )
    try:
        return _solve_validated(
            problem,
            selected,
            formulation=resolved_formulation,
            solve_state=solve_state,
            orchestration_deadline=orchestration_deadline,
        )
    except Exception as rejected:
        if fallback_optimizer is None:
            raise
        fallback = _resolve_fallback(
            fallback_optimizer,
            active_registry,
        )
        same_optimizer_contract = (
            fallback.optimizer_id == selected.optimizer_id
            and str(getattr(fallback, "optimizer_version", ""))
            == str(getattr(selected, "optimizer_version", ""))
            and str(getattr(fallback, "optimizer_config_digest", ""))
            == str(getattr(selected, "optimizer_config_digest", ""))
        )
        if same_optimizer_contract and resolved_formulation is None:
            raise
        fallback_candidates: list[SchedulingFormulation | None] = []
        preservation_error: Exception | None = None
        if resolved_formulation is not None and not same_optimizer_contract:
            try:
                fallback_candidates.append(
                    _resolve_formulation(
                        fallback,
                        resolved_formulation,
                        active_formulation_registry,
                    )
                )
            except Exception as exc:
                preservation_error = exc
        try:
            fallback_default = _resolve_formulation(
                fallback,
                None,
                active_formulation_registry,
            )
        except Exception as exc:
            fallback_default = None
            if preservation_error is None:
                preservation_error = exc
        else:
            duplicates_rejected_stack = (
                same_optimizer_contract
                and _formulation_identity(fallback_default)
                == _formulation_identity(resolved_formulation)
            )
            if not duplicates_rejected_stack and (
                not fallback_candidates
                or _formulation_identity(fallback_candidates[-1])
                != _formulation_identity(fallback_default)
            ):
                fallback_candidates.append(fallback_default)

        fallback_errors: list[Exception] = []
        repaired = None
        fallback_formulation = None
        for candidate_formulation in fallback_candidates:
            if perf_counter() >= orchestration_deadline:
                fallback_errors.append(
                    TimeoutError(
                        "shared scheduling-epoch solve budget expired before "
                        "fallback"
                    )
                )
                break
            try:
                repaired = _solve_validated(
                    problem,
                    fallback,
                    formulation=candidate_formulation,
                    solve_state=solve_state,
                    orchestration_deadline=orchestration_deadline,
                )
            except Exception as fallback_error:
                fallback_errors.append(fallback_error)
                continue
            fallback_formulation = candidate_formulation
            break
        if repaired is None:
            fallback_error = (
                fallback_errors[-1]
                if fallback_errors
                else preservation_error
                if preservation_error is not None
                else rejected
            )
            raise RuntimeError(
                f"optimizer {selected.optimizer_id!r} failed and fallback "
                f"{fallback.optimizer_id!r} also failed"
            ) from fallback_error
        requested_formulation_id = _formulation_identity(
            resolved_formulation
        )
        fallback_formulation_id = _formulation_identity(
            fallback_formulation
        )
        formulation_changed = (
            requested_formulation_id != fallback_formulation_id
        )
        # The built-in transition from exact one-hot to the legacy heuristic's
        # unformulated domain admits DROP/defer decisions and is a real domain
        # relaxation. Other future domain changes are only classified as
        # changed until a formulation relation contract exists.
        formulation_relaxed = bool(
            resolved_formulation is not None
            and resolved_formulation.spec == ONE_HOT_PLACEMENT_SPEC
            and fallback_formulation is None
        )
        audited = replace(
            repaired,
            diagnostics={
                **repaired.diagnostics,
                "repaired_from_optimizer": selected.optimizer_id,
                "repair_reason": (
                    f"{type(rejected).__name__}: {rejected}"
                ),
                "fallback_optimizer": fallback.optimizer_id,
                "repaired_from_formulation": (
                    resolved_formulation.spec.formulation_id
                    if resolved_formulation is not None
                    else ""
                ),
                "fallback_formulation": (
                    fallback_formulation.spec.formulation_id
                    if fallback_formulation is not None
                    else ""
                ),
                "repaired_from_formulation_version": (
                    resolved_formulation.spec.formulation_version
                    if resolved_formulation is not None
                    else ""
                ),
                "repaired_from_formulation_digest": requested_formulation_id,
                "fallback_formulation_version": (
                    fallback_formulation.spec.formulation_version
                    if fallback_formulation is not None
                    else ""
                ),
                "fallback_formulation_digest": fallback_formulation_id,
                "fallback_attempt_count": len(fallback_errors) + 1,
                "formulation_relaxed": formulation_relaxed,
                "formulation_changed": formulation_changed,
                "formulation_fallback_mode": (
                    "preserved"
                    if not formulation_changed
                    else "fallback_default"
                ),
            },
        )
        if solve_state is not None:
            fallback_request_id = audited.solve_request_id
            context = solve_state.latest_context(
                problem.problem_id,
                fallback.optimizer_id,
                fallback_request_id,
            )
            if context is not None:
                solve_state.record(
                    context,
                    SolveTracePhase.FALLBACK,
                    iteration=audited.iteration_count,
                    elapsed_ms=audited.solve_elapsed_ms,
                    solve_status=audited.solve_status,
                    termination_reason="fallback_solution_selected",
                    has_incumbent=bool(audited.assignments),
                    evaluated_work_units=audited.iteration_count,
                    total_work_units=(
                        int(audited.diagnostics["total_combinations"])
                        if isinstance(
                            audited.diagnostics.get("total_combinations"),
                            int,
                        )
                        else None
                    ),
                    objective_key=audited.objective_key,
                    objective_components={
                        item.objective_id: item.raw_value
                        for item in audited.objective_evaluations
                    },
                    selected_targets={
                        item.task_id: item.target_node_id
                        for item in audited.assignments
                    },
                    details={
                        "ready_task_count": len(problem.epoch.ready_tasks),
                        "repaired_from_optimizer": selected.optimizer_id,
                        "repaired_from_formulation_digest": (
                            requested_formulation_id
                        ),
                        "fallback_formulation_digest": (
                            fallback_formulation_id
                        ),
                        "formulation_relaxed": formulation_relaxed,
                        "formulation_changed": formulation_changed,
                        "fallback_attempt_count": len(fallback_errors) + 1,
                        "formulation_exhausted": bool(
                            audited.diagnostics.get(
                                "formulation_exhausted",
                                False,
                            )
                        ),
                        "communication_time_ms": sum(
                            item.communication_ms
                            for item in audited.assignments
                        ),
                    },
                )
        return audited


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
    """Deterministic utilization update used by simulation snapshots."""
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
    profile: ExecutionProfile | None = None,
) -> ResourceDemand:
    cpu_units, gpu_units, memory_gb = task_resource_demand(task, node)
    if profile is not None:
        cpu_units = (
            cpu_units if profile.cpu_units is None else profile.cpu_units
        )
        gpu_units = (
            gpu_units if profile.gpu_units is None else profile.gpu_units
        )
        memory_gb = profile.peak_memory_mb / 1024.0
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
        output_size_mb=0.0,
        success_probability=0.0,
        input_locations=(),
        reason=reason,
    )


def _solve_validated(
    problem: SchedulingProblem,
    optimizer: Optimizer,
    *,
    formulation: SchedulingFormulation | None = None,
    solve_state: OptimizerSolveState | None = None,
    orchestration_deadline: float | None = None,
) -> SchedulingPlan:
    solve_started = perf_counter()
    if (
        orchestration_deadline is not None
        and solve_started >= orchestration_deadline
    ):
        raise TimeoutError(
            "shared scheduling-epoch solve budget expired before optimizer "
            "invocation"
        )
    generic_context = None
    optimizer_version = str(getattr(optimizer, "optimizer_version", ""))
    solve_request = (
        build_solve_request(problem, optimizer, formulation)
        if formulation is not None
        else None
    )
    trusted_domain_validator = (
        formulation.validate_plan_domain
        if formulation is not None
        else None
    )
    prepared = None
    use_stateful_formulated_api = (
        solve_request is not None
        and solve_state is not None
        and isinstance(optimizer, StatefulFormulatedOptimizer)
    )
    use_stateful_api = (
        solve_request is None
        and solve_state is not None
        and isinstance(optimizer, StatefulOptimizer)
    )
    if solve_state is not None:
        generic_context = solve_state.begin(
            problem,
            optimizer_id=optimizer.optimizer_id,
            optimizer_version=optimizer_version,
            work_unit=str(
                getattr(optimizer, "solve_work_unit", "iteration")
            ),
            solve_request=(
                solve_request
            ),
        )

    try:
        if formulation is not None:
            assert solve_request is not None
            prepared = compile_solve_request(
                solve_request,
                formulation,
                solve_deadline_monotonic=orchestration_deadline,
            )
        if use_stateful_formulated_api:
            assert solve_state is not None
            assert generic_context is not None
            plan = optimizer.solve_formulated_with_state(
                prepared,
                solve_state,
                context=generic_context,
            )
        elif solve_request is not None:
            assert prepared is not None
            if not isinstance(optimizer, FormulatedOptimizer):
                raise FormulationCompatibilityError(
                    f"optimizer {optimizer.optimizer_id!r} does not support "
                    "compiled formulations"
                )
            plan = optimizer.solve_formulated(prepared)
        elif use_stateful_api:
            assert solve_state is not None
            assert generic_context is not None
            plan = optimizer.solve_with_state(
                problem,
                solve_state,
                context=generic_context,
            )
        else:
            plan = optimizer.solve(problem)
        actual_elapsed_ms = (perf_counter() - solve_started) * 1000.0
        if actual_elapsed_ms > plan.solve_elapsed_ms:
            plan = replace(plan, solve_elapsed_ms=actual_elapsed_ms)
        shared_deadline_exceeded = (
            orchestration_deadline is not None
            and perf_counter() >= orchestration_deadline
        )
        # A TIME_LIMIT incumbent may consume its own invocation budget, but it
        # may not cross the single budget shared by the primary and every
        # fallback attempt. Accepting a late incumbent would make fallback
        # multiply the caller's declared scheduling-epoch deadline.
        invocation_budget_exceeded = (
            actual_elapsed_ms >= problem.solve_limits.solve_budget_ms
        )
        if shared_deadline_exceeded or (
            invocation_budget_exceeded
            and plan.solve_status is not SolveStatus.TIME_LIMIT
        ):
            raise TimeoutError(
                f"optimizer {optimizer.optimizer_id!r} exceeded the shared "
                f"solve budget: elapsed_ms={actual_elapsed_ms:.3f}, "
                f"budget_ms={problem.solve_limits.solve_budget_ms:.3f}"
            )
    except Exception as exc:
        if solve_state is not None:
            context = solve_state.latest_context(
                problem.problem_id,
                optimizer.optimizer_id,
                (
                    solve_request.solve_request_id
                    if solve_request is not None
                    else ""
                ),
            )
            latest = next(
                (
                    entry
                    for entry in reversed(solve_state.entries)
                    if context is not None
                    and entry.context.solve_id == context.solve_id
                ),
                None,
            )
            already_failed = (
                latest is not None
                and latest.phase is SolveTracePhase.FAILED
            )
            if not already_failed:
                if generic_context is None:
                    generic_context = context
                assert generic_context is not None
                solve_state.record(
                    generic_context,
                    SolveTracePhase.FAILED,
                    elapsed_ms=(perf_counter() - solve_started) * 1000.0,
                    solve_status=formulation_failure_status(exc),
                    termination_reason=f"{type(exc).__name__}: {exc}",
                    details={
                        "ready_task_count": len(problem.epoch.ready_tasks),
                    },
                )
        raise

    if (
        solve_state is not None
        and not use_stateful_api
        and not use_stateful_formulated_api
    ):
        assert generic_context is not None
        _record_plan_trace(
            solve_state,
            generic_context,
            SolveTracePhase.COMPLETED,
            plan,
            problem,
        )

    try:
        if plan.optimizer_id != optimizer.optimizer_id:
            raise PlanValidationError(
                "plan optimizer_id does not match the selected optimizer"
            )
        # ``validate_plan(problem, plan)`` remains compatible with direct
        # formulated optimizer calls, where the caller has no request object.
        # The orchestrator does know the selected mode, so it must reject an
        # unformulated plug-in that invents otherwise unverifiable provenance.
        if solve_request is None and any(
            (
                plan.solve_request_id,
                plan.formulation_id,
                plan.formulation_version,
                plan.formulation_digest,
            )
        ):
            raise PlanValidationError(
                "an unformulated plan cannot claim formulation provenance"
            )
        if formulation is not None:
            assert prepared is not None
            assert trusted_domain_validator is not None
            trusted_domain_validator(
                problem,
                prepared.model,
                plan,
            )
        validated = validate_plan(
            problem,
            plan,
            solve_request=(
                solve_request
            ),
        )
    except Exception as exc:
        if solve_state is not None:
            context = generic_context
            if context is None:
                context = solve_state.latest_context(
                    problem.problem_id,
                    optimizer.optimizer_id,
                    (
                        solve_request.solve_request_id
                        if solve_request is not None
                        else ""
                    ),
                )
            if context is not None:
                _record_plan_trace(
                    solve_state,
                    context,
                    SolveTracePhase.REJECTED,
                    plan,
                    problem,
                    termination_reason=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
        raise

    if solve_state is not None:
        context = generic_context
        if context is None:
            context = solve_state.latest_context(
                problem.problem_id,
                optimizer.optimizer_id,
                (
                    solve_request.solve_request_id
                    if solve_request is not None
                    else ""
                ),
            )
        if context is not None:
            _record_plan_trace(
                solve_state,
                context,
                SolveTracePhase.VALIDATED,
                validated,
                problem,
            )
    return validated


def _record_plan_trace(
    state: OptimizerSolveState,
    context: SolveTraceContext,
    phase: SolveTracePhase,
    plan: SchedulingPlan,
    problem: SchedulingProblem,
    *,
    termination_reason: str | None = None,
) -> None:
    total_combinations = plan.diagnostics.get("total_combinations")
    reported_status = plan.solve_status
    effective_status = (
        SolveStatus.ERROR
        if phase is SolveTracePhase.REJECTED
        else reported_status
    )
    state.record(
        context,
        phase,
        iteration=plan.iteration_count,
        elapsed_ms=plan.solve_elapsed_ms,
        solve_status=effective_status,
        termination_reason=(
            plan.termination_reason
            if termination_reason is None
            else termination_reason
        ),
        has_incumbent=bool(plan.assignments),
        evaluated_work_units=plan.iteration_count,
        total_work_units=(
            int(total_combinations)
            if isinstance(total_combinations, int)
            else None
        ),
        objective_key=plan.objective_key,
        objective_components={
            item.objective_id: item.raw_value
            for item in plan.objective_evaluations
        },
        selected_targets={
            item.task_id: item.target_node_id
            for item in plan.assignments
        },
        details={
            "ready_task_count": len(problem.epoch.ready_tasks),
            **(
                {"reported_solve_status": reported_status.value}
                if phase is SolveTracePhase.REJECTED
                else {}
            ),
            "formulation_exhausted": bool(
                plan.diagnostics.get("formulation_exhausted", False)
            ),
            "communication_time_ms": sum(
                item.communication_ms for item in plan.assignments
            ),
        },
    )


def _resolve_fallback(
    optimizer: str | Optimizer,
    registry: OptimizerRegistry,
) -> Optimizer:
    if not isinstance(optimizer, str):
        return registry.resolve(optimizer)
    if optimizer == "heuristic" or optimizer in algorithm_aliases():
        return built_in_registry().resolve("heuristic")
    try:
        return registry.resolve(optimizer)
    except KeyError:
        return built_in_registry().resolve(optimizer)


def _resolve_formulation(
    optimizer: Optimizer,
    formulation: str | SchedulingFormulation | None,
    registry: FormulationRegistry,
) -> SchedulingFormulation | None:
    selected = formulation
    if selected is None:
        selected = getattr(optimizer, "default_formulation_id", None)
    if selected is None:
        selected = getattr(optimizer, "default_formulation", None)
    if selected is None:
        return None
    resolved = registry.resolve(selected)
    if not isinstance(optimizer, FormulatedOptimizer):
        raise FormulationCompatibilityError(
            f"optimizer {optimizer.optimizer_id!r} does not implement the "
            "formulated optimizer contract"
        )
    supported = optimizer.supported_formulation_ids
    if not isinstance(supported, frozenset) or any(
        not isinstance(item, str) or not item.strip()
        for item in supported
    ):
        raise FormulationCompatibilityError(
            f"optimizer {optimizer.optimizer_id!r} has an invalid "
            "supported_formulation_ids contract"
        )
    supports = optimizer.supports_formulation(resolved.spec)
    if not isinstance(supports, bool):
        raise FormulationCompatibilityError(
            f"optimizer {optimizer.optimizer_id!r} supports_formulation "
            "must return bool"
        )
    if not supports:
        raise FormulationCompatibilityError(
            f"optimizer {optimizer.optimizer_id!r} does not support "
            f"formulation contract {resolved.spec.formulation_id!r} "
            f"version={resolved.spec.formulation_version!r} "
            f"digest={resolved.spec.formulation_digest!r}; supported ids="
            f"{sorted(supported)}"
        )
    return resolved


def _formulation_identity(
    formulation: SchedulingFormulation | None,
) -> str:
    return (
        formulation.spec.formulation_digest
        if formulation is not None
        else ""
    )


def _resolve_policy(
    policy: str | SchedulingPolicy | None,
) -> SchedulingPolicy:
    if policy is None:
        return built_in_policy("greedy_cost")
    if isinstance(policy, str):
        return built_in_policy(policy)
    if not isinstance(policy, SchedulingPolicy):
        raise TypeError(
            "policy must be a built-in id or SchedulingPolicy"
        )
    return policy


def _normalize_input_artifact_bindings(
    epoch: SchedulingEpoch,
    *,
    input_artifact_bindings: Mapping[
        str, Iterable[InputArtifactBinding]
    ] | None,
    parent_artifacts: Mapping[str, Iterable[ArtifactRef]] | None,
) -> dict[str, tuple[InputArtifactBinding, ...]]:
    """Normalize flat compatibility inputs into port-specific bindings."""

    if (
        input_artifact_bindings is not None
        and parent_artifacts is not None
    ):
        raise ValueError(
            "provide input_artifact_bindings or parent_artifacts, not both"
        )
    task_by_id = {
        task.task_id: task for task in epoch.ready_tasks
    }
    provided_ids = set(
        (
            input_artifact_bindings
            if input_artifact_bindings is not None
            else parent_artifacts or {}
        )
    )
    unknown_ids = provided_ids - set(task_by_id)
    if unknown_ids:
        raise ValueError(
            "input artifacts reference tasks outside the epoch: "
            f"{sorted(unknown_ids)}"
        )

    normalized: dict[str, tuple[InputArtifactBinding, ...]] = {}
    for task_id, task in task_by_id.items():
        if input_artifact_bindings is not None:
            bindings = list(
                input_artifact_bindings.get(task_id, ())
            )
        else:
            legacy_artifacts = tuple(
                (parent_artifacts or {}).get(task_id, ())
            )
            bindings = [
                InputArtifactBinding(
                    consumer_task_id=task_id,
                    consumer_port=f"__legacy_input_{index}",
                    artifact=artifact,
                )
                for index, artifact in enumerate(legacy_artifacts)
            ]

        bound_ports = {
            binding.consumer_port for binding in bindings
        }
        unbound_ports = tuple(
            port.name
            for port in task.spec.input_ports
            if port.name not in bound_ports
        )
        if input_artifact_bindings is not None:
            needs_external = bool(unbound_ports) or (
                not task.spec.input_ports
                and not bindings
                and not task.dependency_task_ids
            )
        else:
            needs_external = (
                not bindings and not task.dependency_task_ids
            )
        if needs_external:
            if (
                input_artifact_bindings is not None
                and task.spec.input_ports
            ):
                external_size_mb = (
                    task.spec.input_size_mb
                    * len(unbound_ports)
                    / len(task.spec.input_ports)
                )
            else:
                external_size_mb = task.spec.input_size_mb
            external = ArtifactRef(
                artifact_id=(
                    f"input:{task.workflow_id}:{task.task_id}"
                ),
                producer_task_id="",
                node_id=task.source_node_id,
                size_mb=external_size_mb,
                uri=(
                    f"source://{task.source_node_id}/"
                    f"{task.workflow_id}/{task.task_id}"
                ),
                producer_port="external_input",
                message_type="external_input_batch",
            )
            external_ports = (
                unbound_ports
                if input_artifact_bindings is not None
                else ("__external_input__",)
            ) or ("__external_input__",)
            bindings.extend(
                InputArtifactBinding(
                    consumer_task_id=task_id,
                    consumer_port=consumer_port,
                    artifact=external,
                )
                for consumer_port in external_ports
            )
        normalized[task_id] = tuple(bindings)
    return normalized


def _resolve_selection(
    optimizer: str | Optimizer,
    policy: str | SchedulingPolicy | None,
    registry: OptimizerRegistry,
) -> tuple[Optimizer, SchedulingPolicy]:
    aliases = algorithm_aliases()
    if isinstance(optimizer, str) and optimizer in aliases:
        alias_policy = built_in_policy(
            aliases[optimizer]["policy_id"]
        )
        if policy is not None and _resolve_policy(policy) != alias_policy:
            raise ValueError(
                f"algorithm alias {optimizer!r} already selects policy "
                f"{alias_policy.policy_id!r}; use optimizer='heuristic' "
                "for an explicit policy"
            )
        return registry.resolve("heuristic"), alias_policy
    selected = registry.resolve(optimizer)
    if policy is not None:
        return selected, _resolve_policy(policy)
    default_policy = getattr(selected, "default_policy", None)
    if default_policy is None:
        return selected, _resolve_policy(None)
    if not isinstance(default_policy, SchedulingPolicy):
        raise TypeError(
            f"optimizer {selected.optimizer_id!r} default_policy must be a "
            "SchedulingPolicy"
        )
    return selected, default_policy


def _contract_digest(value: object) -> str:
    payload = json.dumps(
        _canonical_contract_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _canonical_contract_value(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_contract_value(
                getattr(value, item.name)
            )
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_contract_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [
            _canonical_contract_value(item)
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        canonical_items = [
            _canonical_contract_value(item)
            for item in value
        ]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "Infinity"
            if value > 0
            else "-Infinity"
            if value < 0
            else "NaN"
        )
    return value
