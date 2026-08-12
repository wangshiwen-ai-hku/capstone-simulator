from __future__ import annotations

from dataclasses import replace

import pytest

from mars.domain.artifact import ArtifactRef, InputArtifactBinding
from mars.domain.execution import ExecutionMode
from mars.domain.task import (
    PlacementConstraints,
    TaskClass,
    TaskInstance,
    TaskSpec,
)
from mars.domain.topology import (
    LinkSnapshot,
    LinkSpec,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
)
from mars.domain.transfer import TransferEstimate
from mars.optimizers import (
    BinaryOffloadOptimizer,
    CandidateEstimate,
    HeuristicOptimizer,
    ObjectiveMetric,
    OptimizerRegistry,
    OptimizerSolveState,
    PlanValidationError,
    PlannedResourceReservation,
    ResourceDemand,
    SchedulingEpoch,
    SchedulingProblem,
    SchedulingSnapshot,
    SolveLimits,
    SolveStatus,
    SolveTracePhase,
    binary_offload_policy,
    maximum_resource_utilization,
    metric_contract_id,
    node_resource_utilization,
    validate_plan,
)
from mars.profiling import (
    ExecutionProfile,
    ProfileCatalog,
    profile_catalog_from_workloads,
)
from mars.scheduler import build_scheduling_problem, plan_scheduling_epoch
from mars.synthetic_workloads import (
    ExecutionTarget,
    load_default_synthetic_workloads,
)


def _task(
    task_id: str,
    source_node_id: str,
    *,
    priority: int,
    cpu: float,
    gpu: float,
) -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        workflow_id="stage-2-workflow",
        name=task_id,
        source_node_id=source_node_id,
        spec=TaskSpec(
            task_type=task_id,
            task_class=TaskClass.REALTIME_OFFLOADABLE,
            compute_demand=cpu,
            gpu_demand=gpu,
            placement_constraints=PlacementConstraints(
                allowed_node_kinds=(NodeKind.EDGE,),
                allow_source_node=True,
                allow_other_robots=False,
                allow_fallback=True,
            ),
        ),
        priority=priority,
    )


def _candidate(
    task: TaskInstance,
    node: NodeSpec,
    *,
    compute_ms: float,
    communication_ms: float,
    energy_j: float,
    success_probability: float,
    demand: ResourceDemand,
    transfer: TransferEstimate | None = None,
) -> CandidateEstimate:
    return CandidateEstimate(
        task_id=task.task_id,
        node_id=node.node_id,
        node_kind=node.kind,
        source_node_id=task.source_node_id,
        feasible=True,
        ready_time_ms=0.0,
        start_ms=0.0,
        finish_ms=compute_ms + communication_ms,
        compute_ms=compute_ms,
        communication_ms=communication_ms,
        energy_j=energy_j,
        resource_demand=demand,
        input_locations=(task.source_node_id,),
        transfers=(transfer,) if transfer is not None else (),
        success_probability=success_probability,
    )


def _problem() -> SchedulingProblem:
    jetson_a = NodeSpec(
        "jetson_a",
        NodeKind.ROBOT,
        cpu_capacity=4.0,
        gpu_capacity=1.0,
        memory_gb=8.0,
        bandwidth_mbps=100.0,
        base_latency_ms=0.0,
        max_concurrency=2,
    )
    jetson_b = NodeSpec(
        "jetson_b",
        NodeKind.ROBOT,
        cpu_capacity=4.0,
        gpu_capacity=1.0,
        memory_gb=8.0,
        bandwidth_mbps=100.0,
        base_latency_ms=0.0,
        max_concurrency=2,
    )
    edge = NodeSpec(
        "edge",
        NodeKind.EDGE,
        cpu_capacity=4.0,
        gpu_capacity=1.0,
        memory_gb=16.0,
        bandwidth_mbps=1000.0,
        base_latency_ms=0.0,
        max_concurrency=2,
    )
    nodes = (jetson_a, jetson_b, edge)
    snapshots = (
        NodeSnapshot(
            "jetson_a",
            cpu_util=0.25,
            gpu_util=0.10,
            memory_util=0.25,
            remaining_energy_j=10.0,
        ),
        NodeSnapshot(
            "jetson_b",
            cpu_util=0.50,
            gpu_util=0.20,
            memory_util=0.375,
            remaining_energy_j=6.0,
        ),
        NodeSnapshot(
            "edge",
            cpu_util=0.25,
            gpu_util=0.10,
            memory_util=0.25,
        ),
    )
    links = (
        LinkSpec("jetson_a-edge", "jetson_a", "edge", 100.0),
        LinkSpec("jetson_b-edge", "jetson_b", "edge", 100.0),
    )
    link_snapshots = (
        LinkSnapshot("jetson_a-edge", 100.0),
        LinkSnapshot("jetson_b-edge", 100.0),
    )
    task_1 = _task(
        "task_1",
        "jetson_a",
        priority=5,
        cpu=1.0,
        gpu=0.6,
    )
    task_2 = _task(
        "task_2",
        "jetson_b",
        priority=3,
        cpu=2.0,
        gpu=0.5,
    )
    artifact_1 = ArtifactRef(
        "input-1",
        "external",
        "jetson_a",
        size_mb=0.5,
        producer_port="input",
    )
    artifact_2 = ArtifactRef(
        "input-2",
        "external",
        "jetson_b",
        size_mb=0.25,
        producer_port="input",
    )
    bindings = {
        "task_1": (
            InputArtifactBinding("task_1", "input", artifact_1),
        ),
        "task_2": (
            InputArtifactBinding("task_2", "input", artifact_2),
        ),
    }
    demand_1 = ResourceDemand(1.0, 0.6, 1.0)
    demand_2 = ResourceDemand(2.0, 0.5, 2.0)
    transfer_1 = TransferEstimate(
        "transfer-1",
        "jetson_a",
        "edge",
        0.5,
        ("jetson_a-edge",),
        100.0,
        40.0,
    )
    transfer_2 = TransferEstimate(
        "transfer-2",
        "jetson_b",
        "edge",
        0.25,
        ("jetson_b-edge",),
        100.0,
        20.0,
    )
    candidates = {
        "task_1": (
            _candidate(
                task_1,
                jetson_a,
                compute_ms=100.0,
                communication_ms=0.0,
                energy_j=3.0,
                success_probability=0.90,
                demand=demand_1,
            ),
            _candidate(
                task_1,
                edge,
                compute_ms=50.0,
                communication_ms=40.0,
                energy_j=4.0,
                success_probability=0.98,
                demand=demand_1,
                transfer=transfer_1,
            ),
        ),
        "task_2": (
            _candidate(
                task_2,
                jetson_b,
                compute_ms=100.0,
                communication_ms=0.0,
                energy_j=4.0,
                success_probability=0.95,
                demand=demand_2,
            ),
            _candidate(
                task_2,
                edge,
                compute_ms=80.0,
                communication_ms=20.0,
                energy_j=5.0,
                success_probability=0.97,
                demand=demand_2,
                transfer=transfer_2,
            ),
        ),
    }
    epoch = SchedulingEpoch("stage-2-epoch", 0.0, (task_1, task_2))
    snapshot = SchedulingSnapshot(
        schema_version="mars.scheduling-snapshot.v1",
        snapshot_id="stage-2-snapshot",
        captured_at_ms=0.0,
        epoch=epoch,
        node_specs=nodes,
        node_snapshots=snapshots,
        link_specs=links,
        link_snapshots=link_snapshots,
        candidates=candidates,
        input_artifact_bindings=bindings,
        node_available_ms={node.node_id: 0.0 for node in nodes},
        link_available_ms={link.link_id: 0.0 for link in links},
    )
    policy = binary_offload_policy()
    return SchedulingProblem(
        problem_id="stage-2-problem",
        snapshot=snapshot,
        policy=policy,
        solve_limits=SolveLimits(solve_budget_ms=1000.0),
        metric_contract_id=metric_contract_id(policy),
    )


def _with_policy(
    problem: SchedulingProblem,
    policy,
    **changes,
) -> SchedulingProblem:
    return replace(
        problem,
        policy=policy,
        metric_contract_id=metric_contract_id(policy),
        **changes,
    )


def test_binary_offload_matches_stage_1_and_validates() -> None:
    optimizer = BinaryOffloadOptimizer()
    assert not hasattr(optimizer, "solve_history")
    problem = _with_policy(_problem(), optimizer.default_policy)
    state = OptimizerSolveState(session_id="stage-1-validation")

    plan = validate_plan(problem, optimizer.solve_with_state(problem, state))
    summary = state.invocation_summaries()[0]

    assert plan.assignment_by_task["task_1"].target_node_id == "edge"
    assert plan.assignment_by_task["task_2"].target_node_id == "edge"
    assert plan.policy_id == "binary_offload"
    assert plan.objective_value == pytest.approx(0.55375)
    assert plan.solve_status is SolveStatus.OPTIMAL
    assert plan.iteration_count == 4
    assert plan.diagnostics["total_combinations"] == 4
    assert not {
        "alpha",
        "beta",
        "gamma",
        "binary_objective_value",
        "enumerated_combinations",
        "placement_search_exhaustive",
        "solve_budget_ms",
        "max_iterations",
    } & set(plan.diagnostics)
    assert summary["enumerated_combinations"] == plan.iteration_count
    assert summary["total_combinations"] == 4
    assert summary["objective_components"][
        "expected_weighted_success_ratio"
    ] == pytest.approx(0.97625)
    assert summary["communication_time_ms"] == pytest.approx(60.0)
    assert summary["objective_components"][
        "normalized_communication_ratio"
    ] == pytest.approx(0.03)
    assert summary["objective_components"][
        "maximum_resource_utilization"
    ] == pytest.approx(0.75)
    assert summary["objective_value"] == pytest.approx(plan.objective_value)
    assert summary["solve_status"] == "optimal"
    assert state.entries[0].phase is SolveTracePhase.STARTED
    assert state.entries[-1].phase is SolveTracePhase.COMPLETED
    assert any(
        entry.phase is SolveTracePhase.INCUMBENT
        for entry in state.entries
    )


def test_binary_offload_can_be_registered() -> None:
    registry = OptimizerRegistry()
    registry.register(BinaryOffloadOptimizer())

    resolved = registry.resolve("binary_offload")

    assert resolved.optimizer_id == "binary_offload"


def test_binary_offload_is_builtin_and_selects_its_configured_policy() -> None:
    problem = _problem()
    task = problem.epoch.ready_tasks[0]
    optimizer = BinaryOffloadOptimizer(alpha=2.0, beta=0.5, gamma=1.5)
    registry = OptimizerRegistry()
    registry.register(optimizer)

    plan = plan_scheduling_epoch(
        SchedulingEpoch("configured-binary", 0.0, (task,)),
        optimizer="binary_offload",
        registry=registry,
        node_specs=problem.node_by_id,
        node_snapshots=problem.snapshot_by_id,
        parent_artifacts={task.task_id: ()},
        ready_time_ms={task.task_id: 0.0},
        link_specs=problem.link_specs,
        link_snapshots=problem.link_snapshots,
    )

    assert plan.optimizer_id == "binary_offload"
    assert plan.policy_id == optimizer.default_policy.policy_id
    weights = {
        item.metric: item.weight
        for item in optimizer.default_policy.objectives
    }
    assert weights == {
        ObjectiveMetric.EXPECTED_WEIGHTED_SUCCESS_RATIO: 2.0,
        ObjectiveMetric.NORMALIZED_COMMUNICATION_RATIO: 0.5,
        ObjectiveMetric.MAXIMUM_RESOURCE_UTILIZATION: 1.5,
    }


def test_binary_offload_accepts_edge_only_without_source_candidate() -> None:
    original = _problem()
    task = original.epoch.ready_tasks[0]
    edge = next(
        item
        for item in original.candidates[task.task_id]
        if item.node_kind is NodeKind.EDGE
    )
    snapshot = replace(
        original.snapshot,
        epoch=SchedulingEpoch("edge-only", 0.0, (task,)),
        candidates={task.task_id: (edge,)},
        input_artifact_bindings={
            task.task_id: original.input_artifact_bindings[task.task_id]
        },
    )
    optimizer = BinaryOffloadOptimizer(alpha=1.0, beta=0.0, gamma=0.0)
    problem = _with_policy(
        original,
        optimizer.default_policy,
        snapshot=snapshot,
    )

    plan = validate_plan(problem, optimizer.solve(problem))

    assert plan.assignments[0].target_node_id == "edge"
    assert plan.assignments[0].execution_mode is ExecutionMode.EDGE
    assert plan.diagnostics["total_combinations"] == 1


def test_binary_offload_enumerates_peer_edge_cloud_and_source() -> None:
    original = _problem()
    base_task = original.epoch.ready_tasks[0]
    task = replace(
        base_task,
        spec=replace(
            base_task.spec,
            placement_constraints=PlacementConstraints(
                allowed_node_kinds=(
                    NodeKind.ROBOT,
                    NodeKind.EDGE,
                    NodeKind.CLOUD,
                ),
                allow_source_node=True,
                allow_other_robots=True,
            ),
        ),
    )
    peer = NodeSpec(
        "peer",
        NodeKind.ROBOT,
        cpu_capacity=4.0,
        gpu_capacity=1.0,
        memory_gb=8.0,
        bandwidth_mbps=100.0,
        base_latency_ms=0.0,
    )
    cloud = NodeSpec(
        "cloud",
        NodeKind.CLOUD,
        cpu_capacity=16.0,
        gpu_capacity=8.0,
        memory_gb=64.0,
        bandwidth_mbps=1000.0,
        base_latency_ms=0.0,
    )
    original_candidates = original.candidates[task.task_id]
    peer_candidate = _candidate(
        task,
        peer,
        compute_ms=30.0,
        communication_ms=0.0,
        energy_j=1.0,
        success_probability=0.99,
        demand=ResourceDemand(0.5, 0.1, 0.5),
    )
    cloud_candidate = _candidate(
        task,
        cloud,
        compute_ms=20.0,
        communication_ms=0.0,
        energy_j=1.0,
        success_probability=1.0,
        demand=ResourceDemand(0.5, 0.1, 0.5),
    )
    snapshot = replace(
        original.snapshot,
        epoch=SchedulingEpoch("all-targets", 0.0, (task,)),
        node_specs=(*original.node_specs, peer, cloud),
        node_snapshots=(
            *original.node_snapshots,
            NodeSnapshot("peer"),
            NodeSnapshot("cloud"),
        ),
        candidates={
            task.task_id: (
                *original_candidates,
                peer_candidate,
                cloud_candidate,
            )
        },
        input_artifact_bindings={
            task.task_id: original.input_artifact_bindings[task.task_id]
        },
        node_available_ms={
            **original.node_available_ms,
            "peer": 0.0,
            "cloud": 0.0,
        },
    )
    optimizer = BinaryOffloadOptimizer(alpha=1.0, beta=0.0, gamma=0.0)
    problem = _with_policy(
        original,
        optimizer.default_policy,
        snapshot=snapshot,
    )

    plan = validate_plan(problem, optimizer.solve(problem))

    assert plan.assignments[0].target_node_id == "cloud"
    assert plan.assignments[0].execution_mode is ExecutionMode.CLOUD
    assert plan.diagnostics["total_combinations"] == 4


def test_source_selected_against_preference_is_fallback_local() -> None:
    original = _problem()
    base_task = original.epoch.ready_tasks[0]
    task = replace(
        base_task,
        spec=replace(
            base_task.spec,
            placement_constraints=PlacementConstraints(
                allowed_node_kinds=(NodeKind.EDGE,),
                preferred_node_kinds=(NodeKind.EDGE,),
                allow_source_node=True,
                allow_fallback=True,
            ),
        ),
    )
    candidates = tuple(
        replace(
            item,
            success_probability=1.0 if item.is_source else 0.0,
        )
        for item in original.candidates[task.task_id]
    )
    snapshot = replace(
        original.snapshot,
        epoch=SchedulingEpoch("fallback-local", 0.0, (task,)),
        candidates={task.task_id: candidates},
        input_artifact_bindings={
            task.task_id: original.input_artifact_bindings[task.task_id]
        },
    )
    optimizer = BinaryOffloadOptimizer(alpha=1.0, beta=0.0, gamma=0.0)
    problem = _with_policy(
        original,
        optimizer.default_policy,
        snapshot=snapshot,
    )

    plan = validate_plan(problem, optimizer.solve(problem))

    assert plan.assignments[0].target_node_id == task.source_node_id
    assert plan.assignments[0].execution_mode is ExecutionMode.FALLBACK_LOCAL


def test_binary_offload_returns_truthful_iteration_limited_incumbent() -> None:
    optimizer = BinaryOffloadOptimizer()
    problem = _with_policy(
        _problem(),
        optimizer.default_policy,
        solve_limits=SolveLimits(
            solve_budget_ms=1000.0,
            max_iterations=1,
        ),
    )
    state = OptimizerSolveState(session_id="iteration-limited")

    plan = validate_plan(problem, optimizer.solve_with_state(problem, state))
    summary = state.invocation_summaries()[0]

    assert plan.solve_status is SolveStatus.ITERATION_LIMIT
    assert plan.iteration_count == 1
    assert plan.diagnostics["total_combinations"] == 4
    assert "incumbent" in plan.termination_reason
    assert summary["solve_status"] == "iteration_limit"
    assert summary["has_incumbent"] is True
    assert summary["enumerated_combinations"] == 1
    assert summary["total_combinations"] == 4
    assert state.entries[-1].phase is SolveTracePhase.COMPLETED


def test_binary_offload_raises_when_time_limit_precedes_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = BinaryOffloadOptimizer()
    problem = _with_policy(
        _problem(),
        optimizer.default_policy,
        solve_limits=SolveLimits(solve_budget_ms=1.0),
    )
    ticks = iter((0.0, 0.1, 0.1))
    monkeypatch.setattr(
        "mars.optimizers.binary_offload.perf_counter",
        lambda: next(ticks),
    )
    state = OptimizerSolveState(session_id="timeout-no-incumbent")

    with pytest.raises(TimeoutError, match="before an incumbent"):
        optimizer.solve_with_state(problem, state)
    summary = state.invocation_summaries()[0]
    assert summary["solve_status"] == "time_limit"
    assert summary["has_incumbent"] is False
    assert summary["total_combinations"] == 4
    assert state.entries[-1].phase is SolveTracePhase.FAILED


def test_binary_offload_returns_time_limited_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = BinaryOffloadOptimizer()
    problem = _with_policy(
        _problem(),
        optimizer.default_policy,
        solve_limits=SolveLimits(solve_budget_ms=1.0),
    )
    ticks = iter((0.0, 0.0, 0.0, 0.0, 0.1, 0.1))
    monkeypatch.setattr(
        "mars.optimizers.binary_offload.perf_counter",
        lambda: next(ticks),
    )
    state = OptimizerSolveState(session_id="timeout-with-incumbent")

    plan = validate_plan(problem, optimizer.solve_with_state(problem, state))
    summary = state.invocation_summaries()[0]

    assert plan.solve_status is SolveStatus.TIME_LIMIT
    assert plan.iteration_count == 1
    assert plan.diagnostics["total_combinations"] == 4
    assert summary["solve_status"] == "time_limit"
    assert summary["has_incumbent"] is True
    assert summary["enumerated_combinations"] == 1
    assert summary["total_combinations"] == 4
    assert state.entries[-1].phase is SolveTracePhase.COMPLETED


def test_binary_offload_audits_infeasible_solve_without_candidates() -> None:
    original = _problem()
    task = original.epoch.ready_tasks[0]
    snapshot = replace(
        original.snapshot,
        epoch=SchedulingEpoch("no-candidates", 0.0, (task,)),
        candidates={task.task_id: ()},
        input_artifact_bindings={
            task.task_id: original.input_artifact_bindings[task.task_id]
        },
    )
    optimizer = BinaryOffloadOptimizer()
    problem = _with_policy(
        original,
        optimizer.default_policy,
        snapshot=snapshot,
    )
    state = OptimizerSolveState(session_id="infeasible-no-candidates")

    with pytest.raises(ValueError, match="no feasible placement"):
        optimizer.solve_with_state(problem, state)

    summary = state.invocation_summaries()[0]
    assert summary["solve_status"] == "infeasible"
    assert summary["has_incumbent"] is False
    assert state.entries[0].phase is SolveTracePhase.STARTED
    assert state.entries[-1].phase is SolveTracePhase.FAILED


def test_energy_budget_applies_to_non_source_target_and_shared_validator() -> None:
    original = _problem()
    task = original.epoch.ready_tasks[0]
    edge = next(
        item
        for item in original.candidates[task.task_id]
        if item.node_kind is NodeKind.EDGE
    )
    snapshots = tuple(
        replace(
            item,
            remaining_energy_j=(
                1.0 if item.node_id == edge.node_id else item.remaining_energy_j
            ),
        )
        for item in original.node_snapshots
    )
    snapshot = replace(
        original.snapshot,
        epoch=SchedulingEpoch("edge-energy", 0.0, (task,)),
        node_snapshots=snapshots,
        candidates={task.task_id: (edge,)},
        input_artifact_bindings={
            task.task_id: original.input_artifact_bindings[task.task_id]
        },
    )
    optimizer = BinaryOffloadOptimizer()
    problem = _with_policy(
        original,
        optimizer.default_policy,
        snapshot=snapshot,
    )

    with pytest.raises(ValueError, match="no feasible complete"):
        optimizer.solve(problem)

    heuristic_plan = HeuristicOptimizer().solve(problem)
    with pytest.raises(PlanValidationError, match="remaining energy"):
        validate_plan(problem, heuristic_plan)


def test_shared_umax_includes_memory_and_does_not_double_count_carry_in() -> None:
    original = _problem()
    snapshots = tuple(
        replace(
            item,
            cpu_util=0.5 if item.node_id == "jetson_a" else 0.0,
            gpu_util=0.0,
            memory_util=0.9 if item.node_id == "jetson_a" else 0.0,
        )
        for item in original.node_snapshots
    )
    carry_in = PlannedResourceReservation(
        reservation_id="carry-in",
        epoch_id="previous",
        task_id="previous-task",
        node_id="jetson_a",
        start_ms=0.0,
        finish_ms=100.0,
        demand=ResourceDemand(1.0, 0.0, 0.0),
    )
    problem = replace(
        original,
        snapshot=replace(
            original.snapshot,
            node_snapshots=snapshots,
            existing_node_reservations=(carry_in,),
        ),
    )

    utilization = node_resource_utilization(
        problem,
        "jetson_a",
        (),
        problem.epoch.now_ms,
    )
    assert utilization.cpu == pytest.approx(0.5)
    assert utilization.memory == pytest.approx(0.9)
    assert maximum_resource_utilization(problem) == pytest.approx(0.9)


def test_profile_target_facts_are_carried_into_candidate_and_assignment() -> None:
    node = NodeSpec(
        "profile-node",
        NodeKind.ROBOT,
        cpu_capacity=4.0,
        gpu_capacity=2.0,
        memory_gb=8.0,
        bandwidth_mbps=100.0,
        base_latency_ms=0.0,
    )
    task = _task(
        "profile-task",
        node.node_id,
        priority=1,
        cpu=1.0,
        gpu=0.1,
    )
    profile = ExecutionProfile(
        task_type=task.spec.task_type,
        task_class=task.spec.task_class,
        node_kind=NodeKind.ROBOT,
        model_variant="measured",
        input_shape="1x1",
        precision="fp32",
        batch_size=1,
        p50_ms=10.0,
        p95_ms=12.0,
        p99_ms=15.0,
        throughput_per_s=50.0,
        peak_memory_mb=2048.0,
        energy_j=2.0,
        output_size_mb=0.75,
        failure_rate=0.2,
        cpu_units=1.5,
        gpu_units=0.25,
    )
    problem = build_scheduling_problem(
        SchedulingEpoch("profile-epoch", 0.0, (task,)),
        node_specs={node.node_id: node},
        node_snapshots={node.node_id: NodeSnapshot(node.node_id)},
        parent_artifacts={},
        ready_time_ms={task.task_id: 0.0},
        link_specs=(),
        link_snapshots=(),
        profiles=ProfileCatalog([profile]),
        policy=binary_offload_policy(),
    )
    candidate = problem.candidates[task.task_id][0]

    plan = validate_plan(problem, BinaryOffloadOptimizer().solve(problem))

    assert candidate.resource_demand == ResourceDemand(1.5, 0.25, 2.0)
    assert candidate.output_size_mb == pytest.approx(0.75)
    assert candidate.success_probability == pytest.approx(0.8)
    assert plan.assignments[0].output_size_mb == pytest.approx(0.75)
    assert plan.assignments[0].success_probability == pytest.approx(0.8)


def test_public_workload_profile_conversion_preserves_target_resources() -> None:
    workloads = load_default_synthetic_workloads()
    profiles = profile_catalog_from_workloads(workloads)
    workload = workloads.get("object_detection")
    robot = workload.profile_for(ExecutionTarget.ORIN)
    converted = profiles.lookup("object_detection", NodeKind.ROBOT)

    assert converted is not None
    assert converted.cpu_units == robot.resources.cpu_cores
    assert converted.gpu_units == robot.resources.gpu_units
    assert converted.peak_memory_mb == robot.resources.memory_mb
    assert converted.output_size_mb == robot.output_size_mb.typical
    assert converted.failure_rate == robot.failure_rate
