from __future__ import annotations

from dataclasses import replace

import pytest

from mars.engine import run_workflow_simulation
from mars.models import (
    ArtifactRef,
    ExecutionMode,
    LinkSnapshot,
    LinkSpec,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    PlacementConstraints,
    TaskClass,
    TaskInstance,
    TaskSpec,
    WorkflowSpec,
)
from mars.network import NetworkTopology, synthesize_legacy_full_mesh
from mars.optimizers import (
    HeuristicOptimizer,
    OptimizerRegistry,
    PlanValidationError,
    ResourceDemand,
    SchedulingEpoch,
    SchedulingPlan,
    validate_plan,
)
from mars.scheduler import (
    allowed_nodes,
    build_scheduling_problem,
    plan_scheduling_epoch,
)


def _nodes() -> dict[str, NodeSpec]:
    return {
        "robot": NodeSpec(
            "robot",
            NodeKind.ROBOT,
            4,
            2,
            16,
            100,
            2,
            capabilities=("camera",),
        ),
        "edge": NodeSpec(
            "edge",
            NodeKind.EDGE,
            12,
            8,
            64,
            500,
            5,
            capabilities=("vision",),
        ),
    }


def _snapshots(
    nodes: dict[str, NodeSpec],
) -> dict[str, NodeSnapshot]:
    return {
        node_id: NodeSnapshot(
            node_id,
            power_w=25 if spec.kind is NodeKind.ROBOT else 120,
        )
        for node_id, spec in nodes.items()
    }


def _task(
    task_id: str,
    *,
    task_class: TaskClass = TaskClass.REALTIME_OFFLOADABLE,
    source: str = "robot",
    placement: PlacementConstraints | None = None,
    input_size_mb: float = 0.0,
    bandwidth_requirement_mbps: float = 0.0,
) -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        workflow_id="wf",
        name=task_id,
        source_node_id=source,
        spec=TaskSpec(
            task_type="test_task",
            task_class=task_class,
            compute_demand=2.0,
            gpu_demand=1.0,
            input_size_mb=input_size_mb,
            bandwidth_requirement_mbps=bandwidth_requirement_mbps,
            placement_constraints=placement,
        ),
        deadline_time_ms=10_000,
        expected_accuracy=1.0,
    )


def _direct_links() -> tuple[tuple[LinkSpec, ...], tuple[LinkSnapshot, ...]]:
    specs = (
        LinkSpec("robot-edge", "robot", "edge", 100, 3),
        LinkSpec("edge-robot", "edge", "robot", 40, 7),
    )
    snapshots = (
        LinkSnapshot("robot-edge", 80, latency_ms=2),
        LinkSnapshot("edge-robot", 30, latency_ms=5),
    )
    return specs, snapshots


def test_explicit_constraints_replace_task_class_placement_logic() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    placement = PlacementConstraints(
        pinned_node_id="edge",
        allowed_node_kinds=(NodeKind.EDGE,),
        preferred_node_kinds=(NodeKind.EDGE,),
        required_capabilities=("vision",),
        allow_source_node=False,
    )
    labels = (
        TaskClass.LOCAL_SAFETY,
        TaskClass.REALTIME_OFFLOADABLE,
        TaskClass.EDGE_HEAVY,
    )

    for label in labels:
        candidates = allowed_nodes(
            _task(label.value, task_class=label, placement=placement),
            nodes.values(),
            snapshots,
        )
        assert [node.node_id for node in candidates] == ["edge"]


def test_required_capability_is_a_hard_filter() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = _task(
        "capability",
        placement=PlacementConstraints(
            allowed_node_kinds=(NodeKind.EDGE,),
            required_capabilities=("tensor_rt",),
            allow_source_node=False,
        ),
    )

    assert allowed_nodes(task, nodes.values(), snapshots) == []


def test_directed_links_are_asymmetric() -> None:
    nodes = _nodes()
    specs = (LinkSpec("up", "robot", "edge", 100, 1),)
    snapshots = (LinkSnapshot("up", 100),)
    topology = NetworkTopology(nodes, specs, snapshots)

    upload = topology.estimate(
        transfer_id="upload",
        source_node_id="robot",
        target_node_id="edge",
        size_mb=10,
    )
    download = topology.estimate(
        transfer_id="download",
        source_node_id="edge",
        target_node_id="robot",
        size_mb=10,
    )

    assert upload.feasible
    assert upload.path_link_ids == ("up",)
    assert not download.feasible
    assert download.reason == "no_online_link_path"


def test_offline_and_bandwidth_constrained_links_are_infeasible() -> None:
    nodes = _nodes()
    spec = (LinkSpec("up", "robot", "edge", 100),)
    offline = NetworkTopology(
        nodes,
        spec,
        (LinkSnapshot("up", 100, online=False),),
    )
    limited = NetworkTopology(
        nodes,
        spec,
        (LinkSnapshot("up", 40),),
    )

    offline_estimate = offline.estimate(
        transfer_id="x",
        source_node_id="robot",
        target_node_id="edge",
        size_mb=1,
        minimum_bandwidth_mbps=50,
    )
    assert not offline_estimate.feasible
    assert offline_estimate.reason == "no_online_link_path"
    constrained = limited.estimate(
        transfer_id="x",
        source_node_id="robot",
        target_node_id="edge",
        size_mb=1,
        minimum_bandwidth_mbps=50,
    )
    assert not constrained.feasible
    assert constrained.reason == "bandwidth_below_requirement"


def test_legacy_link_synthesis_preserves_endpoint_formula() -> None:
    nodes = (
        NodeSpec("a", NodeKind.ROBOT, 4, 1, 8, 100, 2),
        NodeSpec("b", NodeKind.EDGE, 8, 4, 32, 50, 5),
    )
    snapshots = (
        NodeSnapshot("a", network_latency_ms=3),
        NodeSnapshot("b", network_latency_ms=4),
    )
    specs, states = synthesize_legacy_full_mesh(nodes, snapshots)
    topology = NetworkTopology(("a", "b"), specs, states)

    estimate = topology.estimate(
        transfer_id="legacy",
        source_node_id="a",
        target_node_id="b",
        size_mb=10,
    )

    assert estimate.transfer_time_ms == pytest.approx(
        2 + 5 + 3 + 4 + (10 * 8 / 50 * 1000)
    )


def test_explicit_empty_topology_drops_remote_only_task() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = _task(
        "remote",
        input_size_mb=5,
        placement=PlacementConstraints(
            pinned_node_id="edge",
            allowed_node_kinds=(NodeKind.EDGE,),
            allow_source_node=False,
        ),
    )
    epoch = SchedulingEpoch("empty-links", 0, (task,))

    problem = build_scheduling_problem(
        epoch,
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"remote": 0},
        link_specs=(),
        link_snapshots=(),
    )
    plan = plan_scheduling_epoch(
        epoch,
        optimizer="greedy_cost",
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"remote": 0},
        link_specs=(),
        link_snapshots=(),
    )

    assert problem.link_specs == ()
    assert not problem.candidates["remote"][0].feasible
    assert plan.assignments[0].execution_mode is ExecutionMode.DROP


def test_remote_zero_size_input_needs_no_link_path() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = _task(
        "zero-byte",
        placement=PlacementConstraints(
            pinned_node_id="edge",
            allowed_node_kinds=(NodeKind.EDGE,),
        ),
    )

    problem = build_scheduling_problem(
        SchedulingEpoch("zero-byte", 0, (task,)),
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"zero-byte": 0},
        link_specs=(),
        link_snapshots=(),
    )

    candidate = problem.candidates["zero-byte"][0]
    assert candidate.feasible
    assert candidate.transfers[0].path_link_ids == ()
    assert candidate.communication_ms == 0


def test_offline_source_does_not_hide_edge_resident_parent_artifact() -> None:
    nodes = _nodes()
    snapshots = {
        **_snapshots(nodes),
        "robot": NodeSnapshot("robot", online=False),
    }
    task = replace(
        _task(
            "downstream",
            placement=PlacementConstraints(
                pinned_node_id="edge",
                allowed_node_kinds=(NodeKind.EDGE,),
            ),
        ),
        dependency_task_ids=("upstream",),
    )
    artifact = ArtifactRef("parent", "upstream", "edge", 4)

    problem = build_scheduling_problem(
        SchedulingEpoch("offline-source", 0, (task,)),
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={"downstream": (artifact,)},
        ready_time_ms={"downstream": 0},
        link_specs=(),
        link_snapshots=(),
    )

    assert problem.candidates["downstream"][0].feasible


def test_dependency_without_parent_artifact_is_not_treated_as_root_input() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = replace(
        _task(
            "downstream",
            placement=PlacementConstraints(
                pinned_node_id="edge",
                allowed_node_kinds=(NodeKind.EDGE,),
            ),
        ),
        dependency_task_ids=("upstream",),
    )

    problem = build_scheduling_problem(
        SchedulingEpoch("missing-parent", 0, (task,)),
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"downstream": 0},
    )

    candidate = problem.candidates["downstream"][0]
    assert not candidate.feasible
    assert candidate.reason == "dependency_artifact_unavailable"


def test_constraint_capabilities_are_normalized_once() -> None:
    constraints = PlacementConstraints(
        allowed_node_kinds=(NodeKind.EDGE,),
        required_capabilities=(" vision ",),
        allow_source_node=False,
    )
    assert constraints.required_capabilities == ("vision",)


def test_scheduling_problem_contains_complete_node_and_link_inventory() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    links, link_snapshots = _direct_links()
    task = _task("inventory")
    problem = build_scheduling_problem(
        SchedulingEpoch("inventory", 0, (task,)),
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"inventory": 0},
        link_specs=links,
        link_snapshots=link_snapshots,
    )

    assert {node.node_id for node in problem.node_specs} == set(nodes)
    assert {link.link_id for link in problem.link_specs} == {
        "robot-edge",
        "edge-robot",
    }
    assert set(problem.link_available_ms) == {
        "robot-edge",
        "edge-robot",
    }


def test_ready_batch_can_start_on_independent_nodes_concurrently() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    robot_task = _task(
        "a-robot",
        placement=PlacementConstraints(
            pinned_node_id="robot",
            allowed_node_kinds=(NodeKind.ROBOT,),
        ),
    )
    edge_task = _task(
        "b-edge",
        placement=PlacementConstraints(
            pinned_node_id="edge",
            allowed_node_kinds=(NodeKind.EDGE,),
        ),
    )
    epoch = SchedulingEpoch("parallel", 0, (robot_task, edge_task))

    plan = plan_scheduling_epoch(
        epoch,
        optimizer="greedy_cost",
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"a-robot": 0, "b-edge": 0},
        link_specs=(),
        link_snapshots=(),
    )

    assert {item.task_id for item in plan.assignments} == {
        "a-robot",
        "b-edge",
    }
    assert {item.estimated_start_ms for item in plan.assignments} == {0}


@pytest.mark.parametrize(
    ("max_concurrency", "gpu_capacity", "parallel"),
    (
        (2, 4.0, True),
        (2, 1.0, False),
        (1, 4.0, False),
    ),
)
def test_batch_respects_node_concurrency_and_capacity(
    max_concurrency: int,
    gpu_capacity: float,
    parallel: bool,
) -> None:
    node = NodeSpec(
        "edge",
        NodeKind.EDGE,
        12,
        gpu_capacity,
        64,
        500,
        1,
        max_concurrency=max_concurrency,
    )
    nodes = {"edge": node}
    snapshots = {"edge": NodeSnapshot("edge", power_w=120)}
    placement = PlacementConstraints(
        pinned_node_id="edge",
        allowed_node_kinds=(NodeKind.EDGE,),
    )
    task_a = _task("a", source="edge", placement=placement)
    task_b = _task("b", source="edge", placement=placement)

    plan = plan_scheduling_epoch(
        SchedulingEpoch("node-capacity", 0, (task_a, task_b)),
        optimizer="greedy_cost",
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"a": 0, "b": 0},
        link_specs=(),
        link_snapshots=(),
    )
    starts = {
        assignment.task_id: assignment.estimated_start_ms
        for assignment in plan.assignments
    }

    assert (starts["a"] == starts["b"]) is parallel


def test_ready_batch_serializes_transfers_that_share_a_link() -> None:
    nodes = {
        "robot": NodeSpec(
            "robot", NodeKind.ROBOT, 4, 2, 16, 100, 1
        ),
        "hub": NodeSpec(
            "hub", NodeKind.CLOUD, 8, 2, 32, 100, 1
        ),
        "edge-a": NodeSpec(
            "edge-a", NodeKind.EDGE, 8, 4, 32, 100, 1
        ),
        "edge-b": NodeSpec(
            "edge-b", NodeKind.EDGE, 8, 4, 32, 100, 1
        ),
    }
    snapshots = _snapshots(nodes)
    links = (
        LinkSpec("shared", "robot", "hub", 100),
        LinkSpec("to-a", "hub", "edge-a", 100),
        LinkSpec("to-b", "hub", "edge-b", 100),
    )
    link_states = tuple(
        LinkSnapshot(link.link_id, 100) for link in links
    )
    task_a = _task(
        "a",
        input_size_mb=10,
        placement=PlacementConstraints(
            pinned_node_id="edge-a",
            allowed_node_kinds=(NodeKind.EDGE,),
        ),
    )
    task_b = _task(
        "b",
        input_size_mb=10,
        placement=PlacementConstraints(
            pinned_node_id="edge-b",
            allowed_node_kinds=(NodeKind.EDGE,),
        ),
    )

    plan = plan_scheduling_epoch(
        SchedulingEpoch("shared-link", 0, (task_a, task_b)),
        optimizer="greedy_cost",
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"a": 0, "b": 0},
        link_specs=links,
        link_snapshots=link_states,
    )

    shared = sorted(
        (
            reservation.start_ms,
            reservation.finish_ms,
        )
        for reservation in plan.transfer_reservations
        if "shared" in reservation.path_link_ids
    )
    assert len(shared) == 2
    assert shared[1][0] >= shared[0][1]


class _DelegatingOptimizer:
    optimizer_id = "custom"

    def solve(self, problem):
        baseline = HeuristicOptimizer("greedy_cost").solve(problem)
        return replace(
            baseline,
            optimizer_id=self.optimizer_id,
            assignments=tuple(
                replace(
                    assignment,
                    optimizer_id=self.optimizer_id,
                    reason="custom optimizer",
                )
                for assignment in baseline.assignments
            ),
        )


class _InvalidOptimizer:
    optimizer_id = "invalid"

    def solve(self, problem):
        return SchedulingPlan(
            epoch_id=problem.epoch.epoch_id,
            optimizer_id=self.optimizer_id,
            assignments=(),
        )


class _ExplodingOptimizer:
    optimizer_id = "exploding"

    def solve(self, problem):
        raise RuntimeError("solver process crashed")


def test_optimizer_registry_runs_a_custom_optimizer() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = _task("custom")
    registry = OptimizerRegistry()
    registry.register(_DelegatingOptimizer(), aliases=("custom-alias",))

    plan = plan_scheduling_epoch(
        SchedulingEpoch("custom-epoch", 0, (task,)),
        optimizer="custom-alias",
        registry=registry,
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"custom": 0},
    )

    assert plan.optimizer_id == "custom"
    assert plan.assignments[0].optimizer_id == "custom"


def test_invalid_plugin_plan_is_repaired_by_safe_fallback() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = _task("repair")
    registry = OptimizerRegistry()
    registry.register(_InvalidOptimizer())

    plan = plan_scheduling_epoch(
        SchedulingEpoch("repair-epoch", 0, (task,)),
        optimizer="invalid",
        registry=registry,
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"repair": 0},
    )

    assert plan.optimizer_id == "dag_deadline"
    assert plan.diagnostics["repaired_from_optimizer"] == "invalid"
    assert "assign or explicitly defer" in plan.diagnostics["repair_reason"]


def test_plugin_exception_is_repaired_by_safe_fallback() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = _task("repair-exception")
    registry = OptimizerRegistry()
    registry.register(_ExplodingOptimizer())

    plan = plan_scheduling_epoch(
        SchedulingEpoch("repair-exception", 0, (task,)),
        optimizer="exploding",
        registry=registry,
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"repair-exception": 0},
    )

    assert plan.optimizer_id == "dag_deadline"
    assert "solver process crashed" in plan.diagnostics["repair_reason"]


def test_plan_repair_can_be_disabled_for_strict_solver_evaluation() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = _task("strict")
    registry = OptimizerRegistry()
    registry.register(_InvalidOptimizer())

    with pytest.raises(PlanValidationError):
        plan_scheduling_epoch(
            SchedulingEpoch("strict-epoch", 0, (task,)),
            optimizer="invalid",
            registry=registry,
            fallback_optimizer=None,
            node_specs=nodes,
            node_snapshots=snapshots,
            parent_artifacts={},
            ready_time_ms={"strict": 0},
        )


def test_unknown_optimizer_has_an_actionable_error() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = _task("unknown")

    with pytest.raises(KeyError, match="available"):
        plan_scheduling_epoch(
            SchedulingEpoch("unknown-epoch", 0, (task,)),
            optimizer="does-not-exist",
            node_specs=nodes,
            node_snapshots=snapshots,
            parent_artifacts={},
            ready_time_ms={"unknown": 0},
        )


def test_validator_rejects_fabricated_timing_and_resource_demand() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    task = replace(_task("strict-plan"), arrival_time_ms=100)
    problem = build_scheduling_problem(
        SchedulingEpoch("strict-plan", 100, (task,)),
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"strict-plan": 100},
    )
    valid = HeuristicOptimizer("greedy_cost").solve(problem)
    assignment = valid.assignments[0]
    reservation = valid.node_reservations[0]
    early = replace(
        valid,
        assignments=(
            replace(
                assignment,
                estimated_start_ms=0,
            ),
        ),
    )
    zero_demand = replace(
        valid,
        node_reservations=(
            replace(
                reservation,
                demand=ResourceDemand(0, 0, 0),
            ),
        ),
    )

    with pytest.raises(PlanValidationError, match="before it is ready"):
        validate_plan(problem, early)
    with pytest.raises(PlanValidationError, match="does not match candidate"):
        validate_plan(problem, zero_demand)


def test_validator_requires_real_candidate_transfers() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    links, link_snapshots = _direct_links()
    task = _task(
        "remote-transfer",
        input_size_mb=5,
        placement=PlacementConstraints(
            pinned_node_id="edge",
            allowed_node_kinds=(NodeKind.EDGE,),
        ),
    )
    problem = build_scheduling_problem(
        SchedulingEpoch("remote-transfer", 0, (task,)),
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"remote-transfer": 0},
        link_specs=links,
        link_snapshots=link_snapshots,
    )
    valid = HeuristicOptimizer("greedy_cost").solve(problem)
    fabricated = replace(
        valid,
        assignments=(
            replace(
                valid.assignments[0],
                communication_ms=0,
                transfer_link_ids=(),
            ),
        ),
        transfer_reservations=(),
    )

    with pytest.raises(
        PlanValidationError,
        match="transfer reservations do not match",
    ):
        validate_plan(problem, fabricated)


def test_validator_respects_declared_node_and_link_availability() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    links, link_snapshots = _direct_links()
    task = _task(
        "availability",
        input_size_mb=5,
        placement=PlacementConstraints(
            pinned_node_id="edge",
            allowed_node_kinds=(NodeKind.EDGE,),
        ),
    )
    problem = build_scheduling_problem(
        SchedulingEpoch("availability", 0, (task,)),
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"availability": 0},
        node_available_ms={"robot": 0, "edge": 200},
        link_specs=links,
        link_snapshots=link_snapshots,
        link_available_ms={"robot-edge": 100, "edge-robot": 0},
    )
    valid = HeuristicOptimizer("greedy_cost").solve(problem)
    transfer = valid.transfer_reservations[0]
    early_transfer = replace(
        valid,
        assignments=(
            replace(
                valid.assignments[0],
                estimated_start_ms=0,
            ),
        ),
        transfer_reservations=(
            replace(
                transfer,
                start_ms=0,
                finish_ms=(
                    transfer.finish_ms - transfer.start_ms
                ),
            ),
        ),
    )
    resource = valid.node_reservations[0]
    early_resource = replace(
        valid,
        assignments=(
            replace(
                valid.assignments[0],
                estimated_start_ms=0,
                estimated_finish_ms=resource.finish_ms - 200,
            ),
        ),
        node_reservations=(
            replace(
                resource,
                start_ms=0,
                finish_ms=resource.finish_ms - 200,
            ),
        ),
    )

    with pytest.raises(PlanValidationError, match="link is available"):
        validate_plan(problem, early_transfer)
    with pytest.raises(PlanValidationError, match="node is available"):
        validate_plan(problem, early_resource)


def test_event_engine_executes_independent_batch_in_parallel() -> None:
    nodes = _nodes()
    snapshots = _snapshots(nodes)
    robot_task = _task(
        "robot-task",
        placement=PlacementConstraints(
            pinned_node_id="robot",
            allowed_node_kinds=(NodeKind.ROBOT,),
        ),
    )
    edge_task = _task(
        "edge-task",
        placement=PlacementConstraints(
            pinned_node_id="edge",
            allowed_node_kinds=(NodeKind.EDGE,),
        ),
    )
    report = run_workflow_simulation(
        WorkflowSpec("wf", (robot_task, edge_task)),
        list(nodes.values()),
        list(snapshots.values()),
        algorithm="greedy_cost",
        network_jitter=0,
        resource_noise=0,
        link_specs=[],
        link_snapshots=[],
    )
    results = {item.task_id: item for item in report.task_results}

    assert results["robot-task"].start_time_ms == 0
    assert results["edge-task"].start_time_ms == 0
    assert report.metrics["makespan_ms"] == max(
        results["robot-task"].finish_time_ms,
        results["edge-task"].finish_time_ms,
    )


def test_event_engine_keeps_transfer_and_compute_reservations_separate() -> None:
    nodes = {
        "robot": NodeSpec(
            "robot", NodeKind.ROBOT, 4, 2, 16, 100, 1
        ),
        "edge": NodeSpec(
            "edge",
            NodeKind.EDGE,
            12,
            4,
            64,
            100,
            1,
            max_concurrency=2,
        ),
    }
    snapshots = _snapshots(nodes)
    edge_placement = PlacementConstraints(
        pinned_node_id="edge",
        allowed_node_kinds=(NodeKind.EDGE,),
    )
    local_compute = _task(
        "a-local-compute",
        source="edge",
        placement=edge_placement,
    )
    remote_transfer = _task(
        "b-remote-transfer",
        source="robot",
        placement=edge_placement,
        input_size_mb=10,
    )
    links = [LinkSpec("robot-edge", "robot", "edge", 100)]
    link_snapshots = [LinkSnapshot("robot-edge", 100)]

    report = run_workflow_simulation(
        WorkflowSpec("wf", (local_compute, remote_transfer)),
        list(nodes.values()),
        list(snapshots.values()),
        algorithm="greedy_cost",
        network_jitter=0,
        resource_noise=0,
        link_specs=links,
        link_snapshots=link_snapshots,
    )
    results = {item.task_id: item for item in report.task_results}

    assert results["a-local-compute"].start_time_ms == 0
    assert results["b-remote-transfer"].start_time_ms == 0
    assert results["b-remote-transfer"].communication_time_ms > 0


def test_event_engine_uses_same_node_max_concurrency() -> None:
    node = NodeSpec(
        "edge",
        NodeKind.EDGE,
        12,
        4,
        64,
        500,
        1,
        max_concurrency=2,
    )
    placement = PlacementConstraints(
        pinned_node_id="edge",
        allowed_node_kinds=(NodeKind.EDGE,),
    )
    tasks = (
        _task("a", source="edge", placement=placement),
        _task("b", source="edge", placement=placement),
    )

    report = run_workflow_simulation(
        WorkflowSpec("wf", tasks),
        [node],
        [NodeSnapshot("edge", power_w=120)],
        algorithm="greedy_cost",
        network_jitter=0,
        resource_noise=0,
        link_specs=[],
        link_snapshots=[],
    )

    assert {
        item.start_time_ms for item in report.task_results
    } == {0}
