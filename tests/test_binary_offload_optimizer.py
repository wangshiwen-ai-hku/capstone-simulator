from __future__ import annotations

import pytest

from mars.domain.artifact import ArtifactRef, InputArtifactBinding
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
    OptimizerRegistry,
    ResourceDemand,
    SchedulingEpoch,
    SchedulingProblem,
    SchedulingSnapshot,
    SolveLimits,
    SolveStatus,
    built_in_policy,
    validate_plan,
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
    return SchedulingProblem(
        problem_id="stage-2-problem",
        snapshot=snapshot,
        policy=built_in_policy("greedy_cost"),
        solve_limits=SolveLimits(solve_budget_ms=1000.0),
    )


def test_binary_offload_matches_stage_1_and_validates() -> None:
    problem = _problem()
    optimizer = BinaryOffloadOptimizer(alpha=1.0, beta=0.01, gamma=2.0)

    plan = validate_plan(problem, optimizer.solve(problem))

    assert plan.assignment_by_task["task_1"].target_node_id == "jetson_a"
    assert plan.assignment_by_task["task_2"].target_node_id == "edge"
    assert plan.diagnostics["binary_objective_value"] == pytest.approx(-5.71)
    assert plan.solve_status is SolveStatus.OPTIMAL
    assert plan.diagnostics["enumerated_combinations"] == 4
    epoch_record = optimizer.solve_history[0]
    assert epoch_record["success_reward"] == pytest.approx(7.41)
    assert epoch_record["communication_time_ms"] == pytest.approx(20.0)
    assert epoch_record["maximum_resource_utilization"] == pytest.approx(0.75)
    assert epoch_record["objective_value"] == pytest.approx(-5.71)


def test_binary_offload_can_be_registered() -> None:
    registry = OptimizerRegistry()
    registry.register(BinaryOffloadOptimizer())

    resolved = registry.resolve("binary_offload")

    assert resolved.optimizer_id == "binary_offload"
