"""Regression tests for runtime scheduling invariants."""

from __future__ import annotations

import asyncio

import pytest

from mars.coordinator import CentralCoordinator
from mars.engine import run_workflow_simulation
from mars.domain import (
    ArtifactRef,
    DataEdge,
    DataPort,
    FailurePolicy,
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
from mars.optimizers import (
    OptimizerRegistry,
    SchedulingEpoch,
)
from mars.runtime import InProcessRuntime
from mars.scheduler import (
    allowed_nodes,
    build_scheduling_problem,
    plan_scheduling_epoch,
)


def _node(
    node_id: str,
    kind: NodeKind,
    *,
    cpu: float = 12,
    gpu: float = 4,
    max_concurrency: int = 1,
) -> NodeSpec:
    return NodeSpec(
        node_id=node_id,
        kind=kind,
        cpu_capacity=cpu,
        gpu_capacity=gpu,
        memory_gb=64,
        bandwidth_mbps=500,
        base_latency_ms=1,
        max_concurrency=max_concurrency,
    )


def _pinned_task(
    task_id: str,
    node_id: str,
    kind: NodeKind,
    *,
    compute: float = 1,
    input_size_mb: float = 0,
    arrival_time_ms: float = 0,
    dependencies: tuple[str, ...] = (),
    expected_accuracy: float = 1,
) -> TaskInstance:
    return TaskInstance(
        task_id=task_id,
        workflow_id="wf",
        name=task_id,
        source_node_id=node_id,
        spec=TaskSpec(
            task_type=f"custom_{task_id}",
            task_class=TaskClass.REALTIME_OFFLOADABLE,
            compute_demand=compute,
            input_size_mb=input_size_mb,
            placement_constraints=PlacementConstraints(
                pinned_node_id=node_id,
                allowed_node_kinds=(kind,),
            ),
        ),
        dependency_task_ids=dependencies,
        arrival_time_ms=arrival_time_ms,
        deadline_time_ms=100_000,
        expected_accuracy=expected_accuracy,
    )


class _DelayedQueuedCompletionRuntime(InProcessRuntime):
    def __init__(self, *args, delayed_task_ids: tuple[str, ...], **kwargs):
        super().__init__(*args, **kwargs)
        self._delayed_task_ids = frozenset(delayed_task_ids)
        self._task_by_dispatch: dict[str, str] = {}

    async def dispatch(self, command):
        acknowledgement = await super().dispatch(command)
        self._task_by_dispatch[acknowledgement.dispatch_id] = (
            command.task.task_id
        )
        return acknowledgement

    async def receive_completion(self, dispatch_id):
        if self._task_by_dispatch[dispatch_id] in self._delayed_task_ids:
            await asyncio.sleep(0.01)
        return await super().receive_completion(dispatch_id)


def test_engine_preserves_free_capacity_across_scheduling_epochs() -> None:
    edge = _node(
        "edge",
        NodeKind.EDGE,
        cpu=20,
        gpu=4,
        max_concurrency=2,
    )
    long_task = _pinned_task(
        "long",
        "edge",
        NodeKind.EDGE,
        compute=10,
    )
    short_task = _pinned_task(
        "short",
        "edge",
        NodeKind.EDGE,
        compute=1,
    )
    child = _pinned_task(
        "child",
        "edge",
        NodeKind.EDGE,
        compute=2,
        dependencies=("short",),
    )

    report = run_workflow_simulation(
        WorkflowSpec("wf", (long_task, short_task, child)),
        [edge],
        [NodeSnapshot("edge")],
        algorithm="greedy_cost",
        network_jitter=0,
        resource_noise=0,
        link_specs=[],
        link_snapshots=[],
    )
    by_id = {item.task_id: item for item in report.task_results}

    assert by_id["child"].start_time_ms == pytest.approx(
        by_id["short"].finish_time_ms,
        abs=0.01,
    )
    assert (
        by_id["child"].start_time_ms
        < by_id["long"].finish_time_ms
    )
    assert all(
        0 <= utilization <= 1
        for utilization in report.node_utilization.values()
    )


def test_execution_noise_cannot_overlap_a_single_runtime_slot() -> None:
    edge = _node(
        "edge",
        NodeKind.EDGE,
        cpu=20,
        gpu=4,
        max_concurrency=1,
    )
    tasks = (
        _pinned_task(
            "a",
            "edge",
            NodeKind.EDGE,
            compute=10,
        ),
        _pinned_task(
            "b",
            "edge",
            NodeKind.EDGE,
            compute=10,
        ),
    )

    report = run_workflow_simulation(
        WorkflowSpec("wf", tasks),
        [edge],
        [NodeSnapshot("edge")],
        algorithm="greedy_cost",
        seed=1,
        network_jitter=0,
        resource_noise=0.5,
        link_specs=[],
        link_snapshots=[],
    )
    ordered = sorted(
        report.task_results,
        key=lambda item: item.start_time_ms,
    )

    assert ordered[1].start_time_ms >= ordered[0].finish_time_ms
    assert report.metrics["makespan_ms"] == max(
        item.finish_time_ms for item in ordered
    )


def test_accuracy_failure_replanning_preserves_edge_queue() -> None:
    edge = _node(
        "edge",
        NodeKind.EDGE,
        cpu=100,
        gpu=4,
        max_concurrency=2,
    )
    tasks = (
        _pinned_task(
            "a-failing",
            "edge",
            NodeKind.EDGE,
            compute=1,
            expected_accuracy=0,
        ),
        _pinned_task(
            "b-queued",
            "edge",
            NodeKind.EDGE,
            compute=20,
        ),
        _pinned_task(
            "c-queued",
            "edge",
            NodeKind.EDGE,
            compute=20,
        ),
        _pinned_task(
            "d-queued",
            "edge",
            NodeKind.EDGE,
            compute=20,
        ),
        _pinned_task(
            "e-queued",
            "edge",
            NodeKind.EDGE,
            compute=20,
        ),
    )
    runtime = _DelayedQueuedCompletionRuntime(
        [edge],
        [NodeSnapshot("edge")],
        delayed_task_ids=(
            "b-queued",
            "c-queued",
            "d-queued",
            "e-queued",
        ),
        execution_noise=0,
        respect_expected_accuracy=True,
    )

    report = CentralCoordinator(
        runtime,
        link_specs=(),
        link_snapshots=(),
    ).run(
        WorkflowSpec("wf", tasks),
        algorithm="greedy_cost",
        seed=0,
        max_attempts=2,
    )

    by_id = {item["task_id"]: item for item in report.task_results}
    assert by_id["a-failing"]["attempt_count"] == 2
    assert by_id["b-queued"]["state"] == "succeeded"
    assert by_id["c-queued"]["state"] == "succeeded"
    assert by_id["d-queued"]["state"] == "succeeded"
    assert by_id["e-queued"]["state"] == "succeeded"


@pytest.mark.parametrize("optimizer_id", ["greedy_cost", "dag_deadline"])
def test_optimizers_honor_declared_node_kind_preference(
    optimizer_id: str,
) -> None:
    nodes = {
        "robot": _node("robot", NodeKind.ROBOT, cpu=30, gpu=8),
        "edge": _node("edge", NodeKind.EDGE, cpu=1, gpu=0),
    }
    snapshots = {
        node_id: NodeSnapshot(node_id)
        for node_id in nodes
    }
    placement = PlacementConstraints(
        allowed_node_kinds=(NodeKind.EDGE,),
        preferred_node_kinds=(NodeKind.EDGE,),
        allow_source_node=True,
        allow_fallback=True,
    )
    task = TaskInstance(
        "preferred",
        "wf",
        "preferred",
        "robot",
        TaskSpec(
            "custom_preferred",
            TaskClass.REALTIME_OFFLOADABLE,
            compute_demand=10,
            placement_constraints=placement,
        ),
        deadline_time_ms=100_000,
    )
    plan = plan_scheduling_epoch(
        SchedulingEpoch("preference", 0, (task,)),
        optimizer=optimizer_id,
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts={},
        ready_time_ms={"preferred": 0},
        link_specs=[],
        link_snapshots=[],
    )

    assert plan.assignments[0].target_node_id == "edge"


def test_disabling_fallback_turns_preferences_into_a_hard_filter() -> None:
    nodes = {
        "robot": _node("robot", NodeKind.ROBOT),
        "edge": _node("edge", NodeKind.EDGE),
    }
    task = TaskInstance(
        "strict",
        "wf",
        "strict",
        "robot",
        TaskSpec(
            "custom_strict",
            TaskClass.REALTIME_OFFLOADABLE,
            placement_constraints=PlacementConstraints(
                allowed_node_kinds=(NodeKind.EDGE,),
                preferred_node_kinds=(NodeKind.EDGE,),
                allow_source_node=True,
                allow_fallback=False,
            ),
        ),
    )

    candidates = allowed_nodes(
        task,
        nodes.values(),
        {
            node_id: NodeSnapshot(node_id)
            for node_id in nodes
        },
    )

    assert [item.node_id for item in candidates] == ["edge"]


def test_custom_registry_extends_instead_of_hiding_builtin_optimizers() -> None:
    class UnusedOptimizer:
        optimizer_id = "custom_unused"

        def solve(self, problem):
            raise AssertionError("custom optimizer should not be selected")

    node = _node("edge", NodeKind.EDGE)
    task = _pinned_task("task", "edge", NodeKind.EDGE)
    registry = OptimizerRegistry()
    registry.register(UnusedOptimizer())

    plan = plan_scheduling_epoch(
        SchedulingEpoch("registry", 0, (task,)),
        optimizer="greedy_cost",
        node_specs={"edge": node},
        node_snapshots={"edge": NodeSnapshot("edge")},
        parent_artifacts={},
        ready_time_ms={"task": 0},
        link_specs=[],
        link_snapshots=[],
        registry=registry,
    )

    assert plan.optimizer_id == "heuristic"
    assert plan.policy_id == "greedy_cost"
    assert plan.assignments[0].target_node_id == "edge"


def test_candidate_generation_materializes_artifact_iterables_once() -> None:
    nodes = {
        "a-source": _node("a-source", NodeKind.ROBOT),
        "b-edge": _node("b-edge", NodeKind.EDGE),
    }
    task = TaskInstance(
        "consumer",
        "wf",
        "consumer",
        "a-source",
        TaskSpec(
            "custom_consumer",
            TaskClass.REALTIME_OFFLOADABLE,
            placement_constraints=PlacementConstraints(
                allowed_node_kinds=(NodeKind.EDGE,),
                allow_source_node=True,
            ),
        ),
        dependency_task_ids=("producer",),
    )
    artifact = (
        item
        for item in (
            # A one-shot iterable must produce the same immutable inputs for
            # every node candidate.
            ArtifactRef(
                "artifact",
                "producer",
                "a-source",
                1,
            ),
        )
    )
    problem = build_scheduling_problem(
        SchedulingEpoch("generator-input", 0, (task,)),
        node_specs=nodes,
        node_snapshots={
            node_id: NodeSnapshot(node_id)
            for node_id in nodes
        },
        parent_artifacts={"consumer": artifact},
        ready_time_ms={"consumer": 0},
        link_specs=(
            LinkSpec(
                "source-edge",
                "a-source",
                "b-edge",
                100,
            ),
        ),
        link_snapshots=(
            LinkSnapshot("source-edge", 100),
        ),
    )

    assert len(problem.candidates["consumer"]) == 2
    assert all(
        candidate.feasible
        for candidate in problem.candidates["consumer"]
    )


def test_failed_builtin_id_override_uses_builtin_fallback() -> None:
    class BrokenHeuristic:
        optimizer_id = "heuristic"

        def solve(self, problem):
            raise RuntimeError("broken override")

    node = _node("edge", NodeKind.EDGE)
    task = _pinned_task("task", "edge", NodeKind.EDGE)
    registry = OptimizerRegistry()
    registry.register(BrokenHeuristic())

    plan = plan_scheduling_epoch(
        SchedulingEpoch("registry-repair", 0, (task,)),
        optimizer="dag_deadline",
        node_specs={"edge": node},
        node_snapshots={"edge": NodeSnapshot("edge")},
        parent_artifacts={},
        ready_time_ms={"task": 0},
        link_specs=[],
        link_snapshots=[],
        registry=registry,
    )

    assert plan.optimizer_id == "heuristic"
    assert plan.policy_id == "dag_deadline"
    assert plan.assignments[0].target_node_id == "edge"
    assert plan.diagnostics["fallback_optimizer"] == "heuristic"
    assert (
        plan.diagnostics["repaired_from_optimizer"]
        == "heuristic"
    )


def test_engine_and_coordinator_resolve_partial_external_inputs_equally() -> None:
    robot = _node("robot", NodeKind.ROBOT, max_concurrency=2)
    edge = _node("edge", NodeKind.EDGE, max_concurrency=2)
    producer = TaskInstance(
        "producer",
        "wf",
        "producer",
        "robot",
        TaskSpec(
            "custom_producer",
            TaskClass.REALTIME_OFFLOADABLE,
            output_size_mb=20,
            output_ports=(DataPort("features", "features"),),
            placement_constraints=PlacementConstraints(
                pinned_node_id="robot",
                allowed_node_kinds=(NodeKind.ROBOT,),
            ),
        ),
        deadline_time_ms=100_000,
        expected_accuracy=1,
    )
    consumer = TaskInstance(
        "consumer",
        "wf",
        "consumer",
        "robot",
        TaskSpec(
            "custom_consumer",
            TaskClass.REALTIME_OFFLOADABLE,
            input_size_mb=20,
            input_ports=(
                DataPort("features", "features"),
                DataPort("camera", "image"),
            ),
            output_ports=(DataPort("result", "result"),),
            placement_constraints=PlacementConstraints(
                pinned_node_id="edge",
                allowed_node_kinds=(NodeKind.EDGE,),
            ),
        ),
        dependency_task_ids=("producer",),
        deadline_time_ms=100_000,
        expected_accuracy=1,
    )
    workflow = WorkflowSpec(
        "wf",
        (producer, consumer),
        data_edges=(
            DataEdge(
                "producer",
                "features",
                "consumer",
                "features",
                "features",
            ),
        ),
    )
    nodes = [robot, edge]
    snapshots = [NodeSnapshot("robot"), NodeSnapshot("edge")]
    links = [LinkSpec("robot-edge", "robot", "edge", 100)]
    link_snapshots = [LinkSnapshot("robot-edge", 100)]

    simulated = run_workflow_simulation(
        workflow,
        nodes,
        snapshots,
        network_jitter=0,
        resource_noise=0,
        link_specs=links,
        link_snapshots=link_snapshots,
    )
    coordinated = CentralCoordinator(
        InProcessRuntime(nodes, snapshots),
        link_specs=links,
        link_snapshots=link_snapshots,
    ).run(workflow)

    assert simulated.metrics["bandwidth_mb"] == pytest.approx(30)
    assert coordinated.metrics["transferred_mb"] == pytest.approx(30)


def test_late_drop_contributes_to_engine_and_coordinator_makespan() -> None:
    robot = _node("robot", NodeKind.ROBOT)
    edge = _node("edge", NodeKind.EDGE)
    task = TaskInstance(
        "late-drop",
        "wf",
        "late-drop",
        "robot",
        TaskSpec(
            "custom_drop",
            TaskClass.REALTIME_OFFLOADABLE,
            input_size_mb=1,
            placement_constraints=PlacementConstraints(
                pinned_node_id="edge",
                allowed_node_kinds=(NodeKind.EDGE,),
            ),
        ),
        arrival_time_ms=100,
        deadline_time_ms=100_000,
    )
    workflow = WorkflowSpec("wf", (task,))
    nodes = [robot, edge]
    snapshots = [NodeSnapshot("robot"), NodeSnapshot("edge")]

    simulated = run_workflow_simulation(
        workflow,
        nodes,
        snapshots,
        link_specs=[],
        link_snapshots=[],
    )
    coordinated = CentralCoordinator(
        InProcessRuntime(nodes, snapshots),
        link_specs=(),
        link_snapshots=(),
    ).run(workflow)

    assert simulated.task_results[0].finish_time_ms == 100
    assert simulated.metrics["makespan_ms"] == 100
    assert coordinated.metrics["makespan_ms"] == 100
    assert coordinated.task_results[0]["attempt_count"] == 1


def test_fail_fast_drop_does_not_reserve_sibling_batch_work() -> None:
    robot = _node(
        "robot",
        NodeKind.ROBOT,
        max_concurrency=2,
    )
    edge = _node("edge", NodeKind.EDGE)
    dropped = TaskInstance(
        "a-drop",
        "wf",
        "a-drop",
        "robot",
        TaskSpec(
            "custom_drop",
            TaskClass.REALTIME_OFFLOADABLE,
            input_size_mb=1,
            placement_constraints=PlacementConstraints(
                pinned_node_id="edge",
                allowed_node_kinds=(NodeKind.EDGE,),
            ),
        ),
        deadline_time_ms=100_000,
    )
    sibling = _pinned_task(
        "b-sibling",
        "robot",
        NodeKind.ROBOT,
        compute=10,
    )

    report = run_workflow_simulation(
        WorkflowSpec(
            "wf",
            (dropped, sibling),
            failure_policy=FailurePolicy.FAIL_FAST,
        ),
        [robot, edge],
        [NodeSnapshot("robot"), NodeSnapshot("edge")],
        network_jitter=0,
        resource_noise=0,
        link_specs=[],
        link_snapshots=[],
    )
    by_id = {item.task_id: item for item in report.task_results}

    assert by_id["a-drop"].state == "dropped"
    assert by_id["b-sibling"].state == "skipped"
    assert report.node_utilization["robot"] == 0


def test_fail_fast_execution_failure_does_not_start_sibling_work() -> None:
    robot = _node(
        "robot",
        NodeKind.ROBOT,
        cpu=100,
        max_concurrency=2,
    )
    failing = _pinned_task(
        "a-failing",
        "robot",
        NodeKind.ROBOT,
        compute=1,
        expected_accuracy=0,
    )
    sibling = _pinned_task(
        "b-sibling",
        "robot",
        NodeKind.ROBOT,
        compute=50,
    )

    report = run_workflow_simulation(
        WorkflowSpec(
            "wf",
            (failing, sibling),
            failure_policy=FailurePolicy.FAIL_FAST,
        ),
        [robot],
        [NodeSnapshot("robot")],
        seed=0,
        network_jitter=0,
        resource_noise=0,
        link_specs=[],
        link_snapshots=[],
    )
    by_id = {item.task_id: item for item in report.task_results}

    assert by_id["a-failing"].state == "failed"
    assert by_id["b-sibling"].state == "skipped"
    assert by_id["b-sibling"].compute_time_ms == 0
    assert 0 < report.node_utilization["robot"] <= 1


def test_dispatch_rejection_is_recorded_as_an_attempt() -> None:
    robot = _node("robot", NodeKind.ROBOT)
    task = _pinned_task(
        "unsupported",
        "robot",
        NodeKind.ROBOT,
        arrival_time_ms=50,
    )
    runtime = InProcessRuntime(
        [robot],
        [NodeSnapshot("robot")],
        supported_task_types={"robot": ("different_type",)},
    )

    report = CentralCoordinator(
        runtime,
        link_specs=(),
        link_snapshots=(),
    ).run(
        WorkflowSpec("wf", (task,)),
        max_attempts=1,
    )
    result = report.task_results[0]

    assert result["state"] == "dropped"
    assert result["attempt_count"] == 1
    assert result["attempts"][0]["state"] == "failed"
    assert (
        result["attempts"][0]["error_code"]
        == "task_capability_not_declared"
    )
    assert report.metrics["makespan_ms"] == 50
