from __future__ import annotations

from dataclasses import replace

from mars.coordinator import CentralCoordinator
from mars.models import (
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
    HeuristicOptimizer,
    OptimizerRegistry,
    SchedulingPlan,
)
from mars.runtime import InProcessRuntime


def _inventory() -> tuple[list[NodeSpec], list[NodeSnapshot]]:
    nodes = [
        NodeSpec(
            "robot",
            NodeKind.ROBOT,
            4,
            2,
            16,
            100,
            2,
        ),
        NodeSpec(
            "edge",
            NodeKind.EDGE,
            12,
            8,
            64,
            500,
            5,
            capabilities=("vision",),
            max_concurrency=2,
        ),
    ]
    return nodes, [
        NodeSnapshot("robot", power_w=25),
        NodeSnapshot("edge", power_w=120),
    ]


def _task(
    task_id: str,
    *,
    label: TaskClass = TaskClass.REALTIME_OFFLOADABLE,
    target: str = "robot",
    input_size_mb: float = 0,
) -> TaskInstance:
    kind = NodeKind.ROBOT if target == "robot" else NodeKind.EDGE
    return TaskInstance(
        task_id,
        "wf",
        task_id,
        "robot",
        TaskSpec(
            task_type=f"custom_{task_id}",
            task_class=label,
            compute_demand=1,
            gpu_demand=0.5,
            input_size_mb=input_size_mb,
            placement_constraints=PlacementConstraints(
                pinned_node_id=target,
                allowed_node_kinds=(kind,),
            ),
        ),
        deadline_time_ms=10_000,
    )


class _TrackingOptimizer:
    optimizer_id = "tracking"

    def __init__(self) -> None:
        self.epochs: list[tuple[str, ...]] = []
        self.link_inventories: list[tuple[str, ...]] = []

    def solve(self, problem):
        self.epochs.append(
            tuple(task.task_id for task in problem.epoch.ready_tasks)
        )
        self.link_inventories.append(
            tuple(link.link_id for link in problem.link_specs)
        )
        baseline = HeuristicOptimizer("greedy_cost").solve(problem)
        return replace(
            baseline,
            optimizer_id=self.optimizer_id,
            assignments=tuple(
                replace(
                    assignment,
                    optimizer_id=self.optimizer_id,
                    reason="tracking optimizer",
                )
                for assignment in baseline.assignments
            ),
        )


class _DeferredOptimizer:
    optimizer_id = "deferred"

    def solve(self, problem):
        return SchedulingPlan(
            epoch_id=problem.epoch.epoch_id,
            optimizer_id=self.optimizer_id,
            assignments=(),
            deferred_task_ids=tuple(
                task.task_id for task in problem.epoch.ready_tasks
            ),
        )


def test_coordinator_plans_the_whole_ready_batch_before_commit() -> None:
    nodes, snapshots = _inventory()
    runtime = InProcessRuntime(nodes, snapshots)
    optimizer = _TrackingOptimizer()
    registry = OptimizerRegistry()
    registry.register(optimizer)
    workflow = WorkflowSpec(
        "wf",
        (
            _task("a", target="robot"),
            _task("b", target="edge"),
        ),
    )

    report = CentralCoordinator(
        runtime,
        link_specs=(),
        link_snapshots=(),
        optimizer_registry=registry,
    ).run(workflow, algorithm="tracking")

    assert report.workflow["state"] == "succeeded"
    assert set(optimizer.epochs[0]) == {"a", "b"}
    assert optimizer.link_inventories[0] == ()
    assert any(
        event.event_type == "scheduling_epoch_planned"
        for event in report.events
    )


def test_explicit_empty_topology_is_respected_by_runtime_path() -> None:
    nodes, snapshots = _inventory()
    workflow = WorkflowSpec(
        "wf",
        (_task("remote", target="edge", input_size_mb=5),),
    )

    report = CentralCoordinator(
        InProcessRuntime(nodes, snapshots),
        link_specs=(),
        link_snapshots=(),
    ).run(workflow)

    result = report.task_results[0]
    assert result["state"] == "dropped"
    assert result["target_node_id"] == ""


def test_runtime_enforces_explicit_constraints_not_business_label() -> None:
    nodes, snapshots = _inventory()
    task = _task(
        "label-independent",
        label=TaskClass.LOCAL_SAFETY,
        target="edge",
    )

    report = CentralCoordinator(
        InProcessRuntime(nodes, snapshots),
        link_specs=(),
        link_snapshots=(),
    ).run(WorkflowSpec("wf", (task,)))

    result = report.task_results[0]
    assert result["state"] == "succeeded"
    assert result["target_node_id"] == "edge"
    assert report.metrics["safety_violation_count"] == 0


def test_deferred_only_plan_fails_fast_instead_of_spinning() -> None:
    nodes, snapshots = _inventory()
    registry = OptimizerRegistry()
    registry.register(_DeferredOptimizer())
    coordinator = CentralCoordinator(
        InProcessRuntime(nodes, snapshots),
        link_specs=(),
        link_snapshots=(),
        optimizer_registry=registry,
    )

    try:
        coordinator.run(
            WorkflowSpec("wf", (_task("deferred"),)),
            algorithm="deferred",
        )
    except RuntimeError as exc:
        assert "deferred every ready task" in str(exc)
    else:
        raise AssertionError("deferred-only plan did not fail fast")
