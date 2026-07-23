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
from mars.synthetic_workloads import load_default_synthetic_workloads


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
        self.problems = []
        self.plans: list[SchedulingPlan] = []

    def solve(self, problem):
        self.problems.append(problem)
        self.epochs.append(
            tuple(task.task_id for task in problem.epoch.ready_tasks)
        )
        self.link_inventories.append(
            tuple(link.link_id for link in problem.link_specs)
        )
        baseline = HeuristicOptimizer().solve(problem)
        plan = replace(
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
        self.plans.append(plan)
        return plan


class _DeferredOptimizer:
    optimizer_id = "deferred"

    def solve(self, problem):
        return SchedulingPlan(
            problem_id=problem.problem_id,
            snapshot_id=problem.snapshot.snapshot_id,
            policy_id=problem.policy.policy_id,
            policy_version=problem.policy.version,
            epoch_id=problem.epoch.epoch_id,
            optimizer_id=self.optimizer_id,
            assignments=(),
            deferred_task_ids=tuple(
                task.task_id for task in problem.epoch.ready_tasks
            ),
        )


class _RecordingRuntime(InProcessRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dispatched_commands = []

    async def dispatch(self, command):
        self.dispatched_commands.append(command)
        return await super().dispatch(command)


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


def test_coordinator_dispatches_the_exact_validated_assignment() -> None:
    nodes, snapshots = _inventory()
    runtime = _RecordingRuntime(nodes, snapshots)
    optimizer = _TrackingOptimizer()
    registry = OptimizerRegistry()
    registry.register(optimizer)
    workload = load_default_synthetic_workloads().get(
        "object_detection"
    )
    spec = replace(
        workload.to_task_spec(),
        placement_constraints=PlacementConstraints(
            pinned_node_id="edge",
            allowed_node_kinds=(NodeKind.EDGE,),
        ),
    )
    task = TaskInstance(
        "detect",
        "wf",
        "detect",
        "robot",
        spec,
        deadline_time_ms=10_000,
    )

    report = CentralCoordinator(
        runtime,
        optimizer_registry=registry,
    ).run(
        WorkflowSpec("wf", (task,)),
        algorithm="tracking",
    )

    assert report.workflow["state"] == "succeeded"
    validated_assignment = optimizer.plans[0].assignments[0]
    validated_resource_reservation = (
        optimizer.plans[0].node_reservations[0]
    )
    validated_transfer_reservations = (
        optimizer.plans[0].transfer_reservations
    )
    dispatch = runtime.dispatched_commands[0]
    dispatched_assignment = dispatch.assignment
    assert dispatched_assignment is validated_assignment
    assert dispatch.resource_reservation is validated_resource_reservation
    assert validated_transfer_reservations
    assert dispatch.transfer_reservations == validated_transfer_reservations
    solved_problem = optimizer.problems[0]
    assert dispatch.problem_id == optimizer.plans[0].problem_id
    assert dispatch.snapshot_id == optimizer.plans[0].snapshot_id
    assert dispatch.policy_id == optimizer.plans[0].policy_id
    assert dispatch.policy_version == optimizer.plans[0].policy_version
    assert (
        dispatch.input_artifact_bindings
        == solved_problem.input_artifact_bindings["detect"]
    )
    assert all(
        dispatched is validated
        for dispatched, validated in zip(
            dispatch.transfer_reservations,
            validated_transfer_reservations,
        )
    )
    assert dispatched_assignment.compute_ms == 22
    assert dispatched_assignment.estimated_finish_ms > 22


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
