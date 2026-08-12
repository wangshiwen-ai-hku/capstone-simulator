from dataclasses import dataclass
import json

import pytest

from mars.coordinator import CentralCoordinator
from mars.domain import (
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
    OptimizerContinuation,
    OptimizerRegistry,
    OptimizerSolveState,
)
from mars.runtime import InProcessRuntime


@dataclass(frozen=True)
class _WarmStart:
    iteration: int
    primal: tuple[float, ...]


class _WrongIdentityOptimizer:
    optimizer_id = "wrong-identity"

    def solve(self, problem):
        return HeuristicOptimizer().solve(problem)


def test_optimizer_solve_state_serializes_typed_continuation() -> None:
    state = OptimizerSolveState(session_id="workflow:warm-start")
    continuation = OptimizerContinuation(
        optimizer_id="milp",
        schema_version="milp.warm-start.v1",
        updated_problem_id="epoch-1:problem",
        payload=_WarmStart(iteration=8, primal=(0.0, 1.0)),
        iteration=8,
        objective_key=(12.5,),
    )
    next_continuation = OptimizerContinuation(
        optimizer_id="milp",
        schema_version="milp.warm-start.v1",
        updated_problem_id="epoch-2:problem",
        payload=_WarmStart(iteration=13, primal=(1.0, 0.0)),
        iteration=13,
        objective_key=(9.0,),
    )

    state.set_continuation(continuation)
    state.set_continuation(next_continuation)

    assert state.continuation_for("milp") is next_continuation
    assert state.as_dict()["continuations"]["milp"] == {
        "optimizer_id": "milp",
        "schema_version": "milp.warm-start.v1",
        "updated_problem_id": "epoch-2:problem",
        "iteration": 13,
        "objective_key": [9.0],
        "payload": {"iteration": 13, "primal": [1.0, 0.0]},
    }
    assert [
        item["iteration"]
        for item in state.as_dict()["continuation_history"]
    ] == [8, 13]
    json.dumps(state.as_dict(), allow_nan=False)

    unsupported = OptimizerContinuation(
        optimizer_id="admm",
        schema_version="admm.state.v1",
        updated_problem_id="epoch-2:problem",
        payload=object(),
    )
    with pytest.raises(TypeError, match="dataclass or JSON-like"):
        state.set_continuation(unsupported)


def test_coordinator_records_validated_solves_across_epochs() -> None:
    nodes = (
        NodeSpec("robot", NodeKind.ROBOT, 4, 1, 8, 100, 1),
    )
    snapshots = (NodeSnapshot("robot", power_w=25),)
    placement = PlacementConstraints(
        pinned_node_id="robot",
        allowed_node_kinds=(NodeKind.ROBOT,),
    )
    task_spec = TaskSpec(
        task_type="traceable",
        task_class=TaskClass.REALTIME_OFFLOADABLE,
        compute_demand=1.0,
        placement_constraints=placement,
    )
    workflow = WorkflowSpec(
        "trace-workflow",
        (
            TaskInstance(
                "frame-1",
                "trace-workflow",
                "frame-1",
                "robot",
                task_spec,
                deadline_time_ms=1_000,
                arrival_time_ms=0,
            ),
            TaskInstance(
                "frame-2",
                "trace-workflow",
                "frame-2",
                "robot",
                task_spec,
                deadline_time_ms=1_000,
                arrival_time_ms=100,
            ),
        ),
    )

    report = CentralCoordinator(
        InProcessRuntime(nodes, snapshots),
        link_specs=(),
        link_snapshots=(),
    ).run(workflow, algorithm="binary_offload")
    solve_state = report.workflow["scheduling"]["optimizer_solve_state"]
    json.dumps(solve_state, allow_nan=False)

    assert solve_state["frame_count"] == 2
    assert solve_state["solve_count"] == 2
    summaries = solve_state["invocation_summaries"]
    assert solve_state["continuations"] == {}
    assert solve_state["continuation_history"] == []
    assert [item["frame_index"] for item in summaries] == [1, 2]
    assert {item["optimizer_id"] for item in summaries} == {
        "binary_offload"
    }
    assert {item["policy_id"] for item in summaries} == {
        "binary_offload"
    }
    phases_by_solve = {}
    for entry in solve_state["trace_entries"]:
        phases_by_solve.setdefault(entry["solve_id"], []).append(
            entry["phase"]
        )
    assert all(
        phases[0] == "started"
        and "incumbent" in phases
        and phases[-1] == "validated"
        for phases in phases_by_solve.values()
    )


def test_stateless_plugin_rejection_and_fallback_are_both_traced() -> None:
    nodes = (NodeSpec("robot", NodeKind.ROBOT, 4, 1, 8, 100, 1),)
    snapshots = (NodeSnapshot("robot", power_w=25),)
    task = TaskInstance(
        "task",
        "fallback-trace",
        "task",
        "robot",
        TaskSpec(
            task_type="traceable",
            task_class=TaskClass.REALTIME_OFFLOADABLE,
            compute_demand=1.0,
            placement_constraints=PlacementConstraints(
                pinned_node_id="robot",
                allowed_node_kinds=(NodeKind.ROBOT,),
            ),
        ),
        deadline_time_ms=1_000,
    )
    registry = OptimizerRegistry()
    registry.register(_WrongIdentityOptimizer())

    report = CentralCoordinator(
        InProcessRuntime(nodes, snapshots),
        link_specs=(),
        link_snapshots=(),
        optimizer_registry=registry,
    ).run(
        WorkflowSpec("fallback-trace", (task,)),
        algorithm="wrong-identity",
    )
    solve_state = report.workflow["scheduling"]["optimizer_solve_state"]
    summaries = solve_state["invocation_summaries"]

    assert solve_state["frame_count"] == 1
    assert solve_state["solve_count"] == 2
    assert [item["optimizer_id"] for item in summaries] == [
        "wrong-identity",
        "heuristic",
    ]
    assert [item["frame_index"] for item in summaries] == [1, 1]
    assert [item["terminal_phase"] for item in summaries] == [
        "rejected",
        "validated",
    ]
    assert report.workflow["scheduling"]["fallback_count"] == 1


def test_failed_coordinator_run_retains_its_solve_trace() -> None:
    nodes = (NodeSpec("robot", NodeKind.ROBOT, 4, 1, 8, 100, 1),)
    snapshots = (NodeSnapshot("robot", power_w=25),)
    task = TaskInstance(
        "task",
        "failed-trace",
        "task",
        "robot",
        TaskSpec(
            task_type="traceable",
            task_class=TaskClass.REALTIME_OFFLOADABLE,
            compute_demand=1.0,
            placement_constraints=PlacementConstraints(
                pinned_node_id="robot",
                allowed_node_kinds=(NodeKind.ROBOT,),
            ),
        ),
        deadline_time_ms=1_000,
    )
    registry = OptimizerRegistry()
    registry.register(_WrongIdentityOptimizer())
    coordinator = CentralCoordinator(
        InProcessRuntime(nodes, snapshots),
        link_specs=(),
        link_snapshots=(),
        optimizer_registry=registry,
        fallback_optimizer=None,
    )

    with pytest.raises(ValueError, match="optimizer_id"):
        coordinator.run(
            WorkflowSpec("failed-trace", (task,)),
            algorithm="wrong-identity",
        )

    assert coordinator.optimizer_solve_state is not None
    summary = coordinator.optimizer_solve_state.invocation_summaries()[0]
    assert summary["optimizer_id"] == "wrong-identity"
    assert summary["terminal_phase"] == "rejected"
