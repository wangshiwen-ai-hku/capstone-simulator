from dataclasses import dataclass, field
import enum
import json
from types import SimpleNamespace

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
    SolveTraceContext,
    SolveTraceEntry,
    SolveTracePhase,
)
from mars.runtime import InProcessRuntime


@dataclass(frozen=True)
class _WarmStart:
    iteration: int
    primal: tuple[float, ...]


@dataclass
class _MutableWarmStart:
    cursor: int


@dataclass(frozen=True)
class _HiddenMutableWarmStart:
    cursor: int
    hidden: list[int] = field(default_factory=list, init=False)


class _WrongIdentityOptimizer:
    optimizer_id = "wrong-identity"

    def solve(self, problem):
        return HeuristicOptimizer().solve(problem)


class _InvalidStateEnum(enum.Enum):
    NAN = float("nan")
    OPAQUE = object()


def _fake_problem():
    return SimpleNamespace(
        problem_id="problem",
        snapshot=SimpleNamespace(snapshot_id="snapshot"),
        epoch=SimpleNamespace(epoch_id="epoch"),
        policy=SimpleNamespace(policy_id="policy", version="1"),
        solve_limits=SimpleNamespace(
            solve_budget_ms=10.0,
            max_iterations=0,
        ),
    )


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

    assert state.continuation_for("milp") == next_continuation
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


def test_continuation_history_snapshots_mutable_payloads() -> None:
    payload = {"cursor": 1, "nested": [1]}
    state = OptimizerSolveState(session_id="mutable-snapshot")
    continuation = OptimizerContinuation(
        optimizer_id="milp",
        schema_version="milp.state.v1",
        updated_problem_id="problem-1",
        payload=payload,
    )

    state.set_continuation(continuation)
    payload["cursor"] = 99
    payload["nested"].append(2)

    archived = state.as_dict()["continuation_history"][0]["payload"]
    assert archived == {"cursor": 1, "nested": [1]}
    current = state.continuation_for("milp")
    assert current is not None
    with pytest.raises(TypeError):
        current.payload["cursor"] = 7
    with pytest.raises(AttributeError):
        current.payload["nested"].append(3)
    assert state.as_dict()["continuation_history"][0]["payload"] == {
        "cursor": 1,
        "nested": [1],
    }
    state.set_continuation(current)
    assert state.continuation_for("milp") == current
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        state.set_continuation(
            OptimizerContinuation(
                optimizer_id="bad-map",
                schema_version="bad-map.v1",
                updated_problem_id="problem-1",
                payload={1: "integer", "1": "string"},
            )
        )

    dataclass_payload = _WarmStart(iteration=4, primal=(0.0, 1.0))
    state.set_continuation(
        OptimizerContinuation(
            optimizer_id="typed",
            schema_version="typed.v1",
            updated_problem_id="problem-1",
            payload=dataclass_payload,
        )
    )
    assert state.continuation_for("typed").payload == dataclass_payload
    with pytest.raises(TypeError, match="must be frozen"):
        state.set_continuation(
            OptimizerContinuation(
                optimizer_id="mutable-dataclass",
                schema_version="mutable.v1",
                updated_problem_id="problem-1",
                payload=_MutableWarmStart(cursor=1),
            )
        )
    with pytest.raises(TypeError, match="fields must all use init=True"):
        state.set_continuation(
            OptimizerContinuation(
                optimizer_id="hidden-field-dataclass",
                schema_version="hidden.v1",
                updated_problem_id="problem-1",
                payload=_HiddenMutableWarmStart(cursor=1),
            )
        )


def test_restored_continuations_are_independent_frozen_snapshots() -> None:
    payload = {"cursor": 1, "nested": [1]}
    continuation = OptimizerContinuation(
        optimizer_id="milp",
        schema_version="milp.state.v1",
        updated_problem_id="problem-1",
        payload=payload,
    )
    state = OptimizerSolveState(
        session_id="restored-state",
        continuations={"milp": continuation},
        continuation_history=[continuation],
    )

    payload["cursor"] = 9
    payload["nested"].append(2)

    assert state.as_dict()["continuations"]["milp"]["payload"] == {
        "cursor": 1,
        "nested": [1],
    }
    assert state.as_dict()["continuation_history"][0]["payload"] == {
        "cursor": 1,
        "nested": [1],
    }


@pytest.mark.parametrize(
    ("payload", "error"),
    (
        (_InvalidStateEnum.NAN, ValueError),
        (_InvalidStateEnum.OPAQUE, TypeError),
    ),
)
def test_continuation_enum_values_must_be_json_safe(
    payload,
    error,
) -> None:
    state = OptimizerSolveState(session_id="invalid-enum")

    with pytest.raises(error):
        state.set_continuation(
            OptimizerContinuation(
                optimizer_id="invalid-enum",
                schema_version="invalid-enum.v1",
                updated_problem_id="problem",
                payload=payload,
            )
        )


@pytest.mark.parametrize(
    ("values", "error"),
    (
        ({"termination_reason": object()}, TypeError),
        ({"has_incumbent": "yes"}, TypeError),
        ({"iteration": True}, ValueError),
        ({"evaluated_work_units": False}, ValueError),
    ),
)
def test_trace_top_level_values_are_strictly_serializable(
    values,
    error,
) -> None:
    state = OptimizerSolveState(session_id="invalid-trace")
    context = state.begin(_fake_problem(), optimizer_id="optimizer")

    with pytest.raises(error):
        state.record(context, SolveTracePhase.ITERATION, **values)


def test_sparse_restored_trace_does_not_reuse_a_solve_id() -> None:
    original_context = SolveTraceContext(
        solve_id="sparse:solve:2",
        frame_index=1,
        problem_id="problem",
        snapshot_id="snapshot",
        epoch_id="epoch",
        policy_id="policy",
        policy_version="1",
        optimizer_id="optimizer",
        optimizer_version="",
        work_unit="iteration",
        solve_budget_ms=10.0,
        max_iterations=0,
    )
    state = OptimizerSolveState(
        session_id="sparse",
        trace_entries=[
            SolveTraceEntry(
                sequence=1,
                context=original_context,
                phase=SolveTracePhase.STARTED,
            )
        ],
    )

    next_context = state.begin(
        _fake_problem(),
        optimizer_id="optimizer",
    )

    assert next_context.solve_id == "sparse:solve:3"
    assert state.as_dict()["solve_count"] == 2


def test_sparse_restored_trace_continues_frame_numbering() -> None:
    restored_context = SolveTraceContext(
        solve_id="sparse-frames:solve:1",
        frame_index=9,
        problem_id="restored-problem",
        snapshot_id="snapshot",
        epoch_id="epoch",
        policy_id="policy",
        policy_version="1",
        optimizer_id="optimizer",
        optimizer_version="",
        work_unit="iteration",
        solve_budget_ms=10.0,
        max_iterations=0,
    )
    state = OptimizerSolveState(
        session_id="sparse-frames",
        trace_entries=[
            SolveTraceEntry(
                sequence=1,
                context=restored_context,
                phase=SolveTracePhase.STARTED,
            )
        ],
    )

    next_context = state.begin(
        _fake_problem(),
        optimizer_id="optimizer",
    )

    assert next_context.frame_index == 10
    assert [
        entry.context.frame_index for entry in state.entries
    ] == [9, 10]
    assert state.as_dict()["frame_count"] == 2


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
        "fallback",
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
