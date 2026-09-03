"""Input-free typed producers must not acquire phantom source artifacts."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent.artifacts import ArtifactFiles
from agent.executor import ExecutionResult
from agent.real_service import ExecutionAgentService, start_execution_server
from agent.service import AgentConfig
from agent.telemetry import detected_node
from mars.coordinator import CentralCoordinator
from mars.dag import TaskManager, resolve_task_input_bindings
from mars.domain import (
    DataPort,
    NodeKind,
    NodeSnapshot,
    NodeSpec,
    PlacementConstraints,
    ResourceClass,
    TaskClass,
    TaskInstance,
    TaskSpec,
    WorkflowSpec,
)
from mars.domain.artifact import ArtifactRef, InputArtifactBinding
from mars.optimizers import SchedulingEpoch
from mars.runtime import GrpcRuntimeAdapter
from mars.scheduler import build_scheduling_problem, estimate_candidate


def _task(*, input_size_mb=0.0, input_ports=(), output_ports=None):
    return TaskInstance(
        "source",
        "typed-source-workflow",
        "Generate data",
        "robot_1",
        TaskSpec(
            "generated_source",
            TaskClass.REALTIME_OFFLOADABLE,
            input_size_mb=input_size_mb,
            dominant_resource=ResourceClass.CPU,
            input_ports=input_ports,
            output_ports=(DataPort("result", "example.Generated"),)
            if output_ports is None
            else output_ports,
            placement_constraints=PlacementConstraints(pinned_node_id="robot_1"),
        ),
        deadline_time_ms=10_000,
        expected_accuracy=1,
    )


def _nodes():
    nodes = {
        "robot_1": NodeSpec("robot_1", NodeKind.ROBOT, 4, 0, 8, 100, 0),
        "edge_pc": NodeSpec("edge_pc", NodeKind.EDGE, 4, 0, 8, 100, 0),
    }
    return nodes, {node_id: NodeSnapshot(node_id) for node_id in nodes}


def _bindings(task):
    manager = TaskManager()
    manager.submit(WorkflowSpec(task.workflow_id, (task,)))
    return resolve_task_input_bindings(manager, task.task_id)


def _problem(task, **kwargs):
    nodes, snapshots = _nodes()
    return build_scheduling_problem(
        SchedulingEpoch("source-epoch", 0, (task,)),
        node_specs=nodes,
        node_snapshots=snapshots,
        ready_time_ms={task.task_id: 0},
        link_specs=(),
        link_snapshots=(),
        **kwargs,
    )


@pytest.mark.parametrize("input_mode", ["exact", "legacy", "omitted"])
def test_typed_source_has_no_bindings_locations_or_transfers(input_mode):
    task = _task()
    assert _bindings(task) == ()
    kwargs = (
        {"input_artifact_bindings": {task.task_id: ()}}
        if input_mode == "exact"
        else {"parent_artifacts": {task.task_id: ()}}
        if input_mode == "legacy"
        else {}
    )
    problem = _problem(task, **kwargs)
    assert problem.input_artifact_bindings[task.task_id] == ()
    candidate = problem.candidates[task.task_id][0]
    assert candidate.feasible
    assert candidate.input_locations == ()
    assert candidate.transfers == ()
    assert candidate.communication_ms == 0


def test_input_free_typed_source_can_execute_remotely_without_an_upload_link():
    task = _task()
    task = replace(
        task,
        spec=replace(
            task.spec,
            placement_constraints=PlacementConstraints(pinned_node_id="edge_pc"),
        ),
    )
    candidate = _problem(task).candidates[task.task_id][0]
    assert candidate.node_id == "edge_pc"
    assert candidate.feasible
    assert candidate.input_locations == ()
    assert candidate.transfers == ()


@pytest.mark.parametrize("input_size_mb", [0.0, 2.0])
def test_declared_external_input_ports_still_have_source_bindings(input_size_mb):
    task = _task(
        input_size_mb=input_size_mb,
        input_ports=(DataPort("image", "example.Image"),),
    )
    bindings = _bindings(task)
    assert len(bindings) == 1
    assert bindings[0].consumer_port == "image"
    assert bindings[0].artifact.size_mb == input_size_mb
    for kwargs in (
        {"input_artifact_bindings": {task.task_id: bindings}},
        {"input_artifact_bindings": {task.task_id: ()}},
        {"parent_artifacts": {}},
    ):
        problem = _problem(task, **kwargs)
        assert len(problem.input_artifact_bindings[task.task_id]) == 1
        candidate = problem.candidates[task.task_id][0]
        assert candidate.feasible
        assert candidate.input_locations == ("robot_1",)
        assert len(candidate.transfers) == 1
        assert candidate.transfers[0].size_mb == input_size_mb


def test_materialized_typed_external_input_remains_exact():
    task = _task(input_ports=(DataPort("image", "example.Image"),))
    artifact = ArtifactRef(
        "actual-image",
        "",
        "robot_1",
        0.125,
        uri="example://images/input",
        producer_port="camera",
        message_type="example.Image",
    )
    binding = InputArtifactBinding(task.task_id, "image", artifact)
    problem = _problem(task, input_artifact_bindings={task.task_id: (binding,)})
    assert problem.input_artifact_bindings[task.task_id] == (binding,)
    candidate = problem.candidates[task.task_id][0]
    assert candidate.input_locations == ("robot_1",)
    assert candidate.transfers[0].transfer_id == "actual-image"
    assert candidate.transfers[0].size_mb == 0.125


@pytest.mark.parametrize("input_size_mb", [0.0, 2.0])
def test_untyped_legacy_tasks_keep_their_implicit_source_input(input_size_mb):
    task = _task(input_size_mb=input_size_mb, output_ports=())
    bindings = _bindings(task)
    assert len(bindings) == 1
    assert bindings[0].consumer_port == "__external_input__"
    nodes, snapshots = _nodes()
    direct = estimate_candidate(
        task,
        nodes["robot_1"],
        ready_time_ms=0,
        node_available_ms=0,
        node_specs=nodes,
        node_snapshots=snapshots,
        parent_artifacts=(),
    )
    normalized = _problem(task).candidates[task.task_id][0]
    assert direct.input_locations == normalized.input_locations == ("robot_1",)
    assert (
        direct.transfers[0].size_mb == normalized.transfers[0].size_mb == input_size_mb
    )


def test_declared_input_bytes_without_ports_keep_implicit_source_input():
    task = _task(input_size_mb=0.25)
    assert len(_bindings(task)) == 1
    candidate = _problem(task).candidates[task.task_id][0]
    assert candidate.input_locations == ("robot_1",)
    assert candidate.transfers[0].size_mb == 0.25


def test_real_runtime_executes_input_free_typed_source_without_artifact_fetch(tmp_path):
    class GeneratorExecutor:
        ports = {
            "generated_source": {
                "inputs": {},
                "outputs": {"result": "example.Generated"},
            },
        }

        def __init__(self):
            self.calls = []

        async def execute(self, task_type, inputs, seed):
            self.calls.append((task_type, inputs, seed))
            return ExecutionResult({"result": {"sum": sum(range(1000))}}, 1.0)

    async def run():
        executor = GeneratorExecutor()
        service = ExecutionAgentService(
            AgentConfig("robot_1", "127.0.0.1:0", detected_node("robot"), {}),
            executor,
            ArtifactFiles(tmp_path),
            {},
        )
        server, port = await start_execution_server(service)
        runtime = GrpcRuntimeAdapter({"robot_1": f"127.0.0.1:{port}"})
        task = _task()
        try:
            report = await CentralCoordinator(
                runtime,
                link_specs=(),
                link_snapshots=(),
            ).run_async(WorkflowSpec(task.workflow_id, (task,)), seed=7)
            assert report.workflow["state"] == "succeeded"
            assert executor.calls == [("generated_source", {}, 7)]
            assert service.records[0]["input_artifacts"] == []
            assert service.records[0]["remote_input_bytes"] == 0
            assert len(service._history[0].outputs) == 1
        finally:
            await runtime.close()
            await service.close()
            await server.stop(0)

    asyncio.run(run())
