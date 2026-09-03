"""Real Agent admission, business boundaries, and trusted-peer artifact transfer."""

from __future__ import annotations

import asyncio
from collections import namedtuple
from contextlib import asynccontextmanager
import json
import math
from types import SimpleNamespace

import grpc
import pytest

from agent.artifacts import (
    MAX_ARTIFACT_BYTES,
    ArtifactFiles,
    ArtifactService,
    canonical_json,
    digest_bytes,
    fetch_artifact,
)
from agent.executor import ExecutionResult, NavigationExecutor
from agent.real_service import ExecutionAgentService, start_execution_server
from agent.service import AgentConfig
from agent.telemetry import HostTelemetry, detected_node
from examples.hardware_workloads import PORT_TYPES
from interfaces.proto.mars.v1 import (
    artifact_service_pb2,
    artifact_service_pb2_grpc,
    common_pb2,
    runtime_pb2,
    runtime_service_pb2,
    runtime_service_pb2_grpc,
    workflow_pb2,
)


class ControlledExecutor:
    ports = PORT_TYPES

    def __init__(self, *, block=False, outputs=None, elapsed_ms=3.0):
        self.block = block
        self.outputs = outputs
        self.elapsed_ms = elapsed_ms
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.calls = []

    async def execute(self, task_type, inputs, seed):
        self.calls.append((task_type, inputs, seed))
        self.started.set()
        try:
            if self.block:
                await self.release.wait()
            outputs = self.outputs
            if outputs is None:
                outputs = {
                    port: {"computed": seed}
                    for port in self.ports[task_type]["outputs"]
                }
            return ExecutionResult(outputs, self.elapsed_ms)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def _service(tmp_path, executor=None, **kwargs):
    return ExecutionAgentService(
        AgentConfig(
            "robot_1",
            "127.0.0.1:0",
            detected_node("robot"),
            {"cpu_util": 0.99, "memory_util": 0.99},
        ),
        executor or ControlledExecutor(),
        ArtifactFiles(tmp_path),
        {},
        **kwargs,
    )


def _request(task_type="hil_sensor", *, attempt_id="attempt-1", inputs=()):
    request = runtime_pb2.DispatchCommand(
        schema_version="mars.runtime.v1",
        attempt_id=attempt_id,
        attempt_number=1,
        problem_id="problem-1",
        snapshot_id="snapshot-1",
        policy_id="policy-1",
        policy_version="1",
        random_seed=7,
        input_artifact_bindings=inputs,
    )
    request.task.task_id = task_type
    request.task.workflow_id = "workflow-1"
    request.task.source_node_id = "robot_1"
    request.task.deadline_time_ms = 10_000
    spec = request.task.spec
    spec.task_type = task_type
    spec.task_class = common_pb2.TASK_CLASS_REALTIME_OFFLOADABLE
    spec.compute_demand = 1
    spec.latency_budget_ms = 10_000
    spec.dominant_resource = common_pb2.RESOURCE_CLASS_CPU
    spec.placement_constraints.pinned_node_id = "robot_1"
    spec.placement_constraints.allow_source_node = True
    spec.placement_constraints.required_capabilities.append("hil_navigation_v1")
    for port, message_type in PORT_TYPES[task_type]["inputs"].items():
        spec.input_ports.add(name=port, message_type=message_type)
    for port, message_type in PORT_TYPES[task_type]["outputs"].items():
        spec.output_ports.add(name=port, message_type=message_type)
    assignment = request.assignment
    assignment.task_id = task_type
    assignment.target_node_id = "robot_1"
    assignment.execution_mode = common_pb2.EXECUTION_MODE_LOCAL
    assignment.estimated_start_time_ms = 100
    assignment.estimated_finish_time_ms = 200
    assignment.compute_time_ms = 100
    assignment.epoch_id = "epoch-1"
    assignment.optimizer_id = "test-optimizer"
    assignment.input_node_ids.extend(binding.artifact.node_id for binding in inputs)
    reservation = request.resource_reservation
    reservation.reservation_id = "reservation-1"
    reservation.epoch_id = "epoch-1"
    reservation.task_id = task_type
    reservation.node_id = "robot_1"
    reservation.start_time_ms = 100
    reservation.finish_time_ms = 200
    reservation.demand.cpu_units = 1
    reservation.demand.memory_gb = 0.01
    return request


async def _completed(service):
    async def wait():
        while not service._history:
            await asyncio.sleep(0)
        return service._history[-1]

    return await asyncio.wait_for(wait(), 2)


def _artifact(files, *, agent_id="robot_1", workflow_id="workflow-1", payload=None):
    envelope = {
        "schema": "mars.hil.artifact.v1",
        "workflow_id": workflow_id,
        "producer_task_id": "hil_sensor",
        "producer_port": "observations",
        "agent_id": agent_id,
        "message_type": "hil.observations.v1",
        "payload": payload if payload is not None else {"real_values": [1, 2, 3]},
    }
    data = canonical_json(envelope)
    digest = files.put(data)
    reference = workflow_pb2.ArtifactRef(
        artifact_id=f"sha256:{digest}",
        producer_task_id="hil_sensor",
        node_id=agent_id,
        size_mb=len(data) / 1_000_000,
        uri=f"mars-artifact://{agent_id}/{digest}",
        checksum=digest,
        producer_port="observations",
        message_type="hil.observations.v1",
    )
    return reference, envelope, data


@asynccontextmanager
async def _artifact_server(service):
    server = grpc.aio.server()
    artifact_service_pb2_grpc.add_ArtifactStoreServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        await server.stop(0)


def test_content_addressed_files_are_immutable_and_hash_checked(tmp_path):
    files = ArtifactFiles(tmp_path)
    data = canonical_json({"b": 2, "a": 1})
    assert data == canonical_json({"a": 1, "b": 2})
    digest = files.put(data)
    assert digest == digest_bytes(data)
    assert files.put(data) == digest
    assert files.read(digest) == data
    assert [path.name for path in tmp_path.iterdir()] == [f"{digest}.json"]
    (tmp_path / f"{digest}.json").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        files.read(digest)
    with pytest.raises(ValueError, match="checksum mismatch"):
        files.put(data)


@pytest.mark.parametrize(
    "key", ["../secret", "/etc/passwd", "a" * 63, "F" * 64, "a" * 64 + "/.."]
)
def test_artifact_paths_cannot_escape_object_store(tmp_path, key):
    with pytest.raises(ValueError, match="SHA-256"):
        ArtifactFiles(tmp_path).read(key)


def test_artifact_store_rejects_symlinks_empty_and_oversized_content(tmp_path):
    files = ArtifactFiles(tmp_path / "objects")
    outside = tmp_path / "outside.json"
    data = b'{"secret":true}'
    outside.write_bytes(data)
    digest = digest_bytes(data)
    (files.directory / f"{digest}.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        files.read(digest)
    for value in (b"", b"x" * (MAX_ARTIFACT_BYTES + 1)):
        with pytest.raises(ValueError, match="empty or exceeds"):
            files.put(value)
    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})


def test_remote_artifact_transfer_is_allowlisted_and_returns_actual_bytes(tmp_path):
    async def run():
        producer = ArtifactFiles(tmp_path / "producer")
        consumer = ArtifactFiles(tmp_path / "consumer")
        reference, expected, data = _artifact(producer, agent_id="edge_pc")
        async with _artifact_server(ArtifactService(producer)) as endpoint:
            for _ in range(2):
                envelope, transferred = await fetch_artifact(
                    reference,
                    agent_id="robot_1",
                    files=consumer,
                    peers={"edge_pc": endpoint},
                )
                assert envelope == expected
                assert transferred == len(data)
                assert consumer.read(reference.checksum) == data
        local_ref, expected, _ = _artifact(consumer)
        envelope, transferred = await fetch_artifact(
            local_ref,
            agent_id="robot_1",
            files=consumer,
            peers={},
        )
        assert envelope == expected
        assert transferred == 0

    asyncio.run(run())


def test_unknown_peer_and_arbitrary_urls_never_open_network_connection(
    tmp_path, monkeypatch
):
    files = ArtifactFiles(tmp_path)
    reference, _, _ = _artifact(files, agent_id="unknown")

    def forbidden_channel(*args, **kwargs):
        raise AssertionError("untrusted reference attempted a connection")

    monkeypatch.setattr(grpc.aio, "insecure_channel", forbidden_channel)

    async def run():
        with pytest.raises(ValueError, match="not a configured peer"):
            await fetch_artifact(reference, agent_id="robot_1", files=files, peers={})
        for uri in (
            "http://127.0.0.1/secret",
            f"mars-artifact://unknown/{reference.checksum}?endpoint=evil",
            f"mars-artifact://other/{reference.checksum}",
            "mars-artifact://unknown/../secret",
        ):
            reference.uri = uri
            with pytest.raises(ValueError, match="invalid artifact reference"):
                await fetch_artifact(
                    reference, agent_id="robot_1", files=files, peers={}
                )

    asyncio.run(run())


def test_remote_checksum_corruption_is_not_cached(tmp_path):
    class CorruptStore(artifact_service_pb2_grpc.ArtifactStoreServicer):
        async def ReadArtifact(self, request, context):
            return artifact_service_pb2.ReadArtifactResponse(
                data=b"corrupt", sha256=request.sha256
            )

    async def run():
        producer = ArtifactFiles(tmp_path / "producer")
        consumer = ArtifactFiles(tmp_path / "consumer")
        reference, _, _ = _artifact(producer, agent_id="edge_pc")
        async with _artifact_server(CorruptStore()) as endpoint:
            with pytest.raises(ValueError, match="remote artifact checksum mismatch"):
                await fetch_artifact(
                    reference,
                    agent_id="robot_1",
                    files=consumer,
                    peers={"edge_pc": endpoint},
                )
        assert not list(consumer.directory.iterdir())

    asyncio.run(run())


def test_artifact_rpc_rejects_invalid_keys_and_missing_objects(tmp_path):
    async def run():
        async with _artifact_server(
            ArtifactService(ArtifactFiles(tmp_path))
        ) as endpoint:
            async with grpc.aio.insecure_channel(endpoint) as channel:
                stub = artifact_service_pb2_grpc.ArtifactStoreStub(channel)
                for digest, status in (
                    ("../secret", grpc.StatusCode.INVALID_ARGUMENT),
                    ("a" * 64, grpc.StatusCode.NOT_FOUND),
                ):
                    with pytest.raises(grpc.aio.AioRpcError) as failure:
                        await stub.ReadArtifact(
                            artifact_service_pb2.ReadArtifactRequest(sha256=digest)
                        )
                    assert failure.value.code() == status

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value",
    [
        ("size_mb", 1.0),
        ("producer_task_id", "other-task"),
        ("producer_port", "other-port"),
        ("message_type", "wrong.type"),
    ],
)
def test_artifact_binding_metadata_and_size_are_verified(tmp_path, field, value):
    async def run():
        files = ArtifactFiles(tmp_path)
        reference, _, _ = _artifact(files)
        setattr(reference, field, value)
        with pytest.raises(ValueError, match="declaration|binding"):
            await fetch_artifact(reference, agent_id="robot_1", files=files, peers={})

    asyncio.run(run())


def test_agent_executes_once_and_replays_identical_ack_before_and_after_completion(
    tmp_path,
):
    async def run():
        executor = ControlledExecutor(block=True)
        service = _service(tmp_path, executor)
        try:
            request = _request()
            ack = await service.DispatchTask(request, None)
            assert ack.accepted
            assert await service.DispatchTask(request, None) == ack
            await executor.started.wait()
            second = await service.DispatchTask(_request(attempt_id="attempt-2"), None)
            assert not second.accepted and second.error_code == "agent_busy"
            request.random_seed += 1
            conflict = await service.DispatchTask(request, None)
            assert (
                not conflict.accepted and conflict.error_code == "conflicting_attempt"
            )
            request.random_seed -= 1
            executor.release.set()
            completion = await _completed(service)
            assert completion.ok
            assert len(executor.calls) == 1
            assert await service.DispatchTask(request, None) == ack
            assert len(service._history) == 1
            assert completion.started_time_ms == 100
            assert completion.finished_time_ms > completion.started_time_ms
            assert completion.compute_time_ms == 3
            assert service.records[0]["energy_j"] is None
            for reference in completion.outputs:
                envelope, transferred = await fetch_artifact(
                    reference,
                    agent_id="robot_1",
                    files=service.files,
                    peers={},
                )
                assert transferred == 0
                assert envelope["payload"] == {"computed": 7}
                assert envelope["execution"]["host"]["agent_pid"] > 0
        finally:
            await service.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", "bad", "unsupported_schema"),
        ("attempt_id", "", "missing_identity"),
        ("assignment.target_node_id", "other", "wrong_target"),
        ("assignment.task_id", "other", "task_mismatch"),
        ("task.spec.task_type", "unknown", "unsupported_task_type"),
        ("inject_failure", True, "mock_failure_injection_not_supported"),
        ("task.spec.gpu_demand", 1.0, "unsupported_hardware_requirement"),
        ("assignment.estimated_start_time_ms", float("nan"), "invalid_plan_time"),
    ],
)
def test_agent_rejects_invalid_dispatch_without_running_business(
    tmp_path, field, value, error
):
    async def run():
        executor = ControlledExecutor()
        service = _service(tmp_path, executor)
        request = _request()
        target = request
        *parents, name = field.split(".")
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, name, value)
        try:
            ack = await service.DispatchTask(request, None)
            assert not ack.accepted and ack.error_code == error
            assert not executor.calls
            assert not service._active
        finally:
            await service.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_version", ""),
        ("resource_reservation.node_id", "wrong-node"),
        ("resource_reservation.task_id", "wrong-task"),
        ("resource_reservation.epoch_id", "wrong-epoch"),
        ("resource_reservation.demand.cpu_units", -1.0),
        ("resource_reservation.demand.cpu_units", float("nan")),
        ("resource_reservation.demand.cpu_units", 1_000_000.0),
        ("resource_reservation.demand.memory_gb", 1_000_000.0),
        ("resource_reservation.demand.gpu_units", 1.0),
        ("assignment.estimated_finish_time_ms", 99.0),
        ("assignment.compute_time_ms", float("nan")),
        ("task.spec.placement_constraints.allow_source_node", False),
        ("task.spec.model_requirement", "unavailable-model"),
    ],
)
def test_agent_rejects_unexecutable_or_mismatched_resource_plan(tmp_path, field, value):
    async def run():
        executor = ControlledExecutor()
        service = _service(tmp_path, executor)
        request = _request()
        target = request
        *parents, name = field.split(".")
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, name, value)
        try:
            ack = await service.DispatchTask(request, None)
            assert not ack.accepted, f"accepted invalid {field}={value}"
            assert not executor.calls
        finally:
            await service.close()

    asyncio.run(run())


def test_agent_checks_port_and_input_binding_contracts(tmp_path):
    async def run():
        service = _service(tmp_path)
        try:
            missing_output = _request()
            missing_output.task.spec.output_ports.pop()
            assert (
                await service.DispatchTask(missing_output, None)
            ).error_code == "port_contract_mismatch"
            missing_input = _request("hil_mapping")
            assert (
                await service.DispatchTask(missing_input, None)
            ).error_code == "missing_input_binding"
            artifact, _, _ = _artifact(service.files)
            wrong_binding = _request(
                "hil_mapping",
                inputs=(
                    workflow_pb2.InputArtifactBinding(
                        consumer_task_id="wrong-task",
                        consumer_port="observations",
                        artifact=artifact,
                    ),
                ),
            )
            assert (
                await service.DispatchTask(wrong_binding, None)
            ).error_code == "input_binding_mismatch"
        finally:
            await service.close()

    asyncio.run(run())


def test_agent_rejects_disallowed_node_kind_and_new_tasks_after_close(tmp_path):
    async def run():
        service = _service(tmp_path)
        request = _request()
        request.task.spec.placement_constraints.allowed_node_kinds.append(
            common_pb2.NODE_KIND_EDGE
        )
        assert (
            await service.DispatchTask(request, None)
        ).error_code == "node_kind_not_allowed"
        await service.close()
        assert (
            await service.DispatchTask(_request(), None)
        ).error_code == "agent_stopping"

    asyncio.run(run())


def test_attempt_history_limit_keeps_deduplication_and_refuses_new_work(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("agent.real_service.MAX_ATTEMPTS_PER_PROCESS", 1)

    async def run():
        service = _service(tmp_path)
        try:
            request = _request()
            ack = await service.DispatchTask(request, None)
            await _completed(service)
            assert await service.DispatchTask(request, None) == ack
            rejected = await service.DispatchTask(
                _request(attempt_id="attempt-2"), None
            )
            assert not rejected.accepted
            assert rejected.error_code == "attempt_history_full_restart_agent"
        finally:
            await service.close()

    asyncio.run(run())


def test_agent_consumes_verified_inputs_and_rejects_cross_workflow_artifacts(tmp_path):
    async def run():
        executor = ControlledExecutor()
        service = _service(tmp_path, executor)
        try:
            artifact, envelope, _ = _artifact(service.files)
            request = _request(
                "hil_mapping",
                inputs=(
                    workflow_pb2.InputArtifactBinding(
                        consumer_task_id="hil_mapping",
                        consumer_port="observations",
                        artifact=artifact,
                    ),
                ),
            )
            assert (await service.DispatchTask(request, None)).accepted
            assert (await _completed(service)).ok
            assert executor.calls[0][1] == {"observations": envelope["payload"]}
            foreign, _, _ = _artifact(service.files, workflow_id="different-workflow")
            request = _request(
                "hil_mapping",
                attempt_id="attempt-2",
                inputs=(
                    workflow_pb2.InputArtifactBinding(
                        consumer_task_id="hil_mapping",
                        consumer_port="observations",
                        artifact=foreign,
                    ),
                ),
            )
            assert (await service.DispatchTask(request, None)).accepted
            await asyncio.gather(*tuple(service._active.values()))
            failure = service._history[-1]
            assert not failure.ok and failure.error_code == "execution_failed"
            assert "different workflow" in failure.error_message
            assert len(executor.calls) == 1
        finally:
            await service.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "outputs,elapsed_ms",
    [
        ({}, 1.0),
        ({"observations": {}, "truth": {}, "extra": {}}, 1.0),
        ({"observations": [], "truth": {}}, 1.0),
        ({"observations": {}, "truth": {}}, -1.0),
        ({"observations": {}, "truth": {}}, float("nan")),
    ],
)
def test_agent_does_not_report_invalid_executor_result_as_success(
    tmp_path, outputs, elapsed_ms
):
    async def run():
        service = _service(
            tmp_path, ControlledExecutor(outputs=outputs, elapsed_ms=elapsed_ms)
        )
        try:
            assert (await service.DispatchTask(_request(), None)).accepted
            completion = await _completed(service)
            assert not completion.ok
            assert completion.error_code == "execution_failed"
            assert not completion.outputs
            assert math.isfinite(completion.compute_time_ms)
            assert completion.compute_time_ms >= 0
        finally:
            await service.close()

    asyncio.run(run())


def test_execution_timeout_cancels_underlying_business_and_reports_failure(tmp_path):
    async def run():
        executor = ControlledExecutor(block=True)
        service = _service(tmp_path, executor, task_timeout_seconds=0.02)
        try:
            assert (await service.DispatchTask(_request(), None)).accepted
            completion = await _completed(service)
            assert not completion.ok and completion.error_code == "execution_timeout"
            assert executor.cancelled
            assert not service._active
        finally:
            await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("wait_for_execution", [True, False])
def test_cancel_always_releases_admission_and_publishes_terminal_result(
    tmp_path, wait_for_execution
):
    async def run():
        executor = ControlledExecutor(block=True)
        service = _service(tmp_path, executor)
        try:
            request = _request()
            assert (await service.DispatchTask(request, None)).accepted
            if wait_for_execution:
                await executor.started.wait()
            cancelled = await service.CancelAttempt(
                runtime_pb2.CancelAttemptRequest(
                    attempt_id=request.attempt_id,
                    reason="test",
                    requested_at_ms=100,
                ),
                None,
            )
            assert cancelled.cancelled
            assert not service._active
            assert len(service._history) == 1
            assert service._history[0].error_code == "cancelled"
            assert not service._history[0].ok
            assert (
                await service.DispatchTask(request, None)
                == service._accepted[request.attempt_id][1]
            )
        finally:
            await service.close()

    asyncio.run(run())


def test_real_heartbeat_uses_host_measurements_not_config_snapshot(
    tmp_path, monkeypatch
):
    cpu_times = namedtuple("CpuTimes", "user system idle")
    samples = iter((cpu_times(0, 0, 0), cpu_times(1, 0, 3)))
    monkeypatch.setattr("agent.telemetry.psutil.cpu_times", lambda: next(samples))
    monkeypatch.setattr(
        "agent.telemetry.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=37.5, total=8_000_000_000, available=5_000_000_000
        ),
    )
    monkeypatch.setattr("agent.telemetry.psutil.cpu_count", lambda: 4)

    async def run():
        service = _service(tmp_path)
        response = await service.GetState(
            runtime_service_pb2.GetStateRequest(agent_id="robot_1"), None
        )
        snapshot = response.heartbeat.node_snapshot
        assert snapshot.cpu_utilization_ratio == 0.25
        assert snapshot.memory_utilization_ratio == 0.375
        assert snapshot.online
        assert response.heartbeat.sequence == 1
        assert response.heartbeat.sampled_at_ms >= 0
        second = await service.GetState(
            runtime_service_pb2.GetStateRequest(agent_id="robot_1"), None
        )
        assert second.heartbeat.sequence == 2
        assert service.node.cpu_capacity_units == 4
        assert service.node.memory_capacity_gb == 8
        identity = service.telemetry.identity()
        assert "energy" in identity["unavailable"]
        assert "cpu_utilization" in identity["measured"]
        await service.close()

    asyncio.run(run())


def test_execution_artifacts_preserve_before_and_after_host_observations(
    tmp_path, monkeypatch
):
    cpu_times = namedtuple("CpuTimes", "user system idle")
    measurements = {
        "clock": 0.0,
        "cpu_times": cpu_times(0, 0, 0),
        "memory_percent": 25.0,
        "available": 6_000_000_000,
    }
    monkeypatch.setattr(
        "agent.telemetry.psutil.cpu_times",
        lambda: measurements["cpu_times"],
    )
    monkeypatch.setattr("agent.telemetry.monotonic", lambda: measurements["clock"])
    monkeypatch.setattr(
        "agent.telemetry.psutil.virtual_memory",
        lambda: SimpleNamespace(
            percent=measurements["memory_percent"],
            total=8_000_000_000,
            available=measurements["available"],
        ),
    )

    class ChangingLoadExecutor(ControlledExecutor):
        async def execute(self, task_type, inputs, seed):
            measurements.update(
                clock=0.4,
                cpu_times=cpu_times(7, 0, 9),
                memory_percent=50.0,
                available=4_000_000_000,
            )
            return await super().execute(task_type, inputs, seed)

    async def run():
        service = _service(tmp_path, ChangingLoadExecutor())
        measurements.update(clock=0.2, cpu_times=cpu_times(1, 0, 7))
        try:
            assert (await service.DispatchTask(_request(), None)).accepted
            completion = await _completed(service)
            assert completion.ok
            observations = service.records[0]["host_observations"]
            before, after = observations["before"], observations["after"]
            assert before["scope"] == after["scope"] == "host"
            assert before["clock"] == after["clock"] == "agent_monotonic_elapsed"
            assert 0 <= before["sampled_at_ms"] <= after["sampled_at_ms"]
            assert (
                before["cpu_sample_window_ms"] == after["cpu_sample_window_ms"] == 200
            )
            assert before["cpu_utilization_ratio"] == 0.125
            assert after["cpu_utilization_ratio"] == 0.75
            assert before["memory_utilization_ratio"] == 0.25
            assert after["memory_utilization_ratio"] == 0.5
            assert (
                before["memory_total_bytes"]
                == after["memory_total_bytes"]
                == 8_000_000_000
            )
            assert before["memory_available_bytes"] == 6_000_000_000
            assert after["memory_available_bytes"] == 4_000_000_000
            for reference in completion.outputs:
                envelope = json.loads(service.files.read(reference.checksum))
                assert envelope["execution"]["host_observations"] == observations
        finally:
            await service.close()

    asyncio.run(run())


def test_host_cpu_samples_use_independent_cached_intervals_without_masking_saturation(
    monkeypatch,
):
    cpu_times = namedtuple("CpuTimes", "user system idle")
    state = {"clock": 0.0, "cpu": cpu_times(0, 0, 0)}
    monkeypatch.setattr("agent.telemetry.monotonic", lambda: state["clock"])
    monkeypatch.setattr("agent.telemetry.psutil.cpu_times", lambda: state["cpu"])
    first = HostTelemetry()
    with pytest.raises(RuntimeError, match="warmup"):
        first.observe()

    state.update(clock=0.05, cpu=cpu_times(1, 0, 4))
    second = HostTelemetry()
    state.update(clock=0.16, cpu=cpu_times(6, 0, 9))
    first_reading = first.observe()
    second_reading = second.observe()
    assert first_reading["cpu_utilization_ratio"] == 0.4
    assert second_reading["cpu_utilization_ratio"] == 0.5
    assert first_reading["cpu_sample_window_ms"] == 160
    assert second_reading["cpu_sample_window_ms"] == 110

    # A tiny interval with all-busy ticks must not replace a meaningful sample.
    state.update(clock=0.162, cpu=cpu_times(7, 0, 9))
    assert first.observe() == first_reading
    assert second.observe() == second_reading

    # OS counters can also remain unchanged beyond the nominal window.
    state.update(clock=0.27, cpu=cpu_times(6, 0, 9))
    assert first.observe() == first_reading
    assert second.observe() == second_reading

    # A full interval with genuine 100% activity remains 100%; never cap load.
    state.update(clock=0.30, cpu=cpu_times(21, 0, 9))
    saturated = first.observe()
    assert saturated["cpu_utilization_ratio"] == 1.0
    assert saturated["cpu_sample_window_start_ms"] == 160
    assert saturated["sampled_at_ms"] == 300
    assert saturated["cpu_sample_window_ms"] >= 100
    assert second.observe()["cpu_utilization_ratio"] == 1.0


def test_host_telemetry_warmup_waits_for_actual_counter_progress(monkeypatch):
    cpu_times = namedtuple("CpuTimes", "user system idle")
    state = {"clock": 0.0}
    monkeypatch.setattr("agent.telemetry.monotonic", lambda: state["clock"])
    monkeypatch.setattr(
        "agent.telemetry.psutil.cpu_times",
        lambda: cpu_times(1, 0, 3) if state["clock"] >= 0.2 else cpu_times(0, 0, 0),
    )

    async def advance_time(delay):
        state["clock"] += delay

    monkeypatch.setattr("agent.telemetry.asyncio.sleep", advance_time)

    async def run():
        telemetry = HostTelemetry()
        assert telemetry._cached is None
        await telemetry.warmup()
        reading = telemetry.observe()
        assert reading["cpu_utilization_ratio"] == 0.25
        assert reading["cpu_sample_window_ms"] == 200
        assert reading["sampled_at_ms"] == 200

    asyncio.run(run())


def test_linux_cpu_counters_do_not_double_count_guests_or_treat_iowait_as_busy(
    monkeypatch,
):
    linux_times = namedtuple(
        "LinuxCpuTimes",
        "user nice system idle iowait irq softirq steal guest guest_nice",
    )
    monkeypatch.setattr(
        "agent.telemetry.psutil.cpu_times",
        lambda: linux_times(10, 4, 3, 20, 5, 1, 2, 0, 2, 1),
    )
    # Guest time is already part of user/nice; Linux iowait belongs to idle.
    assert HostTelemetry._cpu_counters() == (45, 25)


def test_host_telemetry_warmup_fails_boundedly_if_counters_never_advance(monkeypatch):
    cpu_times = namedtuple("CpuTimes", "user system idle")
    state = {"clock": 0.0}
    monkeypatch.setattr("agent.telemetry.monotonic", lambda: state["clock"])
    monkeypatch.setattr("agent.telemetry.psutil.cpu_times", lambda: cpu_times(0, 0, 0))
    monkeypatch.setattr(HostTelemetry, "WARMUP_TIMEOUT_SECONDS", 0.25)

    async def advance_time(delay):
        state["clock"] += delay

    monkeypatch.setattr("agent.telemetry.asyncio.sleep", advance_time)

    async def run():
        telemetry = HostTelemetry()
        with pytest.raises(RuntimeError, match="did not advance during warmup"):
            await telemetry.warmup()
        assert telemetry._cached is None
        assert 0.25 <= state["clock"] <= 0.35

    asyncio.run(run())


def test_real_business_subprocess_round_trip_over_grpc(tmp_path):
    async def run():
        service = _service(tmp_path, NavigationExecutor())
        server, port = await start_execution_server(service)
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                stub = runtime_service_pb2_grpc.AgentRuntimeStub(channel)
                registration = await stub.RegisterAgent(
                    runtime_pb2.RegisterAgentRequest(
                        schema_version="mars.runtime.v1",
                        agent_id="robot_1",
                        agent_instance_id="test-client",
                    )
                )
                assert registration.accepted
                assert registration.registration_id.startswith("hardware:")
                ack = await stub.DispatchTask(_request())
                assert ack.accepted
                from interfaces.proto.mars.v1.runtime_service_pb2 import (
                    CompletionSubscription,
                )

                stream = stub.StreamCompletions(
                    CompletionSubscription(agent_id="robot_1")
                )
                completion = await asyncio.wait_for(stream.read(), 5)
                stream.cancel()
                assert completion.ok
                assert completion.compute_time_ms > 0
                assert {item.producer_port for item in completion.outputs} == {
                    "observations",
                    "truth",
                }
                observation = next(
                    item
                    for item in completion.outputs
                    if item.producer_port == "observations"
                )
                envelope = json.loads(service.files.read(observation.checksum))
                for reading in envelope["execution"]["host_observations"].values():
                    assert reading["scope"] == "host"
                    assert reading["cpu_sample_window_ms"] >= 100
                    assert 0 <= reading["cpu_utilization_ratio"] <= 1
                    assert 0 <= reading["memory_utilization_ratio"] <= 1
                    assert reading["memory_total_bytes"] > 0
                    assert (
                        0
                        <= reading["memory_available_bytes"]
                        <= reading["memory_total_bytes"]
                    )
                assert len(envelope["payload"]["scans"]) == 20
                assert (
                    envelope["payload"]["acquisition"]
                    == "synthetic_known_pose_2d_range_survey"
                )
        finally:
            await service.close()
            await server.stop(0)

    asyncio.run(run())


def test_navigation_executor_rejects_invalid_business_inputs():
    async def run():
        executor = NavigationExecutor()
        with pytest.raises(ValueError, match="business worker failed"):
            await executor.execute("hil_mapping", {"observations": {}}, 7)

    asyncio.run(run())


def test_navigation_executor_cancellation_reaps_actual_subprocess(monkeypatch):
    async def run():
        created = asyncio.Event()
        processes = []
        real_spawn = asyncio.create_subprocess_exec

        async def tracked_spawn(*args, **kwargs):
            process = await real_spawn(*args, **kwargs)
            processes.append(process)
            created.set()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", tracked_spawn)
        task = asyncio.create_task(NavigationExecutor().execute("hil_sensor", {}, 7))
        await asyncio.wait_for(created.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(processes) == 1
        assert processes[0].returncode is not None

    asyncio.run(run())
