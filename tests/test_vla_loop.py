"""Runner contracts and real gRPC transport with explicitly fake GPU outputs.

These tests need no CUDA device and do NOT constitute GPU or model execution.
The GPU executor/measurements/images below are controlled test fixtures; CPU
validation and artifact transfer use the actual implementation.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from pathlib import Path
import struct
import sys
import zlib

import pytest

from agent.artifacts import ArtifactFiles, canonical_json, digest_bytes
from agent.executor import ExecutionResult
from agent.real_service import ExecutionAgentService, start_execution_server
from agent.service import AgentConfig
from agent.telemetry import detected_node
from examples.vla_workloads import PORT_TYPES, execute
from examples.vla_workloads.bundle import POLICY_REVISION, VLM_ID, VLM_REVISION
from mars.domain.task import ResourceClass
from scripts import vla_loop
from scripts.vla_loop import (
    EDGES,
    GPU_TASK_TYPES,
    MODEL_ID,
    TASKS,
    build_parser,
    initial_profiles,
    run_vla_loop,
    verify_gpu_payload,
    vla_workflow,
)


GPU_INFO = {
    "available": True,
    "device": "cuda:0",
    "device_count": 1,
    "device_name": "FAKE GPU FIXTURE — not measured hardware",
    "compute_capability": [8, 7],
    "torch_version": "test-fixture",
}


def _hash(value):
    return digest_bytes(canonical_json(value))


def _measurement(*, vla=False):
    return {
        "device": "cuda:0",
        "device_name": GPU_INFO["device_name"],
        "torch_version": "test-fixture",
        "cuda_version": "test-fixture",
        "cuda_event_ms": [1.0],
        "synchronized_wall_ms": [2.0],
        "peak_memory_allocated_bytes": 4096,
        "memory_allocated_bytes": 1024,
        "warmup": 0,
        "repeats": 1,
        "input_devices": {
            "observation.state": "cuda:0",
            "observation.images.front": "cuda:0",
            "observation.language.tokens": "cuda:0",
            "observation.language.attention_mask": "cuda:0",
        }
        if vla
        else {"first": "cuda:0", "second": "cuda:0"},
        "output_device": "cuda:0",
        "parameter_devices": ["cuda:0"] if vla else [],
        "timing_scope": "policy_predict_action_chunk_only"
        if vla
        else "torch_matmul_only",
    }


def _observation():
    def chunk(name, data):
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )
    return {
        "schema": "mars.vla.observation.v1",
        "task": "fixture instruction",
        "state": [0.0, 1.0],
        "images": {
            "observation.images.front": {
                "encoding": "png",
                "data_base64": base64.b64encode(png).decode(),
            }
        },
        "provenance": {"test_fixture": True, "real_sensor": False},
    }


def _actions(observation, seed=19):
    chunk = [[0.1, 0.2], [0.3, 0.4]]
    return {
        "schema": "mars.vla.actions.v1",
        "source_hashes": {"observation": _hash(observation)},
        "task": observation["task"],
        "state_dim": 2,
        "image_keys": sorted(observation["images"]),
        "missing_camera_keys": [],
        "action_chunk": chunk,
        "action_sha256": _hash(chunk),
        "shape": [2, 2],
        "model": {
            "policy": {"repo_id": MODEL_ID, "revision": POLICY_REVISION},
            "vlm": {"repo_id": VLM_ID, "revision": VLM_REVISION},
            "manifest_sha256": "c" * 64,
            "state_dim": 2,
            "action_dim": 2,
            "chunk_size": 2,
            "image_keys": sorted(observation["images"]),
            "weights_verified": True,
            "strict_load": True,
            "test_fixture": True,
        },
        "inference": {"seed": seed, "num_steps": 20},
        "measurement": _measurement(vla=True),
        "robot_control_executed": False,
    }


def _smoke(seed=19):
    size = 64
    indexes = [0, 1, size // 2, size - 1]
    samples = [
        {
            "row": row,
            "column": column,
            "value": float(size * ((row + seed) % 17 - 8) * ((column + seed) % 13 - 6)),
        }
        for row in indexes
        for column in indexes
    ]
    return {
        "schema": "mars.cuda.result.v1",
        "operation": "float32_dense_matrix_multiply",
        "matrix_size": size,
        "seed": seed,
        "dtype": "float32",
        "max_abs_error": 0,
        "full_matrix_reference_match": True,
        "samples": samples,
        "sample_sha256": _hash(samples),
        "measurement": _measurement(),
        "test_fixture": True,
    }


class FixtureExecutor:
    """The GPU responses are fabricated solely to test transport contracts."""

    def __init__(self, gpu, *, tamper=None, unchecked_validation=False):
        self.ports = {
            key: value
            for key, value in PORT_TYPES.items()
            if (key in GPU_TASK_TYPES) == gpu
        }
        self.gpu_demands = {key: 1.0 for key in self.ports if key in GPU_TASK_TYPES}
        self.tamper = tamper
        self.unchecked_validation = unchecked_validation

    async def execute(self, task_type, inputs, seed):
        if task_type == "hil_vla_observe":
            outputs = {"observation": _observation()}
        elif task_type in GPU_TASK_TYPES:
            name = "actions" if task_type == "hil_vla_infer" else "cuda_result"
            payload = (
                _actions(inputs["observation"], seed)
                if name == "actions"
                else _smoke(seed)
            )
            if self.tamper:
                self.tamper(payload)
            outputs = {name: payload}
        elif self.unchecked_validation:
            outputs = {
                "validation": {
                    "valid": True,
                    "source_hashes": {
                        name: _hash(value) for name, value in inputs.items()
                    },
                }
            }
        else:
            outputs = execute(task_type, inputs, seed)
        return ExecutionResult(outputs, 1.0)


@asynccontextmanager
async def _agents(tmp_path: Path, gpu_agent="robot_1", **fixture_options):
    services, servers, endpoints = {}, [], {}
    try:
        for node_id in ("robot_1", "edge_pc"):
            gpu = node_id == gpu_agent
            node = detected_node(
                "robot" if node_id == "robot_1" else "edge",
                gpu_info=GPU_INFO if gpu else None,
                capabilities=["hil_cuda_v1", "hil_smolvla_v1"]
                if gpu
                else ["hil_vla_io_v1"],
                supported_models=[MODEL_ID] if gpu else [],
            )
            # Simulated capacity keeps a GPU-free CI host from setting scheduling
            # eligibility; its live memory fraction is still real host telemetry.
            node["memory_gb"] = 64
            service = ExecutionAgentService(
                AgentConfig(node_id, "127.0.0.1:0", node, {}),
                FixtureExecutor(gpu, **fixture_options),
                ArtifactFiles(tmp_path / node_id),
                {},
                task_timeout_seconds=5,
            )
            server, port = await start_execution_server(service)
            services[node_id], endpoints[node_id] = service, f"127.0.0.1:{port}"
            servers.append(server)
        for service in services.values():
            service.peers.update(endpoints)
        yield services, endpoints
    finally:
        await asyncio.gather(*(service.close() for service in services.values()))
        await asyncio.gather(*(server.stop(0) for server in servers))


@pytest.mark.parametrize("workload", ["cuda", "smolvla"])
@pytest.mark.parametrize("gpu_agent", ["robot_1", "edge_pc"])
def test_dag_has_typed_gpu_and_io_placements_and_explicit_reservations(
    workload, gpu_agent
):
    workflow = vla_workflow(workload, gpu_agent, "specific-id", deadline_ms=1_800_000)
    io_agent = "edge_pc" if gpu_agent == "robot_1" else "robot_1"
    tasks = {task.task_id: task for task in workflow.tasks}
    assert set(tasks) == set(TASKS[workload])
    assert workflow.deadline_time_ms == 1_800_000
    for task in tasks.values():
        gpu = task.spec.task_type in GPU_TASK_TYPES
        assert task.workflow_id == "specific-id"
        assert task.deadline_time_ms == task.spec.latency_budget_ms == 1_800_000
        assert task.spec.placement_constraints.pinned_node_id == (
            gpu_agent if gpu else io_agent
        )
        assert task.spec.gpu_demand == int(gpu)
        assert task.spec.dominant_resource == (
            ResourceClass.GPU if gpu else ResourceClass.CPU
        )
        assert task.spec.placement_constraints.allow_fallback is False
        assert task.spec.placement_constraints.allow_other_robots is True
        if task.spec.task_type == "hil_vla_infer":
            assert task.spec.model_requirement == MODEL_ID
        else:
            assert task.spec.model_requirement == ""
        profile = initial_profiles(workload).lookup(
            task.spec.task_type, task.spec.placement_constraints.allowed_node_kinds[0]
        )
        assert profile.gpu_units == task.spec.gpu_demand
        assert profile.cpu_units == 1
        assert profile.provenance.startswith("unmeasured_")
    for edge in workflow.data_edges:
        assert (
            edge.producer_task,
            edge.producer_port,
            edge.consumer_task,
            edge.consumer_port,
        ) in EDGES[workload]
        assert (
            edge.message_type
            == PORT_TYPES[tasks[edge.producer_task].spec.task_type]["outputs"][
                edge.producer_port
            ]
        )
        assert (
            edge.message_type
            == PORT_TYPES[tasks[edge.consumer_task].spec.task_type]["inputs"][
                edge.consumer_port
            ]
        )
    if workload == "smolvla":
        assert tasks["observe"].dependency_task_ids == ()
        assert tasks["infer"].dependency_task_ids == ("observe",)
        assert set(tasks["validate"].dependency_task_ids) == {"observe", "infer"}


@pytest.mark.parametrize(
    "workload,gpu_agent",
    [("smolvla", "robot_1"), ("smolvla", "edge_pc"), ("cuda", "robot_1")],
)
def test_fixture_gpu_contract_transfers_real_bytes_without_testing_hardware(
    tmp_path, workload, gpu_agent
):
    async def run():
        async with _agents(tmp_path, gpu_agent) as (_, endpoints):
            report = await run_vla_loop(
                endpoints,
                workload=workload,
                gpu_agent=gpu_agent,
                artifact_directory=tmp_path / "received",
                workflow_timeout_seconds=15,
                task_completion_timeout_seconds=5,
            )
            assert report["status"] == "succeeded", report["error"]
            assert report["scope"] == "same_host_cuda_execution"
            assert report["executing_host_count"] == 1
            assert report["physical_actuation"] is False
            assert report["control_success_tested"] is False
            assert report["energy_j"] is None
            assert report["gpu_execution"]["agent_id"] == gpu_agent
            assert (
                "FAKE GPU FIXTURE"
                in report["gpu_execution"]["measurement"]["device_name"]
            )
            records = {record["task_id"]: record for record in report["executions"]}
            assert records["validate"]["remote_input_bytes"] > 0
            if workload == "smolvla":
                assert records["infer"]["remote_input_bytes"] > 0
                assert report["gpu_execution"]["action_shape"] == [2, 2]
            assert len(report["final_node_observations"]) == 2

    asyncio.run(run())


def test_runner_rejects_cpu_output_even_if_fixture_validator_says_valid(tmp_path):
    async def run():
        def tamper(payload):
            payload["measurement"]["output_device"] = "cpu"

        async with _agents(tmp_path, tamper=tamper, unchecked_validation=True) as (
            _,
            endpoints,
        ):
            report = await run_vla_loop(
                endpoints,
                workload="cuda",
                artifact_directory=tmp_path / "received",
                workflow_timeout_seconds=15,
                task_completion_timeout_seconds=5,
            )
            assert report["coordinator_report"]["workflow"]["state"] == "succeeded"
            assert report["status"] == "failed"
            assert report["gpu_tested"] is False
            assert "output tensor" in report["error"]

    asyncio.run(run())


def test_physical_host_guard_keeps_fixture_execution_from_two_host_claim(tmp_path):
    async def run():
        async with _agents(tmp_path) as (_, endpoints):
            report = await run_vla_loop(
                endpoints,
                workload="cuda",
                artifact_directory=tmp_path / "received",
                workflow_timeout_seconds=15,
                task_completion_timeout_seconds=5,
                require_distinct_hosts=True,
            )
            assert report["status"] == "failed"
            assert report["executing_host_count"] == 1
            assert "same reported host" in report["error"]

    asyncio.run(run())


def test_failed_transfer_retains_successful_artifact_and_final_observations(tmp_path):
    async def run():
        async with _agents(tmp_path) as (services, endpoints):
            services["edge_pc"].peers.pop("robot_1")
            report = await run_vla_loop(
                endpoints,
                workload="cuda",
                artifact_directory=tmp_path / "received",
                workflow_timeout_seconds=15,
                task_completion_timeout_seconds=5,
            )
            assert report["status"] == "failed"
            assert "validate" in report["error"]
            assert len(report["artifacts"]) == 1
            assert len(report["final_node_observations"]) == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation",
    [
        "no_measurement",
        "zero_time",
        "cpu_parameters",
        "wrong_action_shape",
        "wrong_observation",
        "unverified_weights",
    ],
)
def test_vla_proof_rejects_missing_or_inconsistent_measurements(mutation):
    observation = _observation()
    payload = _actions(observation)
    if mutation == "no_measurement":
        payload.pop("measurement")
    elif mutation == "zero_time":
        payload["measurement"]["cuda_event_ms"] = [0.0]
    elif mutation == "cpu_parameters":
        payload["measurement"]["parameter_devices"] = ["cpu"]
    elif mutation == "wrong_action_shape":
        payload["shape"] = [1, 2, 2]
    elif mutation == "wrong_observation":
        observation["state"][0] = 99
    else:
        payload["model"]["weights_verified"] = False
    with pytest.raises(ValueError):
        verify_gpu_payload(payload, "smolvla", observation)


def test_cuda_proof_recomputes_output_samples():
    payload = _smoke()
    payload["samples"][0]["value"] += 1
    payload["sample_sha256"] = _hash(payload["samples"])
    with pytest.raises(ValueError, match="independent exact reference"):
        verify_gpu_payload(payload, "cuda")


def test_configurable_completion_timeout_is_not_capped_at_sixty_seconds(
    tmp_path, monkeypatch
):
    captured = {}

    class UnavailableRuntime:
        snapshots = ()

        def __init__(self, endpoints, **kwargs):
            captured.update(kwargs)

        async def start(self, now):
            raise RuntimeError("fixture stops before networking")

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr(vla_loop, "GrpcRuntimeAdapter", UnavailableRuntime)
    report = asyncio.run(
        run_vla_loop(
            {"robot_1": "localhost:1", "edge_pc": "localhost:2"},
            artifact_directory=tmp_path,
            workflow_timeout_seconds=1800,
            task_completion_timeout_seconds=1200,
        )
    )
    assert captured == {"completion_timeout_seconds": 1200, "closed": True}
    assert report["status"] == "failed"
    assert report["final_node_observations"] == []


@pytest.mark.parametrize(
    "options",
    [
        {"workflow_timeout_seconds": float("nan")},
        {"task_completion_timeout_seconds": 0},
        {"workflow_timeout_seconds": 60, "task_completion_timeout_seconds": 61},
        {"seed": -1},
        {"gpu_agent": "unknown"},
        {"workload": "fake"},
        {"endpoints": {"robot_1": "localhost:1"}},
    ],
)
def test_invalid_options_fail_before_connecting(tmp_path, options):
    kwargs = {
        "endpoints": {"robot_1": "localhost:1", "edge_pc": "localhost:2"},
        "artifact_directory": tmp_path,
    }
    kwargs.update(options)
    with pytest.raises(ValueError):
        asyncio.run(run_vla_loop(**kwargs))


def test_parser_defaults_and_existing_report_is_never_overwritten(
    tmp_path, monkeypatch
):
    output = tmp_path / "existing.json"
    args = [
        "--agent",
        "robot_1=localhost:1",
        "--agent",
        "edge_pc=localhost:2",
        "--output",
        str(output),
    ]
    parsed = build_parser().parse_args(args)
    assert parsed.workload == "smolvla" and parsed.gpu_agent == "robot_1"
    assert parsed.workflow_timeout == 600 and parsed.task_completion_timeout == 300
    output.write_bytes(b"existing evidence")
    monkeypatch.setattr(sys, "argv", ["vla_loop", *args])
    with pytest.raises(SystemExit) as result:
        vla_loop.main()
    assert result.value.code == 2
    assert output.read_bytes() == b"existing evidence"
