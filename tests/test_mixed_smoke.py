"""Real localhost gRPC and CPU computation with explicitly fabricated CUDA evidence.

Native CUDA is unavailable on this Mac. These tests NEVER establish GPU or
physical two-host execution, and every fixture run must refuse hardware scope.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import math
import sys
from time import perf_counter

import pytest

from agent.artifacts import ArtifactFiles, canonical_json, digest_bytes
from agent.executor import ExecutionResult
from agent.real_service import ExecutionAgentService, start_execution_server
from agent.service import AgentConfig
from agent.telemetry import detected_node
from examples.hardware_workloads import pipeline as navigation
from examples.mixed_workloads.pipeline import PORT_TYPES, execute
from scripts import mixed_smoke
from scripts.mixed_smoke import EDGES, PLACEMENTS, TASKS, run_mixed_smoke


def _hash(value):
    return digest_bytes(canonical_json(value))


def fixture_measurement():
    return {
        "backend": "cuda_runtime",
        "device": "cuda:0",
        "input_device": "cuda:0",
        "output_device": "cuda:0",
        "device_name": "FAKE CUDA FIXTURE - no GPU execution",
        "timing_scope": "occupancy_inflation_kernel_only",
        "repeats": 1,
        "cuda_event_ms": [1.0],
        "synchronized_wall_ms": [2.0],
        "allocated_device_bytes": 65536,
        "cuda_runtime_version": 13020,
        "cuda_driver_version": 13020,
        "source_sha256": "a" * 64,
        "binary_sha256": "b" * 64,
        "test_fixture": True,
        "compute_capability": [8, 7],
    }


def fixture_gpu_info():
    measurement = fixture_measurement()
    return {
        **{
            key: measurement[key]
            for key in (
                "backend",
                "device",
                "device_name",
                "cuda_runtime_version",
                "cuda_driver_version",
                "source_sha256",
                "binary_sha256",
            )
        },
        "available": True,
        "kernel_execution_verified": True,
        "device_count": 1,
        "compute_capability": [8, 7],
        "test_fixture": True,
    }


def fixture_inflation(mapped):
    width, height = mapped["width_cells"], mapped["height_cells"]
    blocked = navigation._blocked(mapped)  # CPU fixture, deliberately NOT CUDA.
    return {
        "schema_version": 1,
        "kind": "hil.inflated_map.v1",
        "scene_id": mapped["scene_id"],
        "width_cells": width,
        "height_cells": height,
        "source_hashes": {"map": _hash(mapped)},
        "blocked": [
            int((x, y) in blocked) for y in range(height) for x in range(width)
        ],
        "measurement": fixture_measurement(),
        "test_fixture": True,
    }


class FixtureExecutor:
    def __init__(
        self, node_id, *, mutate=None, wrong_seed=False, unchecked_validation=False
    ):
        self.ports = {
            kind: PORT_TYPES[kind]
            for task, kind in TASKS.items()
            if PLACEMENTS[task] == node_id
        }
        self.gpu_demands = {TASKS["inflate"]: 1.0} if node_id == "robot_1" else {}
        self.mutate = mutate
        self.wrong_seed = wrong_seed
        self.unchecked_validation = unchecked_validation
        self.calls = []

    async def execute(self, task_type, inputs, seed):
        self.calls.append((task_type, seed))
        started = perf_counter()
        if task_type == TASKS["inflate"]:
            outputs = {"inflated": fixture_inflation(inputs["map"])}
        elif task_type == TASKS["validate"] and self.unchecked_validation:
            outputs = {
                "validation": {
                    "schema_version": 1,
                    "kind": "hil.mixed_validation.v1",
                    "scene_id": inputs["map"]["scene_id"],
                    "valid": True,
                    "source_hashes": {
                        name: _hash(value) for name, value in inputs.items()
                    },
                }
            }
        else:
            selected_seed = (
                seed + 1 if self.wrong_seed and task_type == TASKS["sense"] else seed
            )
            outputs = execute(task_type, inputs, selected_seed)
        if self.mutate:
            self.mutate(task_type, outputs)
        return ExecutionResult(outputs, (perf_counter() - started) * 1000)


class EvidenceMutationService(ExecutionAgentService):
    """Re-sign altered metadata so checksums alone cannot detect the mutation."""

    envelope_mutation = None

    async def _invoke(self, request, record):
        outputs = await super()._invoke(request, record)
        if self.envelope_mutation:
            for reference in outputs:
                envelope = json.loads(self.files.read(reference.checksum))
                self.envelope_mutation(envelope)
                data = canonical_json(envelope)
                digest = self.files.put(data)
                reference.checksum = digest
                reference.artifact_id = f"sha256:{digest}"
                reference.uri = f"mars-artifact://{self.config.agent_id}/{digest}"
                reference.size_mb = len(data) / 1_000_000
        return outputs


@asynccontextmanager
async def _agents(tmp_path, *, envelope_mutation=None, **executor_options):
    services, servers, endpoints = {}, [], {}
    try:
        for node_id in ("robot_1", "edge_pc"):
            orin = node_id == "robot_1"
            node = detected_node(
                "robot" if orin else "edge",
                gpu_info=fixture_gpu_info() if orin else None,
                capabilities=["hil_mixed_orin_v1" if orin else "hil_mixed_pc_v1"],
            )
            # Only scheduling capacity is fabricated; retain actual same-host
            # identity. No test claims this Mac is an Orin or a second PC.
            node["memory_gb"] = 64
            executor = FixtureExecutor(node_id, **executor_options)
            service = EvidenceMutationService(
                AgentConfig(node_id, "127.0.0.1:0", node, {}),
                executor,
                ArtifactFiles(tmp_path / node_id),
                {},
                task_timeout_seconds=10,
            )
            service.envelope_mutation = envelope_mutation
            original_identity = service.telemetry.identity
            service.telemetry.identity = lambda identity=original_identity: {
                **identity(),
                "test_fixture": True,
            }
            server, port = await start_execution_server(service)
            services[node_id] = service
            endpoints[node_id] = f"127.0.0.1:{port}"
            servers.append(server)
        for service in services.values():
            service.peers.update(endpoints)
        yield services, endpoints
    finally:
        await asyncio.gather(*(service.close() for service in services.values()))
        await asyncio.gather(*(server.stop(0) for server in servers))


async def _run(endpoints, tmp_path, **kwargs):
    options = dict(
        artifact_directory=tmp_path / "received",
        runs=1,
        workflow_timeout_seconds=15,
        task_completion_timeout_seconds=10,
        evidence_timeout_seconds=10,
        allow_same_host=True,
    )
    options.update(kwargs)
    return await run_mixed_smoke(endpoints, **options)


def test_exact_dag_ports_resources_and_configurable_deadlines():
    workflow = mixed_smoke.mixed_workflow("chosen-id", deadline_ms=900_000)
    assert workflow.workflow_id == "chosen-id"
    assert workflow.deadline_time_ms == 900_000
    assert len(workflow.tasks) == 5 and len(workflow.data_edges) == 8
    assert sum(len(task.spec.output_ports) for task in workflow.tasks) == 6
    for task in workflow.tasks:
        assert task.deadline_time_ms == task.spec.latency_budget_ms == 900_000
        assert (
            task.spec.placement_constraints.pinned_node_id == PLACEMENTS[task.task_id]
        )
        assert task.spec.placement_constraints.allow_fallback is False
        assert task.spec.gpu_demand == int(task.task_id == "inflate")
        assert set(task.dependency_task_ids) == {
            edge[0] for edge in EDGES if edge[2] == task.task_id
        }
        assert {
            port.name: port.message_type for port in task.spec.input_ports
        } == PORT_TYPES[task.spec.task_type]["inputs"]


def test_three_real_grpc_runs_are_explicit_transport_fixtures_not_hardware(tmp_path):
    async def run():
        async with _agents(tmp_path) as (services, endpoints):
            report = await _run(endpoints, tmp_path, runs=3)
            assert report["status"] == "succeeded", report["error"]
            assert report["scope"] == "development_transport_fixture"
            assert report["hardware_smoke_passed"] is report["gpu_tested"] is False
            assert [item["seed"] for item in report["runs"]] == [19, 20, 21]
            assert len({item["workflow_id"] for item in report["runs"]}) == 3
            assert not report["stopped_after_failure"]
            assert report["physical_actuation"] is False and report["energy_j"] is None
            for item in report["runs"]:
                assert len(item["artifacts"]) == 6 and len(item["executions"]) == 5
                assert len(item["edges"]) == 8
                assert sum(edge["remote_bytes"] > 0 for edge in item["edges"]) == 5
                assert item["remote_input_bytes"] == sum(
                    edge["remote_bytes"] for edge in item["edges"]
                )
                assert item["independent_validation_sha256"] == _hash(
                    item["validation"]
                )
                assert item["validation"]["gpu_full_reference_match"] is True
                assert item["checks"]["independent_validation"] is True
                assert item["checks"]["no_test_fixtures"] is False
                assert item["executing_host_count"] == 1
                assert item["hardware_smoke_passed"] is item["gpu_tested"] is False
                assert (
                    "FAKE CUDA FIXTURE"
                    in item["gpu_execution"]["measurement"]["device_name"]
                )
            assert len(services["robot_1"].executor.calls) == 9
            assert len(services["edge_pc"].executor.calls) == 6

    asyncio.run(run())


@pytest.mark.parametrize("require_jetpack721", [False, True])
def test_default_gate_rejects_localhost_fixture_and_stops_after_first_failure(
    tmp_path, require_jetpack721
):
    async def run():
        async with _agents(tmp_path) as (services, endpoints):
            report = await _run(
                endpoints,
                tmp_path,
                allow_same_host=False,
                runs=3,
                require_jetpack721=require_jetpack721,
            )
            assert report["status"] == "failed"
            assert report["hardware_smoke_passed"] is report["gpu_tested"] is False
            assert len(report["runs"]) == 1 and report["stopped_after_failure"]
            assert "hardware acceptance failed" in report["error"]
            assert "distinct_machine_ids" in report["runs"][0]["hardware_gate_failures"]
            assert "no_test_fixtures" in report["runs"][0]["hardware_gate_failures"]
            assert (
                "jetpack_profile" in report["runs"][0]["hardware_gate_failures"]
            ) == require_jetpack721
            assert services["robot_1"].executor.calls.count((TASKS["sense"], 19)) == 1

    asyncio.run(run())


def test_sequential_runs_stop_at_second_seed_failure(tmp_path):
    def mutate(task_type, outputs):
        if task_type == TASKS["sense"] and outputs["observations"]["seed"] == 20:
            raise ValueError("fixture failure on second seed")

    async def run():
        async with _agents(tmp_path, mutate=mutate) as (services, endpoints):
            report = await _run(endpoints, tmp_path, runs=3)
            assert report["status"] == "failed" and report["stopped_after_failure"]
            assert [item["seed"] for item in report["runs"]] == [19, 20]
            assert report["runs"][0]["status"] == "succeeded"
            assert report["runs"][1]["status"] == "failed"
            assert all(
                seed != 21
                for service in services.values()
                for _, seed in service.executor.calls
            )

    asyncio.run(run())


def _target_identity_fixture():
    # Pure predicate inputs only, never emitted as actual execution evidence.
    shared = {"runtime_source_sha256": "c" * 64, "git_revision": "d" * 40}
    return {
        "edge_pc": {**shared, "architecture": "x86_64", "machine_id_sha256": "1" * 64},
        "robot_1": {
            **shared,
            "architecture": "aarch64",
            "machine_id_sha256": "2" * 64,
            "jetson_model": "NVIDIA Jetson AGX Orin Developer Kit",
            "jetson_linux": "# R39 (release), REVISION: 2.1, GCID: fixture",
        },
    }


@pytest.mark.parametrize(
    "field,value,failed_check",
    [
        ("machine_id_sha256", "1" * 64, "distinct_machine_ids"),
        ("machine_id_sha256", None, "distinct_machine_ids"),
        ("runtime_source_sha256", "e" * 64, "matching_runtime_source"),
        ("git_revision", "e" * 40, "matching_git_revision"),
        ("architecture", "x86_64", "target_architectures"),
        ("jetson_model", "Jetson Orin Nano", "jetson_agx_orin"),
        ("jetson_linux", "# R39 (release), REVISION: 2.10,", "jetpack721"),
        ("jetson_linux", "# R36 (release), REVISION: 2.1,", "jetpack721"),
    ],
)
def test_hardware_identity_predicates_reject_target_mismatches(
    field, value, failed_check
):
    hosts = _target_identity_fixture()
    hosts["robot_1"][field] = value
    checks = mixed_smoke._hardware_checks(hosts, fixture_measurement(), fixture=True)
    assert checks[failed_check] is False
    assert checks["no_test_fixtures"] is False


def test_only_orin_compute_capability_matches_target():
    for capability, expected in (([8, 7], True), ([8, 6], False), ([9, 0], False)):
        measurement = {**fixture_measurement(), "compute_capability": capability}
        checks = mixed_smoke._hardware_checks(
            _target_identity_fixture(), measurement, fixture=True
        )
        assert checks["orin_compute_capability"] is expected
        assert checks["no_test_fixtures"] is False


@pytest.mark.parametrize(
    "runtime,driver,expected",
    [
        (13020, 13020, True),
        (13020, 13030, True),
        (13000, 13020, False),
        (12060, 13020, False),
    ],
)
def test_jetpack721_requires_cuda132_runtime_but_allows_newer_driver(
    runtime, driver, expected
):
    measurement = {
        **fixture_measurement(),
        "cuda_runtime_version": runtime,
        "cuda_driver_version": driver,
    }
    checks = mixed_smoke._hardware_checks(
        _target_identity_fixture(), measurement, fixture=True
    )
    assert checks["jetpack721"] is expected
    assert checks["no_test_fixtures"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "torch"),
        ("device", "cuda:1"),
        ("device_name", "different GPU"),
        ("cuda_runtime_version", 12060),
        ("cuda_driver_version", 12060),
        ("source_sha256", "f" * 64),
        ("binary_sha256", "f" * 64),
        ("compute_capability", [8, 6]),
        ("kernel_execution_verified", False),
        ("available", False),
        ("device_count", 0),
    ],
)
def test_task_cuda_measurements_must_match_executed_preflight(tmp_path, field, value):
    def mutate(envelope):
        record = envelope["execution"]
        if record["agent_id"] == "robot_1":
            record["host"]["cuda_device"][field] = value

    async def run():
        async with _agents(tmp_path, envelope_mutation=mutate) as (_, endpoints):
            report = await _run(endpoints, tmp_path)
            assert report["status"] == "failed" and "preflight" in report["error"]
            assert not report["hardware_smoke_passed"] and not report["gpu_tested"]
            assert (
                report["runs"][0]["coordinator_report"]["workflow"]["state"]
                == "succeeded"
            )

    asyncio.run(run())


def test_source_seed_must_match_requested_run_even_when_pipeline_is_valid(tmp_path):
    async def run():
        async with _agents(tmp_path, wrong_seed=True) as (_, endpoints):
            report = await _run(endpoints, tmp_path)
            assert report["status"] == "failed"
            assert "run's seed" in report["error"]
            assert (
                report["runs"][0]["coordinator_report"]["workflow"]["state"]
                == "succeeded"
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("placement", "placement/identity"),
        ("mode", "placement/identity"),
        ("input_hash", "reference/remote bytes"),
        ("remote_bytes", "reference/remote bytes"),
        ("missing_edge", "every DAG input"),
        ("duplicate_edge", "duplicate or incorrect input ports"),
        ("host", "inconsistent host identity"),
        ("two_sensor_records", "output ports disagree"),
    ],
)
def test_resigned_execution_evidence_cannot_bypass_graph_checks(
    tmp_path, mutation, message
):
    def mutate(envelope):
        record = envelope["execution"]
        if mutation == "two_sensor_records":
            if envelope["producer_port"] == "truth":
                record["worker_elapsed_ms"] += 1
            return
        if record["task_id"] != "validate":
            return
        if mutation == "placement":
            record["agent_id"] = "edge_pc"
        elif mutation == "mode":
            record["execution_mode"] = "real_cuda"
        elif mutation == "input_hash":
            record["input_artifacts"][0]["sha256"] = "f" * 64
        elif mutation == "remote_bytes":
            record["input_artifacts"][0]["remote_bytes"] = 0
        elif mutation == "missing_edge":
            record["input_artifacts"].pop()
        elif mutation == "duplicate_edge":
            record["input_artifacts"][1] = deepcopy(record["input_artifacts"][0])
        elif mutation == "host":
            record["host"]["hostname"] += "-different"

    async def run():
        async with _agents(tmp_path, envelope_mutation=mutate) as (_, endpoints):
            report = await _run(endpoints, tmp_path)
            assert report["status"] == "failed"
            assert message in report["error"]
            assert report["hardware_smoke_passed"] is False

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("mapping_lineage", "mapping did not consume"),
        ("validation_metric", "independent CPU validation fields/digests"),
        ("validation_hash", "independent CPU validation fields/digests"),
        ("missing_binary_hash", "source/binary SHA256"),
        ("different_binary", "preflight binary_sha256"),
    ],
)
def test_runner_independently_checks_payload_lineage_and_validation(
    tmp_path, mutation, message
):
    def mutate(task_type, outputs):
        if task_type == TASKS["map"] and mutation == "mapping_lineage":
            outputs["map"]["source_hashes"]["observations"] = "e" * 64
        if task_type == TASKS["validate"]:
            if mutation == "validation_metric":
                outputs["validation"]["path_length_m"] += 1
            if mutation == "validation_hash":
                outputs["validation"]["source_hashes"]["inflated"] = "e" * 64
        if task_type == TASKS["inflate"]:
            if mutation == "missing_binary_hash":
                outputs["inflated"]["measurement"].pop("binary_sha256")
            if mutation == "different_binary":
                outputs["inflated"]["measurement"]["binary_sha256"] = "f" * 64

    async def run():
        async with _agents(tmp_path, mutate=mutate) as (_, endpoints):
            report = await _run(endpoints, tmp_path)
            assert report["status"] == "failed"
            assert message in report["error"]
            assert (
                report["runs"][0]["coordinator_report"]["workflow"]["state"]
                == "succeeded"
            )

    asyncio.run(run())


@pytest.mark.parametrize("delta,accepted", [(1e-10, True), (1e-5, False)])
def test_cross_architecture_validation_rounding_preserves_exact_hashes(
    tmp_path, delta, accepted
):
    original = {}

    def mutate(task_type, outputs):
        if task_type == TASKS["validate"]:
            original.update(deepcopy(outputs["validation"]))
            outputs["validation"]["path_length_m"] += delta

    async def run():
        async with _agents(tmp_path, mutate=mutate) as (_, endpoints):
            report = await _run(endpoints, tmp_path)
            result = report["runs"][0]
            assert (report["status"] == "succeeded") is accepted
            assert (
                report["validation_float_tolerance"]
                == result["validation_float_tolerance"]
            )
            assert result["validation_float_tolerance"]["rel_tol"] == 1e-9
            assert result["validation_float_tolerance"]["abs_tol"] == 1e-9
            assert not report["hardware_smoke_passed"] and not report["gpu_tested"]
            if accepted:
                assert result["checks"]["independent_validation"] is True
                assert (
                    result["validation"]["source_hashes"] == original["source_hashes"]
                )
                assert result["independent_validation_sha256"] == _hash(original)
                assert result["payload_sha256"]["validation"] == _hash(
                    result["validation"]
                )
                assert (
                    result["independent_validation_sha256"]
                    != result["payload_sha256"]["validation"]
                )
            else:
                assert "independent CPU validation fields/digests" in report["error"]

    asyncio.run(run())


@pytest.mark.parametrize("field", mixed_smoke.VALIDATION_FLOAT_TOLERANCE["fields"])
def test_only_finite_float_metrics_receive_validation_tolerance(field):
    expected = {field: 2.0}
    assert mixed_smoke._validation_matches(
        expected, {field: math.nextafter(2.0, math.inf)}
    )
    for value in (2, True, "2.0", 2.00001, float("nan"), float("inf")):
        assert not mixed_smoke._validation_matches(expected, {field: value})
    assert not mixed_smoke._validation_matches({field: 2}, {field: 2.0})
    assert not mixed_smoke._validation_matches(
        {field: float("inf")}, {field: float("inf")}
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"valid": 1},
        {"schema_version": True},
        {"gpu_cells_checked": 6144.0},
        {"checked_segments": 3},
        {"robot_radius_m": math.nextafter(0.18, math.inf)},
        {"source_hashes": {"map": "b" * 64}},
        {"checks": ["second", "first"]},
        {"checks": ("first", "second")},
        {"unexpected": 1},
    ],
)
def test_validation_keys_types_counts_inputs_hashes_and_checklists_remain_exact(
    changes,
):
    expected = {
        "valid": True,
        "schema_version": 1,
        "gpu_cells_checked": 6144,
        "checked_segments": 2,
        "robot_radius_m": 0.18,
        "source_hashes": {"map": "a" * 64},
        "checks": ["first", "second"],
    }
    assert not mixed_smoke._validation_matches(expected, {**expected, **changes})
    missing = dict(expected)
    missing.pop("valid")
    assert not mixed_smoke._validation_matches(expected, missing)


def test_independent_cpu_reference_rejects_off_path_bit_despite_forged_valid_flag(
    tmp_path,
):
    def mutate(task_type, outputs):
        if task_type == TASKS["inflate"]:
            outputs["inflated"]["blocked"][-1] ^= 1

    async def run():
        async with _agents(tmp_path, mutate=mutate, unchecked_validation=True) as (
            _,
            endpoints,
        ):
            report = await _run(endpoints, tmp_path)
            assert report["status"] == "failed"
            assert "independent full CPU reference" in report["error"]
            assert (
                report["runs"][0]["coordinator_report"]["workflow"]["state"]
                == "succeeded"
            )
            assert len(report["runs"][0]["artifacts"]) == 6

    asyncio.run(run())


def test_failed_business_transfer_retains_available_artifacts(tmp_path):
    async def run():
        async with _agents(tmp_path) as (services, endpoints):
            services["edge_pc"].peers.pop("robot_1")
            report = await _run(endpoints, tmp_path)
            assert report["status"] == "failed" and "map: failed" in report["error"]
            assert len(report["runs"][0]["artifacts"]) == 2
            assert report["runs"][0]["final_node_observations"]

    asyncio.run(run())


@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
def test_artifact_collection_deadline_and_cancellation_close_runtime(
    tmp_path, monkeypatch, interrupt
):
    async def run():
        entered, cancelled = asyncio.Event(), asyncio.Event()
        calls, closed = [], []
        original_fetch = mixed_smoke.fetch_artifact
        original_close = mixed_smoke.GrpcRuntimeAdapter.close

        async def stalled_fetch(*args, **kwargs):
            if calls:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            calls.append(1)
            return await original_fetch(*args, **kwargs)

        async def tracked_close(runtime):
            await original_close(runtime)
            closed.append(True)

        monkeypatch.setattr(mixed_smoke, "fetch_artifact", stalled_fetch)
        monkeypatch.setattr(mixed_smoke.GrpcRuntimeAdapter, "close", tracked_close)
        async with _agents(tmp_path) as (services, endpoints):
            task = asyncio.create_task(
                _run(
                    endpoints,
                    tmp_path,
                    runs=3,
                    evidence_timeout_seconds=0.1 if interrupt == "timeout" else 10,
                )
            )
            await asyncio.wait_for(entered.wait(), 10)
            if interrupt == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                report = await task
                assert (
                    report["status"] == "failed" and "evidence phase" in report["error"]
                )
                assert "TimeoutError" in report["error"]
                assert len(report["runs"]) == 1
                assert len(report["runs"][0]["artifacts"]) == 1
                assert not report["hardware_smoke_passed"]
            assert cancelled.is_set() and closed
            assert all(not service._active for service in services.values())

    asyncio.run(run())


@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
def test_evidence_reference_subprocess_is_reaped_on_timeout_or_cancel(
    tmp_path, monkeypatch, interrupt
):
    async def run():
        entered = asyncio.Event()
        children = []
        spawn = asyncio.create_subprocess_exec

        async def stalled_reference(*args, **kwargs):
            process = await spawn(
                sys.executable, "-c", "import time; time.sleep(60)", **kwargs
            )
            children.append(process)
            entered.set()
            return process

        # Fixture Agent work executes in-process. The only asynchronous child
        # is the runner's independent CPU validation subprocess.
        monkeypatch.setattr(asyncio, "create_subprocess_exec", stalled_reference)
        async with _agents(tmp_path) as (_, endpoints):
            task = asyncio.create_task(
                _run(
                    endpoints,
                    tmp_path,
                    evidence_timeout_seconds=0.3 if interrupt == "timeout" else 10,
                )
            )
            await asyncio.wait_for(entered.wait(), 10)
            if interrupt == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                report = await task
                assert (
                    report["status"] == "failed" and "evidence phase" in report["error"]
                )
                assert len(report["runs"][0]["artifacts"]) == 6
            assert len(children) == 1 and children[0].returncode is not None

    asyncio.run(run())


def test_workflow_timeout_cancels_active_agent_attempt(tmp_path):
    async def run():
        cancelled = asyncio.Event()
        async with _agents(tmp_path) as (services, endpoints):

            async def stalled(*args):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

            services["robot_1"].executor.execute = stalled
            report = await _run(
                endpoints,
                tmp_path,
                workflow_timeout_seconds=1,
                task_completion_timeout_seconds=1,
            )
            assert report["status"] == "failed" and "workflow phase" in report["error"]
            assert cancelled.is_set()
            assert all(not service._active for service in services.values())

    asyncio.run(run())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"runs": 0},
        {"runs": True},
        {"runs": mixed_smoke.MAX_RUNS + 1},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32 - 1, "runs": 2},
        {"workflow_timeout_seconds": float("nan")},
        {"evidence_timeout_seconds": 0},
        {"task_completion_timeout_seconds": 181},
    ],
)
def test_invalid_configuration_rejected_before_runtime_or_artifact_creation(
    tmp_path, kwargs
):
    with pytest.raises(ValueError):
        asyncio.run(
            run_mixed_smoke(
                {"robot_1": "127.0.0.1:1", "edge_pc": "127.0.0.1:2"},
                artifact_directory=tmp_path / "never-created",
                **kwargs,
            )
        )
    assert not (tmp_path / "never-created").exists()


def test_cli_defaults_and_failure_report_are_exclusive(tmp_path, monkeypatch, capsys):
    output = tmp_path / "report.json"
    arguments = [
        "mixed_smoke",
        "--agent",
        "robot_1=127.0.0.1:50051",
        "--agent",
        "edge_pc=127.0.0.1:50052",
        "--output",
        str(output),
    ]
    captured = {}

    async def failed_run(endpoints, **kwargs):
        captured.update(kwargs)
        return {
            "status": "failed",
            "scope": "unverified",
            "hardware_smoke_passed": False,
            "gpu_tested": False,
            "error": "fixture connection failure",
            "runs": [],
        }

    monkeypatch.setattr(mixed_smoke, "run_mixed_smoke", failed_run)
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as exited:
        mixed_smoke.main()
    assert exited.value.code == 1
    assert json.loads(output.read_text())["status"] == "failed"
    assert captured["runs"] == 3 and captured["seed"] == 19
    assert (
        captured["workflow_timeout_seconds"],
        captured["task_completion_timeout_seconds"],
        captured["evidence_timeout_seconds"],
    ) == (180, 120, 60)
    assert captured["allow_same_host"] is False
    before = output.read_bytes()
    captured.clear()
    with pytest.raises(SystemExit) as exited:
        mixed_smoke.main()
    assert exited.value.code == 2 and not captured
    assert output.read_bytes() == before
    assert "output already exists" in capsys.readouterr().err
