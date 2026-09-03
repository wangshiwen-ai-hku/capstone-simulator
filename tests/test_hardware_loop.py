"""The real business DAG over gRPC, including the documented process entrypoints.

All computations below use NavigationExecutor's actual worker processes. Loopback
is intentionally reported as same-host execution, never a physical Jetson test.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from agent.artifacts import ArtifactFiles, canonical_json, digest_bytes
from agent.executor import NavigationExecutor
from agent.real_service import ExecutionAgentService, start_execution_server
from agent.service import AgentConfig, load_agent_configs, start_agent_server
from agent.telemetry import detected_node
from examples.hardware_workloads import PORT_TYPES, execute
from scripts.hardware_loop import EDGES, TASKS, navigation_workflow, run_hardware_loop


REPO = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def _agents(directory: Path, node_ids: tuple[str, ...] = ("robot_1", "edge_pc")):
    services, servers, endpoints = {}, [], {}
    snapshots = []
    try:
        for node_id in node_ids:
            service = ExecutionAgentService(
                AgentConfig(
                    node_id,
                    "127.0.0.1:0",
                    detected_node("robot" if node_id == "robot_1" else "edge"),
                    {},
                ),
                NavigationExecutor(),
                ArtifactFiles(directory / node_id),
                {},
                task_timeout_seconds=5,
            )

            def capture_sample(*args, sample=service.telemetry.sample):
                measured = sample(*args)
                snapshots.append(
                    {
                        "node_id": measured.node_id,
                        "sequence": measured.snapshot_sequence,
                        "cpu": measured.cpu_utilization_ratio,
                        "memory": measured.memory_utilization_ratio,
                        "sampled_at_ms": measured.sampled_at_ms,
                    }
                )
                return measured

            # Preserve real readings while retaining timing/load failure evidence.
            service.telemetry.sample = capture_sample
            server, port = await start_execution_server(service)
            services[node_id] = service
            servers.append(server)
            endpoints[node_id] = f"127.0.0.1:{port}"
        for service in services.values():
            service.peers.update(endpoints)
        yield services, endpoints
    finally:
        (directory / "execution-records.json").write_bytes(
            canonical_json(
                {node: service.records for node, service in services.items()}
            )
        )
        (directory / "telemetry-samples.json").write_bytes(canonical_json(snapshots))
        await asyncio.gather(*(service.close() for service in services.values()))
        await asyncio.gather(*(server.stop(0) for server in servers))


def _payloads(evidence: dict) -> dict[str, dict]:
    return {
        item["reference"]["producer_port"]: item["envelope"]["payload"]
        for item in evidence["artifacts"]
    }


def _assert_actual_pipeline(evidence: dict, received: Path, seed: int) -> None:
    # Keep the complete failure report under pytest's retained temp directory;
    # async service shutdown must not erase the evidence of rare transport bugs.
    if evidence["status"] != "succeeded":
        failure_path = received.parent / "failure-evidence.json"
        failure_path.write_bytes(canonical_json(evidence))
        assert evidence["status"] == "succeeded", {
            "error": evidence["error"],
            "report": str(failure_path),
            "tasks": evidence.get("coordinator_report", {}).get("task_results"),
        }
    assert evidence["coordinator_report"]["workflow"]["state"] == "succeeded"
    assert evidence["business_execution"] == "real_cpu"
    assert evidence["sensor_source"] == "synthetic_known_pose_range_survey"
    assert evidence["physical_actuation"] is False
    assert evidence["gpu_tested"] is False
    assert evidence["energy_j"] is None
    assert {item["node_id"] for item in evidence["final_node_observations"]} == set(
        evidence["endpoints"]
    )
    for observation in evidence["final_node_observations"]:
        assert observation["online"] is True
        for field in ("cpu_utilization_ratio", "memory_utilization_ratio"):
            assert math.isfinite(observation[field])
            assert 0 <= observation[field] <= 1
    assert evidence["worker_elapsed_ms"] > 0
    assert evidence["validation"]["valid"] is True
    assert len(evidence["executions"]) == 4
    assert len(evidence["artifacts"]) == 5
    assert len(list(received.glob("*.json"))) >= 5
    for record in evidence["executions"]:
        assert record["execution_mode"] == "real_cpu"
        assert record["worker_elapsed_ms"] > 0
        assert record["host"]["agent_pid"] > 0
        assert record["host"]["architecture"]
        assert record["workflow_id"] == evidence["workflow_id"]
    for item in evidence["artifacts"]:
        ref, envelope = item["reference"], item["envelope"]
        data = (received / f"{ref['checksum']}.json").read_bytes()
        assert digest_bytes(data) == ref["checksum"]
        assert data == canonical_json(envelope)
        assert len(data) == pytest.approx(ref["size_mb"] * 1_000_000)
        assert envelope["workflow_id"] == evidence["workflow_id"]
        assert envelope["agent_id"] == ref["node_id"]
    # Compare computed payloads, not nondeterministic PIDs/timing/run identities.
    expected = execute("hil_sensor", {}, seed)
    expected.update(
        execute("hil_mapping", {"observations": expected["observations"]}, seed)
    )
    expected.update(execute("hil_planning", {"map": expected["map"]}, seed))
    expected.update(
        execute(
            "hil_validation",
            {port: expected[port] for port in ("map", "trajectory", "truth")},
            seed,
        )
    )
    assert _payloads(evidence) == expected


def test_navigation_dag_has_typed_fork_and_join_dependencies() -> None:
    workflow = navigation_workflow("split", "specific-workflow")
    tasks = {task.task_id: task for task in workflow.tasks}
    assert set(tasks) == {"sense", "map", "plan", "validate"}
    assert tasks["sense"].dependency_task_ids == ()
    assert tasks["map"].dependency_task_ids == ("sense",)
    assert tasks["plan"].dependency_task_ids == ("map",)
    assert set(tasks["validate"].dependency_task_ids) == {"sense", "map", "plan"}
    assert {
        (edge.producer_task, edge.producer_port, edge.consumer_task, edge.consumer_port)
        for edge in workflow.data_edges
    } == set(EDGES)
    for edge in workflow.data_edges:
        assert (
            edge.message_type
            == PORT_TYPES[TASKS[edge.producer_task]]["outputs"][edge.producer_port]
        )
        assert (
            edge.message_type
            == PORT_TYPES[TASKS[edge.consumer_task]]["inputs"][edge.consumer_port]
        )
    assert {
        task_id: task.spec.placement_constraints.pinned_node_id
        for task_id, task in tasks.items()
    } == {
        "sense": "robot_1",
        "map": "edge_pc",
        "plan": "edge_pc",
        "validate": "robot_1",
    }
    assert all(task.workflow_id == "specific-workflow" for task in tasks.values())
    assert all(
        task.spec.placement_constraints.required_capabilities == ("hil_navigation_v1",)
        for task in tasks.values()
    )


def test_split_hardware_loop_executes_real_workers_and_transfers_both_directions(
    tmp_path: Path,
) -> None:
    async def run():
        async with _agents(tmp_path) as (services, endpoints):
            received = tmp_path / "received"
            evidence = await run_hardware_loop(
                endpoints, artifact_directory=received, workflow_timeout_seconds=15
            )
            _assert_actual_pipeline(evidence, received, 19)
            assert evidence["scope"] == "same_host_cpu_execution"
            assert evidence["executing_host_count"] == 1
            assert evidence["executing_node_ids"] == ["edge_pc", "robot_1"]
            records = {
                record["task_id"]: record
                for service in services.values()
                for record in service.records
            }
            assert set(records) == set(TASKS)
            assert all(record["ok"] for record in records.values())
            assert records["sense"]["remote_input_bytes"] == 0
            assert records["plan"]["remote_input_bytes"] == 0
            assert records["map"]["remote_input_bytes"] > 0
            assert records["validate"]["remote_input_bytes"] > 0
            assert {
                entry["port"]: entry["producer_task_id"]
                for entry in records["validate"]["input_artifacts"]
            } == {"map": "map", "trajectory": "plan", "truth": "sense"}
            assert evidence["remote_input_bytes"] == sum(
                record["remote_input_bytes"] for record in records.values()
            )
            # Separate caches make producer URI references alone insufficient.
            assert (
                services["robot_1"].files.directory
                != services["edge_pc"].files.directory
            )
            for record in records.values():
                for artifact in record["input_artifacts"]:
                    data = services[record["agent_id"]].files.read(artifact["sha256"])
                    assert (
                        len(data) == artifact["remote_bytes"]
                        if artifact["producer_node_id"] != record["agent_id"]
                        else artifact["remote_bytes"] == 0
                    )

    asyncio.run(run())


@pytest.mark.parametrize(
    "peer_mode,failed_task",
    [
        ("missing_forward", "map"),
        ("wrong_forward", "map"),
        ("missing_return", "validate"),
    ],
)
def test_missing_or_wrong_artifact_peer_fails_the_real_workflow(
    tmp_path: Path, peer_mode: str, failed_task: str
) -> None:
    async def run():
        async with _agents(tmp_path) as (services, endpoints):
            if peer_mode == "missing_forward":
                services["edge_pc"].peers.pop("robot_1")
            elif peer_mode == "wrong_forward":
                services["edge_pc"].peers["robot_1"] = endpoints["edge_pc"]
            else:
                services["robot_1"].peers.pop("edge_pc")
            evidence = await run_hardware_loop(
                endpoints,
                artifact_directory=tmp_path / "received",
                workflow_timeout_seconds=10,
            )
            assert evidence["status"] == "failed"
            assert evidence["coordinator_report"]["workflow"]["state"] == "failed"
            assert "validation" not in evidence
            records = [
                record for service in services.values() for record in service.records
            ]
            failures = [
                record for record in records if record["task_id"] == failed_task
            ]
            assert failures, {"evidence": evidence, "executions": records}
            failed = failures[0]
            assert failed["ok"] is False
            assert failed["error_code"] == "execution_failed"
            assert failed["worker_elapsed_ms"] == 0
            if peer_mode == "wrong_forward":
                assert "artifact not found" in failed["error_message"]
            else:
                assert "not a configured peer" in failed["error_message"]

    asyncio.run(run())


def test_distinct_host_guard_rejects_two_local_agents_after_real_execution(
    tmp_path: Path,
) -> None:
    async def run():
        async with _agents(tmp_path) as (_, endpoints):
            evidence = await run_hardware_loop(
                endpoints,
                artifact_directory=tmp_path / "received",
                require_distinct_hosts=True,
                workflow_timeout_seconds=15,
            )
            assert evidence["status"] == "failed"
            assert evidence["coordinator_report"]["workflow"]["state"] == "succeeded"
            assert evidence["validation"]["valid"] is True
            assert evidence["executing_host_count"] == 1
            assert "same reported host" in evidence["error"]

    asyncio.run(run())


def test_hardware_loop_rejects_mock_agent(tmp_path: Path) -> None:
    async def run():
        config = next(
            item
            for item in load_agent_configs(REPO / "configs/mars/agents.local.json")
            if item.agent_id == "robot_1"
        )
        server, port = await start_agent_server(replace(config, listen="127.0.0.1:0"))
        try:
            evidence = await run_hardware_loop(
                {"robot_1": f"127.0.0.1:{port}"},
                placement="orin",
                artifact_directory=tmp_path,
                workflow_timeout_seconds=5,
            )
            assert evidence["status"] == "failed"
            assert "not Mock Agents" in evidence["error"]
            assert not evidence["artifacts"]
            assert "coordinator_report" not in evidence
        finally:
            await server.stop(0)

    asyncio.run(run())


def test_orin_mode_runs_all_four_real_stages_on_one_agent(tmp_path: Path) -> None:
    async def run():
        async with _agents(tmp_path, ("robot_1",)) as (services, endpoints):
            received = tmp_path / "received"
            evidence = await run_hardware_loop(
                endpoints,
                placement="orin",
                artifact_directory=received,
                workflow_timeout_seconds=15,
            )
            _assert_actual_pipeline(evidence, received, 19)
            assert evidence["executing_node_ids"] == ["robot_1"]
            assert evidence["remote_input_bytes"] == 0
            assert len(services["robot_1"].records) == 4
            assert all(
                record["agent_id"] == "robot_1" for record in evidence["executions"]
            )

    asyncio.run(run())


def test_saturated_host_remains_ineligible_for_new_work(tmp_path: Path) -> None:
    async def run():
        async with _agents(tmp_path, ("robot_1",)) as (services, endpoints):
            service = services["robot_1"]
            sample = service.telemetry.sample

            def saturated_sample(*args):
                snapshot = sample(*args)
                # Deliberately inject sustained saturation. Noise filtering must
                # never turn an actually full host into fictitious free capacity.
                snapshot.cpu_utilization_ratio = 1.0
                return snapshot

            service.telemetry.sample = saturated_sample
            evidence = await run_hardware_loop(
                endpoints,
                placement="orin",
                artifact_directory=tmp_path / "received",
                workflow_timeout_seconds=5,
            )
            assert evidence["status"] == "failed"
            assert evidence["coordinator_report"]["workflow"]["state"] == "failed"
            assert "no_feasible_agent" in evidence["error"]
            assert (
                evidence["final_node_observations"][0]["cpu_utilization_ratio"] == 1.0
            )
            sensor = next(
                task
                for task in evidence["coordinator_report"]["task_results"]
                if task["task_id"] == "sense"
            )
            assert sensor["attempts"][0]["error_code"] == "no_feasible_agent"
            assert service.records == []
            assert evidence["artifacts"] == []

    asyncio.run(run())


def test_repeated_runs_preserve_data_identity_and_use_fresh_workflow_ids(
    tmp_path: Path,
) -> None:
    async def run():
        async with _agents(tmp_path) as (services, endpoints):
            reports = []
            for index, seed in enumerate((19, 42, 19)):
                received = tmp_path / f"received-{index}"
                evidence = await run_hardware_loop(
                    endpoints,
                    seed=seed,
                    artifact_directory=received,
                    workflow_timeout_seconds=15,
                )
                _assert_actual_pipeline(evidence, received, seed)
                reports.append(evidence)
            assert len({report["workflow_id"] for report in reports}) == 3
            assert _payloads(reports[0]) == _payloads(reports[2])
            assert (
                _payloads(reports[0])["observations"]
                != _payloads(reports[1])["observations"]
            )
            # Completion replay and local caches must not masquerade as a new run.
            assert len(services["robot_1"].records) == 6
            assert len(services["edge_pc"].records) == 6
            assert all(report["remote_input_bytes"] > 0 for report in reports)
            assert {
                item["reference"]["checksum"] for item in reports[0]["artifacts"]
            }.isdisjoint(
                {item["reference"]["checksum"] for item in reports[2]["artifacts"]}
            )

    asyncio.run(run())


def test_real_downstream_computation_consumes_truncated_sensor_bytes(
    tmp_path: Path,
) -> None:
    class TruncatedSurveyExecutor(NavigationExecutor):
        async def execute(self, task_type: str, inputs: dict, seed: int):
            result = await super().execute(task_type, inputs, seed)
            if task_type == "hil_sensor":
                # Fault injection AFTER real acquisition, representing an
                # incomplete survey. No task's computation is mocked out.
                result.outputs["observations"]["scans"] = result.outputs[
                    "observations"
                ]["scans"][:1]
            return result

    async def run():
        async with _agents(tmp_path) as (services, endpoints):
            services["robot_1"].executor = TruncatedSurveyExecutor()
            evidence = await run_hardware_loop(
                endpoints,
                artifact_directory=tmp_path / "received",
                workflow_timeout_seconds=15,
            )
            (tmp_path / "failure-evidence.json").write_bytes(canonical_json(evidence))
            assert evidence["status"] == "failed"
            records = {
                record["task_id"]: record
                for service in services.values()
                for record in service.records
            }
            assert records["sense"]["ok"] is True
            assert records["map"]["ok"] is True
            assert records["map"]["remote_input_bytes"] > 0
            assert records["plan"]["ok"] is False
            assert "start or goal" in records["plan"]["error_message"]
            assert "validate" not in records
            edge = services["edge_pc"]
            observation_ref = records["map"]["input_artifacts"][0]
            received_observations = json.loads(
                edge.files.read(observation_ref["sha256"])
            )["payload"]
            assert len(received_observations["scans"]) == 1
            map_ref = records["plan"]["input_artifacts"][0]
            actual_map = json.loads(edge.files.read(map_ref["sha256"]))["payload"]
            assert actual_map["statistics"]["rays_integrated"] == 256
            assert actual_map["source_hashes"]["observations"] == digest_bytes(
                canonical_json(received_observations)
            )
            assert (
                actual_map
                == execute("hil_mapping", {"observations": received_observations}, 19)[
                    "map"
                ]
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "options",
    [
        {"endpoints": {"robot_1": "127.0.0.1:1"}},
        {"endpoints": {"robot_1": "127.0.0.1:1"}, "placement": "orin", "seed": -1},
        {
            "endpoints": {"robot_1": "127.0.0.1:1"},
            "placement": "orin",
            "workflow_timeout_seconds": 0,
        },
        {
            "endpoints": {"robot_1": "127.0.0.1:1"},
            "placement": "orin",
            "require_distinct_hosts": True,
        },
    ],
)
def test_hardware_loop_rejects_invalid_configuration_before_connecting(
    tmp_path: Path, options: dict
) -> None:
    with pytest.raises(ValueError):
        asyncio.run(run_hardware_loop(**options, artifact_directory=tmp_path))


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _wait_started(process: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"Agent exited during startup: {log.read_text()}")
        if "REAL CPU navigation" in log.read_text():
            return
        time.sleep(0.03)
    pytest.fail(f"Agent did not become ready: {log.read_text()}")


def test_agent_and_runner_cli_complete_real_split_loop_as_separate_processes(
    tmp_path: Path,
) -> None:
    # Hold two OS-selected ports until immediately before each Agent starts.
    # Child entrypoints do not support inherited listen sockets.
    with ExitStack() as stack:
        reservations = []
        for _ in range(2):
            reserved = stack.enter_context(socket.socket())
            reserved.bind(("127.0.0.1", 0))
            reservations.append(reserved)
        endpoints = {
            node: f"127.0.0.1:{reservation.getsockname()[1]}"
            for node, reservation in zip(("robot_1", "edge_pc"), reservations)
        }
        processes = {}
        for node, reserved in zip(("robot_1", "edge_pc"), reservations):
            log_path = tmp_path / f"{node}.log"
            log = stack.enter_context(log_path.open("w"))
            arguments = [
                sys.executable,
                "-m",
                "agent.main",
                "--executor",
                "navigation",
                "--agent-id",
                node,
                "--kind",
                "robot" if node == "robot_1" else "edge",
                "--listen",
                endpoints[node],
                "--artifact-dir",
                str(tmp_path / node),
                "--task-timeout",
                "10",
            ]
            for peer, endpoint in endpoints.items():
                if peer != node:
                    arguments.extend(["--peer", f"{peer}={endpoint}"])
            reserved.close()
            process = subprocess.Popen(
                arguments, cwd=REPO, stdout=log, stderr=subprocess.STDOUT
            )
            stack.callback(_stop_process, process)
            processes[node] = process
            _wait_started(process, log_path)
        output = tmp_path / "report.json"
        arguments = [
            sys.executable,
            "-m",
            "scripts.hardware_loop",
            "--placement",
            "split",
            "--seed",
            "42",
            "--workflow-timeout",
            "20",
            "--output",
            str(output),
        ]
        for node, endpoint in endpoints.items():
            arguments.extend(["--agent", f"{node}={endpoint}"])
        result = subprocess.run(
            arguments, cwd=REPO, text=True, capture_output=True, timeout=30
        )
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads(result.stdout)
        assert summary["status"] == "succeeded"
        evidence = json.loads(output.read_text())
        _assert_actual_pipeline(evidence, tmp_path / "received-artifacts", 42)
        assert evidence["scope"] == "same_host_cpu_execution"
        assert evidence["remote_input_bytes"] > 0
        records = {record["task_id"]: record for record in evidence["executions"]}
        assert records["sense"]["host"]["agent_pid"] == processes["robot_1"].pid
        assert records["map"]["host"]["agent_pid"] == processes["edge_pc"].pid
        assert (
            records["sense"]["host"]["agent_pid"] != records["map"]["host"]["agent_pid"]
        )
        saved = output.read_bytes()
        rerun = subprocess.run(
            arguments, cwd=REPO, text=True, capture_output=True, timeout=10
        )
        assert rerun.returncode == 2
        assert "output already exists" in rerun.stderr
        assert output.read_bytes() == saved
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            process.wait(timeout=8)
            assert process.returncode == 0
