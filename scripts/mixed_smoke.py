"""Run real CPU/native-CUDA navigation smoke and retain independently checked evidence.

Two physical hosts are required by default. --allow-same-host is development
transport mode and can never grant hardware acceptance. CUDA fixture evidence
is explicitly identified and never counts as GPU execution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

from agent.artifacts import ArtifactFiles, canonical_json, digest_bytes, fetch_artifact
from agent.endpoints import parse_endpoints
from agent.executor import NavigationExecutor
from examples.mixed_workloads.pipeline import PORT_TYPES, verify_measurement
from mars.coordinator import CentralCoordinator
from mars.domain.artifact import ArtifactRef
from mars.domain.task import (
    DataPort,
    PlacementConstraints,
    ResourceClass,
    TaskClass,
    TaskInstance,
    TaskSpec,
)
from mars.domain.topology import LinkSnapshot, LinkSpec, NodeKind
from mars.domain.workflow import DataEdge, FailurePolicy, WorkflowSpec
from mars.profiling import ExecutionProfile, ProfileCatalog
from mars.runtime.grpc import GrpcRuntimeAdapter


TASKS = {
    "sense": "hil_mixed_sensor",
    "map": "hil_mixed_mapping",
    "inflate": "hil_mixed_inflation",
    "plan": "hil_mixed_planning",
    "validate": "hil_mixed_validation",
}
PLACEMENTS = {
    "sense": "robot_1",
    "map": "edge_pc",
    "inflate": "robot_1",
    "plan": "edge_pc",
    "validate": "robot_1",
}
EDGES = (
    ("sense", "observations", "map", "observations"),
    ("map", "map", "inflate", "map"),
    ("map", "map", "plan", "map"),
    ("inflate", "inflated", "plan", "inflated"),
    ("map", "map", "validate", "map"),
    ("inflate", "inflated", "validate", "inflated"),
    ("plan", "trajectory", "validate", "trajectory"),
    ("sense", "truth", "validate", "truth"),
)
MAX_RUNS = 20
PROFILE_SOURCE = "unmeasured_mixed_smoke_bootstrap_prior"
VALIDATION_FLOAT_TOLERANCE = {
    "rel_tol": 1e-9,
    "abs_tol": 1e-9,
    "fields": [
        "path_length_m",
        "planned_motion_duration_s",
        "minimum_obstacle_clearance_m",
        "maximum_speed_m_s",
        "maximum_acceleration_m_s2",
        "maximum_yaw_rate_rad_s",
    ],
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _hash(value) -> str:
    return digest_bytes(canonical_json(value))


def _positive_timeout(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def mixed_workflow(
    workflow_id: str | None = None,
    *,
    deadline_ms: float = 180_000,
) -> WorkflowSpec:
    _positive_timeout(deadline_ms, "workflow deadline")
    workflow_id = workflow_id or f"hil-mixed-{uuid4().hex}"
    tasks = []
    for index, (task_id, task_type) in enumerate(TASKS.items()):
        node = PLACEMENTS[task_id]
        gpu = task_id == "inflate"
        contract = PORT_TYPES[task_type]
        tasks.append(
            TaskInstance(
                task_id=task_id,
                workflow_id=workflow_id,
                name=task_type,
                source_node_id="robot_1",
                stage_index=index,
                deadline_time_ms=deadline_ms,
                dependency_task_ids=tuple(
                    dict.fromkeys(
                        source for source, _, target, _ in EDGES if target == task_id
                    )
                ),
                spec=TaskSpec(
                    task_type=task_type,
                    task_class=TaskClass.REALTIME_OFFLOADABLE,
                    compute_demand=0.1,
                    gpu_demand=float(gpu),
                    latency_budget_ms=deadline_ms,
                    output_size_mb=0.5,
                    dominant_resource=ResourceClass.GPU if gpu else ResourceClass.CPU,
                    input_ports=tuple(
                        DataPort(k, v) for k, v in contract["inputs"].items()
                    ),
                    output_ports=tuple(
                        DataPort(k, v) for k, v in contract["outputs"].items()
                    ),
                    placement_constraints=PlacementConstraints(
                        pinned_node_id=node,
                        allowed_node_kinds=(NodeKind.ROBOT, NodeKind.EDGE),
                        required_capabilities=(
                            "hil_mixed_orin_v1"
                            if node == "robot_1"
                            else "hil_mixed_pc_v1",
                            "cuda" if gpu else "cpu",
                        ),
                        allow_fallback=False,
                    ),
                ),
            )
        )
    return WorkflowSpec(
        workflow_id,
        tuple(tasks),
        deadline_time_ms=deadline_ms,
        failure_policy=FailurePolicy.FAIL_FAST,
        metadata={"purpose": "mixed_cpu_native_cuda_hardware_smoke"},
        data_edges=tuple(
            DataEdge(
                source,
                port,
                target,
                consumer,
                PORT_TYPES[TASKS[source]]["outputs"][port],
            )
            for source, port, target, consumer in EDGES
        ),
    )


def initial_profiles() -> ProfileCatalog:
    """Planning estimates only; no measured performance or energy is implied."""
    return ProfileCatalog(
        [
            ExecutionProfile(
                task_type=task_type,
                task_class=TaskClass.REALTIME_OFFLOADABLE,
                node_kind=NodeKind.ROBOT
                if PLACEMENTS[task_id] == "robot_1"
                else NodeKind.EDGE,
                model_variant=task_type,
                input_shape="bounded_navigation_grid",
                precision="integer_mask_and_python_float64",
                batch_size=1,
                p50_ms=100.0,
                p95_ms=1000.0,
                p99_ms=5000.0,
                throughput_per_s=1.0,
                peak_memory_mb=128.0,
                energy_j=0.0,
                output_size_mb=0.5,
                cpu_units=1.0,
                gpu_units=float(task_id == "inflate"),
                provenance=PROFILE_SOURCE,
            )
            for task_id, task_type in TASKS.items()
        ]
    )


async def _reference_validation(inputs: dict, seed: int) -> dict:
    # A separate process makes the CPU reference genuinely cancellable. A
    # to_thread reference would keep running after the evidence deadline.
    executor = NavigationExecutor()
    executor.worker_module = "examples.mixed_workloads.worker"
    executor.ports = PORT_TYPES
    result = await executor.execute(TASKS["validate"], inputs, seed)
    return result.outputs["validation"]


def _exact_validation_value(expected, returned) -> bool:
    """Exact JSON values and types, including nested hashes and checklists."""
    if type(expected) is not type(returned):
        return False
    if isinstance(expected, dict):
        return expected.keys() == returned.keys() and all(
            _exact_validation_value(value, returned[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(returned) and all(
            _exact_validation_value(first, second)
            for first, second in zip(expected, returned)
        )
    if isinstance(expected, float):
        return (
            math.isfinite(expected) and math.isfinite(returned) and expected == returned
        )
    return expected == returned


def _validation_matches(expected: dict, returned: dict) -> bool:
    """Permit libm rounding only in explicitly listed derived float metrics.

    Copied inputs (including robot_radius_m), integer counts, source digests and
    all other values stay exact. This never changes artifact or GPU-grid checks.
    """
    if type(expected) is not dict or type(returned) is not dict:
        return False
    if expected.keys() != returned.keys():
        return False
    for field, value in expected.items():
        actual = returned[field]
        if field in VALIDATION_FLOAT_TOLERANCE["fields"] and type(value) is float:
            if (
                type(actual) is not float
                or not math.isfinite(value)
                or not math.isfinite(actual)
                or not math.isclose(
                    value,
                    actual,
                    rel_tol=VALIDATION_FLOAT_TOLERANCE["rel_tol"],
                    abs_tol=VALIDATION_FLOAT_TOLERANCE["abs_tol"],
                )
            ):
                return False
        elif not _exact_validation_value(value, actual):
            return False
    return True


def _hardware_checks(hosts: dict, measurement: dict, *, fixture: bool) -> dict:
    pc, orin = hosts["edge_pc"], hosts["robot_1"]
    machines = [host.get("machine_id_sha256") for host in (pc, orin)]
    sources = [host.get("runtime_source_sha256") for host in (pc, orin)]
    revisions = [host.get("git_revision") for host in (pc, orin)]
    release = orin.get("jetson_linux")
    return {
        "distinct_machine_ids": (
            all(
                isinstance(v, str) and _SHA256.fullmatch(v) is not None
                for v in machines
            )
            and machines[0] != machines[1]
        ),
        "matching_runtime_source": (
            all(
                isinstance(v, str) and _SHA256.fullmatch(v) is not None for v in sources
            )
            and sources[0] == sources[1]
        ),
        "matching_git_revision": (
            all(
                isinstance(v, str) and _GIT_REVISION.fullmatch(v) is not None
                for v in revisions
            )
            and revisions[0] == revisions[1]
        ),
        "target_architectures": (
            pc.get("architecture") == "x86_64"
            and orin.get("architecture") in {"aarch64", "arm64"}
        ),
        "jetson_agx_orin": "Jetson AGX Orin" in str(orin.get("jetson_model") or ""),
        "orin_compute_capability": measurement.get("compute_capability") == [8, 7],
        "jetpack721": (
            isinstance(release, str)
            and re.search(r"\bR39\b.*\bREVISION:\s*2\.1(?:\s|,|$)", release) is not None
            and measurement.get("cuda_runtime_version") == 13020
        ),
        "no_test_fixtures": not fixture,
        "native_binary_identity": all(
            isinstance(measurement.get(k), str)
            and _SHA256.fullmatch(measurement[k]) is not None
            for k in ("source_sha256", "binary_sha256")
        ),
    }


def _verify_execution_graph(evidence: dict, report: dict) -> tuple[dict, dict]:
    """Bind every output and input to its successful attempt and exact DAG edge."""
    results = {task["task_id"]: task for task in report["task_results"]}
    if set(results) != set(TASKS) or len(results) != len(report["task_results"]):
        raise ValueError("coordinator task set differs from the mixed DAG")
    expected_ports = {
        (task_id, port)
        for task_id, task_type in TASKS.items()
        for port in PORT_TYPES[task_type]["outputs"]
    }
    artifacts, records, hosts = {}, {}, {}
    for item in evidence["artifacts"]:
        ref, envelope = item["reference"], item["envelope"]
        key = (envelope["producer_task_id"], envelope["producer_port"])
        if key not in expected_ports or key in artifacts:
            raise ValueError("unexpected or duplicate output artifact port")
        task_id, port = key
        result = results[task_id]
        record = envelope.get("execution")
        node = PLACEMENTS[task_id]
        if (
            envelope.get("workflow_id") != evidence["workflow_id"]
            or envelope.get("agent_id") != node
            or envelope.get("message_type")
            != PORT_TYPES[TASKS[task_id]]["outputs"][port]
            or ref not in result["outputs"]
            or not isinstance(record, dict)
            or record.get("schema") != "mars.hil.execution.v1"
            or record.get("workflow_id") != evidence["workflow_id"]
            or record.get("agent_id") != node
            or record.get("task_id") != task_id
            or record.get("task_type") != TASKS[task_id]
            or record.get("execution_mode")
            != ("real_cuda" if task_id == "inflate" else "real_cpu")
            or not isinstance(record.get("dispatch_id"), str)
            or not record["dispatch_id"]
        ):
            raise ValueError(
                f"{task_id} execution placement/identity does not match the DAG"
            )
        attempts = result.get("attempts", [])
        if (
            result.get("state") != "succeeded"
            or result.get("target_node_id") != node
            or len(attempts) != 1
            or attempts[0].get("state") != "succeeded"
            or attempts[0].get("attempt_id") != record.get("attempt_id")
            or attempts[0].get("target_node_id") != node
        ):
            raise ValueError(
                f"{task_id} evidence does not match its successful attempt"
            )
        for field in ("worker_elapsed_ms", "input_fetch_ms"):
            value = record.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {task_id} {field}")
        if task_id in records and canonical_json(records[task_id]) != canonical_json(
            record
        ):
            raise ValueError("multiple output ports disagree about their execution")
        host = record.get("host")
        if not isinstance(host, dict) or any(
            not isinstance(host.get(k), str) or not host[k]
            for k in ("hostname", "architecture")
        ):
            raise ValueError("missing execution host identity")
        if node in hosts and canonical_json(hosts[node]) != canonical_json(host):
            raise ValueError(f"inconsistent host identity for {node}")
        hosts[node] = host
        records[task_id] = record
        artifacts[key] = item
    if set(artifacts) != expected_ports:
        raise ValueError(
            "missing output artifacts: expected six artifacts for five executions"
        )
    if len({r["attempt_id"] for r in records.values()}) != len(TASKS):
        raise ValueError("task executions reused an attempt identity")
    evidence["checks"].update(
        artifact_ports=True,
        execution_placement=True,
        host_identity_consistent=True,
    )
    edges = []
    for task_id, record in records.items():
        expected = [edge for edge in EDGES if edge[2] == task_id]
        bindings = record.get("input_artifacts")
        if not isinstance(bindings, list) or len(bindings) != len(expected):
            raise ValueError(f"{task_id} does not record every DAG input")
        by_port = {binding["port"]: binding for binding in bindings}
        if len(by_port) != len(bindings) or set(by_port) != {
            edge[3] for edge in expected
        }:
            raise ValueError(f"{task_id} has duplicate or incorrect input ports")
        remote_total = 0
        expected_ids = []
        for source, port, target, consumer in expected:
            item = artifacts[source, port]
            ref, binding = item["reference"], by_port[consumer]
            expected_ids.append(ref["artifact_id"])
            remote = PLACEMENTS[source] != PLACEMENTS[target]
            # Count producer-envelope bytes, not canonical payload bytes. Local
            # edges must remain zero even if another node fetched this artifact.
            byte_count = len(canonical_json(item["envelope"])) if remote else 0
            if (
                binding.get("producer_task_id") != source
                or binding.get("producer_node_id") != PLACEMENTS[source]
                or binding.get("sha256") != ref["checksum"]
                or type(binding.get("remote_bytes")) is not int
                or binding["remote_bytes"] != byte_count
            ):
                raise ValueError(
                    f"edge {source}.{port}->{target}.{consumer} reference/remote bytes mismatch"
                )
            remote_total += byte_count
            edges.append(
                {
                    "source_task": source,
                    "source_port": port,
                    "target_task": target,
                    "target_port": consumer,
                    "artifact_sha256": ref["checksum"],
                    "remote_bytes": byte_count,
                }
            )
        if sorted(
            results[task_id]["attempts"][0].get("input_artifact_ids", [])
        ) != sorted(expected_ids):
            raise ValueError(
                f"{task_id} coordinator input references disagree with its edges"
            )
        if (
            type(record.get("remote_input_bytes")) is not int
            or record["remote_input_bytes"] != remote_total
        ):
            raise ValueError(
                f"{task_id} aggregate remote bytes disagree with its edges"
            )
    evidence.update(
        executions=[records[k] for k in TASKS],
        hosts=hosts,
        edges=edges,
        executing_node_ids=sorted(hosts),
        remote_input_bytes=sum(
            record["remote_input_bytes"] for record in records.values()
        ),
        worker_elapsed_ms=sum(
            record["worker_elapsed_ms"] for record in records.values()
        ),
    )
    evidence["checks"]["edge_transfers"] = True
    return artifacts, hosts


async def _collect_and_verify(evidence: dict, report: dict, files, endpoints) -> None:
    for task in report["task_results"]:
        for output in task["outputs"]:
            envelope, _ = await fetch_artifact(
                ArtifactRef(**output),
                agent_id="coordinator",
                files=files,
                peers=endpoints,
            )
            evidence["artifacts"].append({"reference": output, "envelope": envelope})
    if report["workflow"]["state"] != "succeeded":
        failures = "; ".join(
            f"{task['task_id']}: {task['state']} "
            f"({task.get('attempts', [{}])[-1].get('error_code', '') if task.get('attempts') else ''})"
            for task in report["task_results"]
            if task["state"] != "succeeded"
        )
        raise RuntimeError("workflow failed: " + failures)
    artifacts, hosts = _verify_execution_graph(evidence, report)
    payloads = {
        port: item["envelope"]["payload"] for (_, port), item in artifacts.items()
    }
    observations, truth, mapped = (
        payloads[k] for k in ("observations", "truth", "map")
    )
    if any(
        type(payload.get("seed")) is not int or payload["seed"] != evidence["seed"]
        for payload in (observations, truth)
    ):
        raise ValueError("sensor observations/truth do not match this run's seed")
    if mapped.get("source_hashes") != {"observations": _hash(observations)}:
        raise ValueError("mapping did not consume the returned observations")
    if any(
        payload.get("scene_id") != truth.get("scene_id")
        for payload in payloads.values()
    ):
        raise ValueError("returned artifacts disagree about scene identity")
    evidence["checks"]["source_lineage"] = True
    measurement = verify_measurement(payloads["inflated"].get("measurement"))
    if any(
        not isinstance(measurement.get(k), str) or not _SHA256.fullmatch(measurement[k])
        for k in ("source_sha256", "binary_sha256")
    ):
        raise ValueError("native CUDA source/binary SHA256 identity is missing")
    preflight = hosts["robot_1"].get("cuda_device")
    if not isinstance(preflight, dict):
        raise ValueError("Orin execution host is missing CUDA preflight identity")
    if (
        preflight.get("available") is not True
        or preflight.get("kernel_execution_verified") is not True
        or type(preflight.get("device_count")) is not int
        or int(measurement["device"][5:]) >= preflight["device_count"]
    ):
        raise ValueError(
            "CUDA preflight did not verify the selected device by execution"
        )
    capability = measurement.get("compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int for value in capability)
        or capability[0] < 1
        or capability[1] < 0
    ):
        raise ValueError("CUDA measurement is missing a valid compute capability")
    for key in (
        "backend",
        "device",
        "device_name",
        "cuda_runtime_version",
        "cuda_driver_version",
        "source_sha256",
        "binary_sha256",
        "compute_capability",
    ):
        if (
            type(preflight.get(key)) is not type(measurement[key])
            or preflight.get(key) != measurement[key]
        ):
            raise ValueError(f"CUDA execution differs from preflight {key}")
    evidence["checks"]["cuda_measurement"] = True
    evidence["gpu_execution"] = {
        "task_id": "inflate",
        "agent_id": "robot_1",
        "measurement": measurement,
    }
    inputs = {port: payloads[port] for port in PORT_TYPES[TASKS["validate"]]["inputs"]}
    expected_validation = await _reference_validation(inputs, evidence["seed"])
    returned_validation = payloads["validation"]
    if not _validation_matches(expected_validation, returned_validation):
        raise ValueError(
            "returned validation differs from independent CPU validation fields/digests"
        )
    evidence.update(
        validation=returned_validation,
        independent_validation_sha256=_hash(expected_validation),
        payload_sha256={key: _hash(value) for key, value in payloads.items()},
    )
    evidence["checks"]["independent_validation"] = True
    fixture = any(
        value.get("test_fixture") is True
        for value in [
            measurement,
            *payloads.values(),
            *hosts.values(),
            *evidence["executions"],
        ]
    )
    hardware_checks = _hardware_checks(hosts, measurement, fixture=fixture)
    evidence["checks"].update(hardware_checks)
    machines = [host.get("machine_id_sha256") for host in hosts.values()]
    if all(isinstance(value, str) and _SHA256.fullmatch(value) for value in machines):
        evidence["executing_host_count"] = len(set(machines))
        evidence["host_count_basis"] = "machine_id_sha256"
    else:
        evidence["executing_host_count"] = len(
            {(h["hostname"], h["architecture"]) for h in hosts.values()}
        )
        evidence["host_count_basis"] = "reported_hostname_architecture_only"
    evidence["execution_evidence_kind"] = (
        "test_fixture" if fixture else "trusted_agent_report"
    )
    evidence["gpu_tested"] = not fixture
    required = set(hardware_checks) - {"jetpack721"}
    if evidence["require_jetpack721"]:
        required.add("jetpack721")
    evidence["hardware_gate_failures"] = sorted(
        k for k in required if not hardware_checks[k]
    )
    if evidence["allow_same_host"]:
        evidence["scope"] = (
            "development_transport_fixture" if fixture else "development_transport_only"
        )
    elif evidence["hardware_gate_failures"]:
        raise ValueError(
            "hardware acceptance failed: "
            + ", ".join(evidence["hardware_gate_failures"])
        )
    else:
        evidence["scope"] = "cross_host_cpu_native_cuda_execution"
        evidence["hardware_smoke_passed"] = True


async def _run_once(
    endpoints: dict[str, str],
    *,
    seed: int,
    artifact_directory: Path,
    workflow_timeout_seconds: float,
    task_completion_timeout_seconds: float,
    evidence_timeout_seconds: float,
    allow_same_host: bool,
    require_jetpack721: bool,
) -> dict:
    workflow = mixed_workflow(deadline_ms=workflow_timeout_seconds * 1000)
    runtime = GrpcRuntimeAdapter(
        endpoints, completion_timeout_seconds=task_completion_timeout_seconds
    )
    evidence = {
        "schema": "mars.hil.mixed_run.v1",
        "workflow_id": workflow.workflow_id,
        "seed": seed,
        "status": "failed",
        "scope": "unverified",
        "hardware_smoke_passed": False,
        "gpu_tested": False,
        "validation_float_tolerance": VALIDATION_FLOAT_TOLERANCE,
        "allow_same_host": allow_same_host,
        "require_jetpack721": require_jetpack721,
        "artifacts": [],
        "checks": {},
        "error": None,
        "phase": "workflow",
        "energy_j": None,
    }
    started = perf_counter()
    try:
        files = ArtifactFiles(artifact_directory)

        async def execute_workflow():
            inventory = await runtime.start(0)
            nodes = {node.node_id: node for node in inventory.nodes}
            if set(nodes) != set(endpoints):
                raise ValueError("registered node inventory differs from endpoints")
            for task in workflow.tasks:
                node_id = PLACEMENTS[task.task_id]
                node = nodes[node_id]
                if node.kind != (
                    NodeKind.ROBOT if node_id == "robot_1" else NodeKind.EDGE
                ):
                    raise ValueError(f"{node_id} has the wrong node kind")
                if not set(
                    task.spec.placement_constraints.required_capabilities
                ).issubset(node.capabilities):
                    raise ValueError(
                        f"{node_id} lacks required mixed executor capabilities"
                    )
                if task.spec.gpu_demand and (
                    not math.isfinite(node.gpu_capacity) or node.gpu_capacity < 1
                ):
                    raise ValueError("Orin did not advertise CUDA capacity")
            links = tuple(
                LinkSpec(f"mixed:{source}:{target}", source, target, 100.0, 0.0)
                for source in endpoints
                for target in endpoints
                if source != target
            )
            coordinator = CentralCoordinator(
                runtime,
                link_specs=links,
                link_snapshots=tuple(
                    LinkSnapshot(link.link_id, 100.0) for link in links
                ),
                profile_catalog=initial_profiles(),
            )
            return await coordinator.run_async(
                workflow, algorithm="heuristic", seed=seed, max_attempts=1
            )

        report = await asyncio.wait_for(
            execute_workflow(), timeout=workflow_timeout_seconds
        )
        evidence["workflow_wall_elapsed_ms"] = (perf_counter() - started) * 1000
        report = report.as_dict()
        evidence["coordinator_report"] = report
        evidence["phase"] = "evidence"
        evidence_started = perf_counter()
        try:
            await asyncio.wait_for(
                _collect_and_verify(evidence, report, files, endpoints),
                timeout=evidence_timeout_seconds,
            )
        finally:
            evidence["evidence_wall_elapsed_ms"] = (
                perf_counter() - evidence_started
            ) * 1000
        evidence["status"] = "succeeded"
        evidence["phase"] = "complete"
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc} ({evidence['phase']} phase)"
        evidence["hardware_smoke_passed"] = False
    finally:
        evidence["final_node_observations"] = [
            {
                "node_id": snapshot.node_id,
                "cpu_utilization_ratio": snapshot.cpu_util,
                "memory_utilization_ratio": snapshot.memory_util,
                "online": snapshot.online,
            }
            for snapshot in runtime.snapshots
        ]
        try:
            await asyncio.wait_for(runtime.close(), timeout=10.0)
        except Exception as exc:
            evidence["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            evidence["error"] = evidence["error"] or evidence["cleanup_error"]
            evidence["status"] = "failed"
            evidence["hardware_smoke_passed"] = False
        evidence["total_wall_elapsed_ms"] = (perf_counter() - started) * 1000
    return evidence


async def run_mixed_smoke(
    endpoints: dict[str, str],
    *,
    seed: int = 19,
    runs: int = 3,
    artifact_directory: str | Path = ".mars-hil/mixed-received",
    workflow_timeout_seconds: float = 180.0,
    task_completion_timeout_seconds: float = 120.0,
    evidence_timeout_seconds: float = 60.0,
    allow_same_host: bool = False,
    require_jetpack721: bool = False,
) -> dict:
    """Return one aggregate report; sequential seeds stop at the first failed run.

    Invalid configuration raises ValueError. Execution/evidence failures are
    retained in the report. External cancellation propagates after cleanup.
    """
    if set(endpoints) != {"robot_1", "edge_pc"}:
        raise ValueError("mixed smoke requires exactly robot_1 and edge_pc endpoints")
    if type(runs) is not int or not 1 <= runs <= MAX_RUNS:
        raise ValueError(f"runs must be an integer in [1, {MAX_RUNS}]")
    if type(seed) is not int or not 0 <= seed <= 2**32 - runs:
        raise ValueError("all sequential seeds must lie in [0, 2**32)")
    for value, name in (
        (workflow_timeout_seconds, "workflow timeout"),
        (task_completion_timeout_seconds, "task completion timeout"),
        (evidence_timeout_seconds, "evidence timeout"),
    ):
        _positive_timeout(value, name)
    if task_completion_timeout_seconds > workflow_timeout_seconds:
        raise ValueError("task completion timeout must not exceed workflow timeout")
    report = {
        "schema": "mars.hil.mixed_smoke.v1",
        "status": "failed",
        "scope": "unverified",
        "hardware_smoke_passed": False,
        "gpu_tested": False,
        "validation_float_tolerance": VALIDATION_FLOAT_TOLERANCE,
        "seed": seed,
        "requested_runs": runs,
        "endpoints": dict(endpoints),
        "runs": [],
        "allow_same_host": allow_same_host,
        "require_jetpack721": require_jetpack721,
        "sensor_source": "synthetic_known_pose_range_survey",
        "physical_actuation": False,
        "business_execution": "real_cpu_and_native_cuda",
        "energy_j": None,
        "error": None,
        "timeouts_seconds": {
            "workflow": workflow_timeout_seconds,
            "task_completion": task_completion_timeout_seconds,
            "evidence": evidence_timeout_seconds,
        },
        "dag": {"tasks": TASKS, "placements": PLACEMENTS, "edges": EDGES},
        "planning_assumptions": {
            "profiles": PROFILE_SOURCE,
            "link_bandwidth_mbps": 100.0,
            "link_latency_ms": 0.0,
            "scheduler_timestamps": "logical_dispatch_anchor_plus_measured_elapsed",
            "scheduler_communication_metrics": "estimated_not_measured",
            "gpu_utilization": "unmeasured_v1_zero_placeholder",
            "memory": "host_utilization_and_explicit_device_allocations_not_peak_process_memory",
            "concurrency": "one_worker_per_agent",
        },
    }
    started = perf_counter()
    baseline_hosts = None
    for offset in range(runs):
        result = await _run_once(
            endpoints,
            seed=seed + offset,
            artifact_directory=Path(artifact_directory),
            workflow_timeout_seconds=workflow_timeout_seconds,
            task_completion_timeout_seconds=task_completion_timeout_seconds,
            evidence_timeout_seconds=evidence_timeout_seconds,
            allow_same_host=allow_same_host,
            require_jetpack721=require_jetpack721,
        )
        report["runs"].append(result)
        if result["status"] == "succeeded":
            if (
                baseline_hosts is not None
                and canonical_json(result["hosts"]) != baseline_hosts
            ):
                result.update(
                    status="failed",
                    hardware_smoke_passed=False,
                    error="host/runtime identity changed between runs",
                )
            baseline_hosts = canonical_json(result["hosts"])
        if result["status"] != "succeeded":
            report["error"] = result["error"]
            break
    report["completed_runs"] = len(report["runs"])
    report["stopped_after_failure"] = report["runs"][-1]["status"] != "succeeded"
    if len(report["runs"]) == runs and all(
        r["status"] == "succeeded" for r in report["runs"]
    ):
        report["status"] = "succeeded"
        report["scope"] = report["runs"][0]["scope"]
        report["hardware_smoke_passed"] = all(
            r["hardware_smoke_passed"] for r in report["runs"]
        )
    report["gpu_tested"] = (
        all(r["gpu_tested"] for r in report["runs"]) and len(report["runs"]) == runs
    )
    report["total_wall_elapsed_ms"] = (perf_counter() - started) * 1000
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", action="append", required=True, metavar="NODE=HOST:PORT"
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow-timeout", type=float, default=180.0)
    parser.add_argument("--task-completion-timeout", type=float, default=120.0)
    parser.add_argument("--evidence-timeout", type=float, default=60.0)
    parser.add_argument(
        "--allow-same-host",
        action="store_true",
        help="development only; never hardware acceptance",
    )
    parser.add_argument(
        "--require-jetpack721",
        action="store_true",
        help="also require Jetson Linux R39 revision 2.1 and CUDA runtime 13.2",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file to preserve evidence")
    try:
        report = asyncio.run(
            run_mixed_smoke(
                parse_endpoints(args.agent),
                seed=args.seed,
                runs=args.runs,
                artifact_directory=args.output.parent / "received-artifacts",
                workflow_timeout_seconds=args.workflow_timeout,
                task_completion_timeout_seconds=args.task_completion_timeout,
                evidence_timeout_seconds=args.evidence_timeout,
                allow_same_host=args.allow_same_host,
                require_jetpack721=args.require_jetpack721,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            output.write(canonical_json(report) + b"\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"invalid mixed smoke configuration/output: {exc}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.output),
                "scope": report["scope"],
                "hardware_smoke_passed": report["hardware_smoke_passed"],
                "gpu_tested": report["gpu_tested"],
                "error": report["error"],
            }
        )
    )
    raise SystemExit(0 if report["status"] == "succeeded" else 1)


if __name__ == "__main__":
    main()
