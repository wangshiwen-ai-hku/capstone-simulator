"""Run a measured CUDA or pretrained SmolVLA DAG through two MARS Agents.

This verifies inference and artifact transport, without physical actuation.
Run from the checkout root: python -m scripts.vla_loop --help
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
from examples.vla_workloads import PORT_TYPES
from examples.vla_workloads.bundle import (
    POLICY_ID,
    POLICY_REVISION,
    VLM_ID,
    VLM_REVISION,
)
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


MODEL_ID = POLICY_ID
TASKS = {
    "smolvla": {
        "observe": "hil_vla_observe",
        "infer": "hil_vla_infer",
        "validate": "hil_vla_validate",
    },
    "cuda": {"smoke": "hil_cuda_smoke", "validate": "hil_cuda_validate"},
}
EDGES = {
    "smolvla": (
        ("observe", "observation", "infer", "observation"),
        ("observe", "observation", "validate", "observation"),
        ("infer", "actions", "validate", "actions"),
    ),
    "cuda": (("smoke", "cuda_result", "validate", "cuda_result"),),
}
GPU_TASK_TYPES = frozenset({"hil_vla_infer", "hil_cuda_smoke"})
PROFILE_SOURCE = "unmeasured_cuda_vla_bootstrap_prior"


def _positive_timeout(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _io_agent(gpu_agent: str) -> str:
    if gpu_agent not in {"robot_1", "edge_pc"}:
        raise ValueError("GPU agent must be robot_1 or edge_pc")
    return "edge_pc" if gpu_agent == "robot_1" else "robot_1"


def vla_workflow(
    workload: str = "smolvla",
    gpu_agent: str = "robot_1",
    workflow_id: str | None = None,
    *,
    deadline_ms: float = 600_000,
) -> WorkflowSpec:
    if workload not in TASKS:
        raise ValueError("workload must be cuda or smolvla")
    _positive_timeout(deadline_ms, "workflow deadline")
    io_agent = _io_agent(gpu_agent)
    workflow_id = workflow_id or f"hil-{workload}-{uuid4().hex}"
    tasks = []
    for index, (task_id, task_type) in enumerate(TASKS[workload].items()):
        gpu = task_type in GPU_TASK_TYPES
        capability = "hil_smolvla_v1" if task_type == "hil_vla_infer" else "hil_cuda_v1"
        contract = PORT_TYPES[task_type]
        tasks.append(
            TaskInstance(
                task_id=task_id,
                workflow_id=workflow_id,
                name=task_type,
                source_node_id=io_agent,
                spec=TaskSpec(
                    task_type=task_type,
                    task_class=TaskClass.REALTIME_OFFLOADABLE,
                    compute_demand=0.1,
                    gpu_demand=1.0 if gpu else 0.0,
                    latency_budget_ms=deadline_ms,
                    model_requirement=MODEL_ID if task_type == "hil_vla_infer" else "",
                    output_size_mb=1.5 if task_id == "observe" else 0.2,
                    dominant_resource=ResourceClass.GPU if gpu else ResourceClass.CPU,
                    input_ports=tuple(
                        DataPort(k, v) for k, v in contract["inputs"].items()
                    ),
                    output_ports=tuple(
                        DataPort(k, v) for k, v in contract["outputs"].items()
                    ),
                    placement_constraints=PlacementConstraints(
                        pinned_node_id=gpu_agent if gpu else io_agent,
                        allowed_node_kinds=(NodeKind.ROBOT, NodeKind.EDGE),
                        required_capabilities=("cuda", capability)
                        if gpu
                        else ("hil_vla_io_v1",),
                        # The source is the I/O host. When it is the PC, the
                        # explicitly pinned Orin is another robot by topology.
                        allow_other_robots=True,
                        allow_fallback=False,
                    ),
                ),
                dependency_task_ids=tuple(
                    dict.fromkeys(
                        source
                        for source, _, target, _ in EDGES[workload]
                        if target == task_id
                    )
                ),
                stage_index=index,
                deadline_time_ms=deadline_ms,
            )
        )
    return WorkflowSpec(
        workflow_id,
        tuple(tasks),
        deadline_time_ms=deadline_ms,
        failure_policy=FailurePolicy.FAIL_FAST,
        metadata={
            "purpose": "cuda_inference_and_transport_validation",
            "workload": workload,
            "gpu_agent": gpu_agent,
            "io_agent": io_agent,
            "physical_actuation": "false",
        },
        data_edges=tuple(
            DataEdge(
                source,
                output,
                target,
                input_port,
                PORT_TYPES[TASKS[workload][source]]["outputs"][output],
            )
            for source, output, target, input_port in EDGES[workload]
        ),
    )


def initial_profiles(workload: str = "smolvla") -> ProfileCatalog:
    """Scheduling priors only; these are never reported as measured GPU usage."""
    if workload not in TASKS:
        raise ValueError("workload must be cuda or smolvla")
    return ProfileCatalog(
        [
            ExecutionProfile(
                task_type=task_type,
                task_class=TaskClass.REALTIME_OFFLOADABLE,
                node_kind=kind,
                model_variant=MODEL_ID if task_type == "hil_vla_infer" else task_type,
                input_shape="bounded_single_observation_or_matrix",
                precision="workload_configured",
                batch_size=1,
                p50_ms=1000.0 if task_type in GPU_TASK_TYPES else 100.0,
                p95_ms=10_000.0 if task_type in GPU_TASK_TYPES else 200.0,
                p99_ms=60_000.0 if task_type in GPU_TASK_TYPES else 500.0,
                throughput_per_s=1.0 if task_type in GPU_TASK_TYPES else 10.0,
                peak_memory_mb=4096.0
                if task_type == "hil_vla_infer"
                else 1024.0
                if task_type == "hil_cuda_smoke"
                else 256.0,
                energy_j=0.0,
                output_size_mb=1.5 if task_type == "hil_vla_observe" else 0.2,
                cpu_units=1.0,
                gpu_units=1.0 if task_type in GPU_TASK_TYPES else 0.0,
                provenance=PROFILE_SOURCE,
            )
            for task_type in TASKS[workload].values()
            for kind in (NodeKind.ROBOT, NodeKind.EDGE)
        ]
    )


def _positive_measurements(values) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            for value in values
        )
    )


def verify_gpu_payload(
    payload: dict, workload: str, observation: dict | None = None
) -> dict:
    """Check returned measurement evidence independently of GPU registration.

    These are trusted-LAN worker attestations, not cryptographic hardware
    attestation. CUDA device labels alone cannot establish successful execution.
    """
    measurement = payload.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("GPU output has no CUDA measurement evidence")
    device = measurement.get("device")
    if (
        not isinstance(device, str)
        or not device.startswith("cuda:")
        or not device[5:].isdigit()
    ):
        raise ValueError("GPU measurement did not execute on a CUDA device")
    events, wall = (
        measurement.get("cuda_event_ms"),
        measurement.get("synchronized_wall_ms"),
    )
    if (
        not _positive_measurements(events)
        or not _positive_measurements(wall)
        or len(events) != len(wall)
    ):
        raise ValueError(
            "GPU output lacks finite positive CUDA and synchronized wall timings"
        )
    memory = measurement.get("peak_memory_allocated_bytes")
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise ValueError("GPU output lacks measured CUDA allocation")
    for name in ("device_name", "torch_version", "cuda_version"):
        if (
            not isinstance(measurement.get(name), str)
            or not measurement[name].strip()
            or measurement[name] == "None"
        ):
            raise ValueError(f"GPU measurement is missing {name}")
    repeats = measurement.get("repeats")
    if type(repeats) is not int or repeats != len(events):
        raise ValueError("GPU measurement count does not match its recorded repeats")
    if measurement.get("output_device") != device:
        raise ValueError(
            "GPU output tensor was not produced on the selected CUDA device"
        )
    input_devices = measurement.get("input_devices")
    if isinstance(input_devices, dict):
        input_devices = list(input_devices.values())
    if (
        not isinstance(input_devices, list)
        or not input_devices
        or any(item != device for item in input_devices)
    ):
        raise ValueError(
            "GPU input tensors were not placed on the selected CUDA device"
        )
    if workload == "smolvla":
        if measurement.get("timing_scope") != "policy_predict_action_chunk_only":
            raise ValueError("SmolVLA timing scope does not identify policy inference")
        if payload.get("schema") != "mars.vla.actions.v1" or observation is None:
            raise ValueError("missing SmolVLA action or observation payload")
        if payload.get("source_hashes", {}).get("observation") != digest_bytes(
            canonical_json(observation)
        ):
            raise ValueError(
                "SmolVLA actions do not identify the transported observation"
            )
        devices = measurement.get("parameter_devices")
        if (
            not isinstance(devices, list)
            or not devices
            or any(item != device for item in devices)
        ):
            raise ValueError("SmolVLA model parameters were not placed on CUDA")
        chunk = payload.get("action_chunk")
        shape = payload.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in shape
            )
            or not isinstance(chunk, list)
            or len(chunk) != shape[0]
        ):
            raise ValueError("SmolVLA output action shape is invalid")
        for action in chunk:
            if (
                not isinstance(action, list)
                or len(action) != shape[1]
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in action
                )
            ):
                raise ValueError(
                    "SmolVLA action values are non-finite or have the wrong dimension"
                )
        if payload.get("action_sha256") != digest_bytes(canonical_json(chunk)):
            raise ValueError("SmolVLA action checksum does not match its values")
        model = payload.get("model")
        if (
            not isinstance(model, dict)
            or model.get("weights_verified") is not True
            or model.get("strict_load") is not True
        ):
            raise ValueError("SmolVLA output lacks verified pretrained model identity")
        if (
            model.get("policy") != {"repo_id": MODEL_ID, "revision": POLICY_REVISION}
            or model.get("vlm") != {"repo_id": VLM_ID, "revision": VLM_REVISION}
            or not re.fullmatch(r"[0-9a-f]{64}", str(model.get("manifest_sha256", "")))
            or shape != [model.get("chunk_size"), model.get("action_dim")]
        ):
            raise ValueError("SmolVLA model identity or feature shape is inconsistent")
    elif workload == "cuda":
        if measurement.get("timing_scope") != "torch_matmul_only":
            raise ValueError(
                "CUDA timing scope does not identify matrix multiplication"
            )
        size, seed = payload.get("matrix_size"), payload.get("seed")
        if (
            payload.get("schema") != "mars.cuda.result.v1"
            or payload.get("operation") != "float32_dense_matrix_multiply"
            or payload.get("dtype") != "float32"
            or type(size) is not int
            or not 64 <= size <= 4096
            or type(seed) is not int
            or not 0 <= seed < 2**32
            or payload.get("full_matrix_reference_match") is not True
            or payload.get("max_abs_error") != 0
        ):
            raise ValueError(
                "CUDA matrix computation did not pass its full reference check"
            )
        indexes = sorted({0, 1, size // 2, size - 1})
        samples = [
            {
                "row": row,
                "column": column,
                "value": float(
                    size * ((row + seed) % 17 - 8) * ((column + seed) % 13 - 6)
                ),
            }
            for row in indexes
            for column in indexes
        ]
        if payload.get("samples") != samples or payload.get(
            "sample_sha256"
        ) != digest_bytes(canonical_json(samples)):
            raise ValueError(
                "CUDA matrix samples do not match the independent exact reference"
            )
    else:
        raise ValueError("workload must be cuda or smolvla")
    return dict(measurement)


async def run_vla_loop(
    endpoints: dict[str, str],
    *,
    workload: str = "smolvla",
    gpu_agent: str = "robot_1",
    seed: int = 19,
    artifact_directory: str | Path = ".mars-hil/vla-received",
    workflow_timeout_seconds: float = 600.0,
    task_completion_timeout_seconds: float = 300.0,
    require_distinct_hosts: bool = False,
) -> dict:
    _positive_timeout(workflow_timeout_seconds, "workflow timeout")
    _positive_timeout(task_completion_timeout_seconds, "task completion timeout")
    if task_completion_timeout_seconds > workflow_timeout_seconds:
        raise ValueError("task completion timeout must not exceed workflow timeout")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if set(endpoints) != {"robot_1", "edge_pc"}:
        raise ValueError("CUDA/VLA loop requires exactly robot_1 and edge_pc endpoints")
    workflow = vla_workflow(
        workload, gpu_agent, deadline_ms=workflow_timeout_seconds * 1000
    )
    io_agent = _io_agent(gpu_agent)
    gpu_task = "infer" if workload == "smolvla" else "smoke"
    runtime = GrpcRuntimeAdapter(
        endpoints, completion_timeout_seconds=task_completion_timeout_seconds
    )
    files = ArtifactFiles(artifact_directory)
    evidence = {
        "schema": "mars.hil.vla_run.v1",
        "workflow_id": workflow.workflow_id,
        "workload": workload,
        "seed": seed,
        "gpu_agent": gpu_agent,
        "io_agent": io_agent,
        "endpoints": endpoints,
        "status": "failed",
        "scope": "unverified",
        "business_execution": "cuda_inference_and_cpu_io",
        "physical_actuation": False,
        "control_success_tested": False,
        "gpu_tested": False,
        "gpu_execution": None,
        "energy_j": None,
        "timeouts_seconds": {
            "workflow": workflow_timeout_seconds,
            "task_completion": task_completion_timeout_seconds,
        },
        "planning_assumptions": {
            "profiles": PROFILE_SOURCE,
            "peak_host_memory_mb": {
                "hil_vla_infer": 4096.0,
                "hil_cuda_smoke": 1024.0,
                "cpu_io": 256.0,
            },
            "peak_memory_is": "unmeasured_host_memory_prior_not_CUDA_allocation",
            "gpu_units": "one_exclusive_worker_slot_on_selected_device",
            "link_bandwidth_mbps": 100.0,
            "link_latency_ms": 0.0,
            "scheduler_gpu_utilization": "unmeasured_v1_zero_placeholder",
            "scheduler_timestamps": "logical_dispatch_anchor_plus_measured_elapsed",
            "scheduler_communication_metrics": "estimated_not_measured",
        },
        "dag": {"tasks": TASKS[workload], "edges": EDGES[workload]},
        "artifacts": [],
        "error": None,
    }
    started = perf_counter()
    try:

        async def execute():
            inventory = await runtime.start(0)
            nodes = {node.node_id: node for node in inventory.nodes}
            for task in workflow.tasks:
                node = nodes[task.spec.placement_constraints.pinned_node_id]
                if not set(
                    task.spec.placement_constraints.required_capabilities
                ).issubset(node.capabilities):
                    raise ValueError(
                        f"{node.node_id} lacks required executor capabilities for {task.task_id}"
                    )
                if node.kind != (
                    NodeKind.ROBOT if node.node_id == "robot_1" else NodeKind.EDGE
                ):
                    raise ValueError(f"{node.node_id} has wrong node kind")
                if task.spec.gpu_demand and (
                    not math.isfinite(node.gpu_capacity) or node.gpu_capacity < 1
                ):
                    raise ValueError(
                        f"{node.node_id} did not advertise verified CUDA capacity"
                    )
                if (
                    task.spec.model_requirement
                    and task.spec.model_requirement not in node.supported_models
                ):
                    raise ValueError(f"{node.node_id} did not advertise {MODEL_ID}")
            links = tuple(
                LinkSpec(f"vla:{source}:{target}", source, target, 100.0, 0.0)
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
                profile_catalog=initial_profiles(workload),
            )
            return await coordinator.run_async(
                workflow, algorithm="heuristic", seed=seed, max_attempts=1
            )

        report = await asyncio.wait_for(execute(), timeout=workflow_timeout_seconds)
        evidence["workflow_wall_elapsed_ms"] = (perf_counter() - started) * 1000
        evidence["coordinator_report"] = report.as_dict()
        for task in report.task_results:
            for output in task["outputs"]:
                envelope, _ = await fetch_artifact(
                    ArtifactRef(**output),
                    agent_id="coordinator",
                    files=files,
                    peers=endpoints,
                )
                if envelope.get("workflow_id") != workflow.workflow_id:
                    raise ValueError(
                        "received execution artifact belongs to a different workflow"
                    )
                evidence["artifacts"].append(
                    {"reference": output, "envelope": envelope}
                )
        if report.workflow["state"] != "succeeded":
            failures = []
            for task in report.task_results:
                if task["state"] != "succeeded":
                    attempts = task.get("attempts", ())
                    reason = attempts[-1].get("error_code", "") if attempts else ""
                    failures.append(f"{task['task_id']}: {task['state']} ({reason})")
            raise RuntimeError("workflow did not succeed: " + "; ".join(failures))
        artifacts = {
            item["envelope"]["producer_task_id"]: item["envelope"]
            for item in evidence["artifacts"]
        }
        if set(artifacts) != set(TASKS[workload]) or len(artifacts) != len(
            evidence["artifacts"]
        ):
            raise ValueError("missing or duplicate execution artifacts")
        records = {
            task_id: artifact["execution"] for task_id, artifact in artifacts.items()
        }
        for task_id, record in records.items():
            expected_node = gpu_agent if task_id == gpu_task else io_agent
            expected_mode = "real_cuda" if task_id == gpu_task else "real_cpu"
            if (
                record.get("agent_id") != expected_node
                or record.get("task_id") != task_id
                or record.get("task_type") != TASKS[workload][task_id]
                or record.get("execution_mode") != expected_mode
            ):
                raise ValueError(
                    f"{task_id} execution evidence does not match its planned executor"
                )
        evidence["executions"] = list(records.values())
        host_keys = {
            (record["host"]["hostname"], record["host"]["architecture"])
            for record in records.values()
        }
        evidence["executing_host_count"] = len(host_keys)
        evidence["executing_node_ids"] = sorted(
            {record["agent_id"] for record in records.values()}
        )
        evidence["remote_input_bytes"] = sum(
            record["remote_input_bytes"] for record in records.values()
        )
        evidence["worker_elapsed_ms"] = sum(
            record["worker_elapsed_ms"] for record in records.values()
        )
        validation = artifacts["validate"]["payload"]
        if validation.get("valid") is not True:
            raise ValueError("missing successful business output validation")
        evidence["validation"] = validation
        observation = artifacts["observe"]["payload"] if workload == "smolvla" else None
        payload = artifacts[gpu_task]["payload"]
        measurement = verify_gpu_payload(payload, workload, observation)
        measured_seed = (
            payload.get("inference", {}).get("seed")
            if workload == "smolvla"
            else payload.get("seed")
        )
        if measured_seed != seed:
            raise ValueError("GPU output does not match this run's requested seed")
        expected_hashes = (
            {
                "actions": digest_bytes(canonical_json(payload)),
                "observation": digest_bytes(canonical_json(observation)),
            }
            if workload == "smolvla"
            else {"cuda_result": digest_bytes(canonical_json(payload))}
        )
        if validation.get("source_hashes") != expected_hashes:
            raise ValueError(
                "business validation did not consume the returned computation artifacts"
            )
        if records["validate"]["remote_input_bytes"] <= 0 or (
            workload == "smolvla" and records["infer"]["remote_input_bytes"] <= 0
        ):
            raise ValueError(
                "run did not prove the required cross-node artifact transfers"
            )
        evidence["gpu_execution"] = {
            "task_id": gpu_task,
            "agent_id": gpu_agent,
            "measurement": measurement,
        }
        if workload == "smolvla":
            evidence["gpu_execution"].update(
                model=payload["model"],
                action_shape=payload["shape"],
                inference=payload.get("inference"),
            )
            evidence["observation_source"] = observation.get("provenance")
        evidence["gpu_tested"] = True
        evidence["scope"] = (
            "cross_host_cuda_execution"
            if len(host_keys) > 1
            else "same_host_cuda_execution"
        )
        if require_distinct_hosts and len(host_keys) < 2:
            raise ValueError(
                "both Agents executed on the same reported host; physical two-host test not proven"
            )
        evidence["status"] = "succeeded"
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["diagnostic_hint"] = (
            "Inspect the failing Agent log for worker stderr; model loading, CUDA allocation and peer transfer failures remain fatal."
        )
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
        await runtime.close()
        evidence["total_wall_elapsed_ms"] = (perf_counter() - started) * 1000
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", action="append", required=True, metavar="NODE=HOST:PORT"
    )
    parser.add_argument("--workload", choices=("cuda", "smolvla"), default="smolvla")
    parser.add_argument(
        "--gpu-agent", choices=("robot_1", "edge_pc"), default="robot_1"
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow-timeout", type=float, default=600.0)
    parser.add_argument("--task-completion-timeout", type=float, default=300.0)
    parser.add_argument("--require-distinct-hosts", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "output already exists; choose a new run filename to preserve evidence"
        )
    try:
        report = asyncio.run(
            run_vla_loop(
                parse_endpoints(args.agent),
                workload=args.workload,
                gpu_agent=args.gpu_agent,
                seed=args.seed,
                artifact_directory=args.output.parent / "received-artifacts",
                workflow_timeout_seconds=args.workflow_timeout,
                task_completion_timeout_seconds=args.task_completion_timeout,
                require_distinct_hosts=args.require_distinct_hosts,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            output.write(canonical_json(report) + b"\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"invalid CUDA/VLA configuration: {exc}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "report": str(args.output),
                "error": report["error"],
                "gpu_tested": report["gpu_tested"],
            }
        )
    )
    raise SystemExit(0 if report["status"] == "succeeded" else 1)


if __name__ == "__main__":
    main()
