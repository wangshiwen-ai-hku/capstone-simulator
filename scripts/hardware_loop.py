"""Run the real-computation navigation DAG through MARS and save HIL evidence.

No web server, LLM, fake scene catalog or sensor hardware is required.
Run from the checkout root: python -m scripts.hardware_loop --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from agent.artifacts import ArtifactFiles, canonical_json, fetch_artifact
from agent.endpoints import parse_endpoints
from examples.hardware_workloads import PORT_TYPES
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
    "sense": "hil_sensor",
    "map": "hil_mapping",
    "plan": "hil_planning",
    "validate": "hil_validation",
}
EDGES = (
    ("sense", "observations", "map", "observations"),
    ("map", "map", "plan", "map"),
    ("map", "map", "validate", "map"),
    ("plan", "trajectory", "validate", "trajectory"),
    ("sense", "truth", "validate", "truth"),
)


def navigation_workflow(
    placement: str = "split", workflow_id: str | None = None
) -> WorkflowSpec:
    if placement not in {"split", "orin", "edge", "auto"}:
        raise ValueError("unknown HIL placement")
    workflow_id = workflow_id or f"hil-{uuid4().hex}"
    tasks = []
    for index, (task_id, task_type) in enumerate(TASKS.items()):
        pinned = (
            "robot_1"
            if placement == "orin"
            else "edge_pc"
            if placement == "edge"
            else "robot_1"
            if task_id in {"sense", "validate"}
            else "edge_pc"
            if placement == "split"
            else ""
        )
        contract = PORT_TYPES[task_type]
        tasks.append(
            TaskInstance(
                task_id=task_id,
                workflow_id=workflow_id,
                name=task_type,
                source_node_id="robot_1",
                spec=TaskSpec(
                    task_type=task_type,
                    task_class=TaskClass.REALTIME_OFFLOADABLE,
                    compute_demand=0.1,
                    gpu_demand=0.0,
                    latency_budget_ms=120_000,
                    output_size_mb=0.2,
                    dominant_resource=ResourceClass.CPU,
                    input_ports=tuple(
                        DataPort(name, kind)
                        for name, kind in contract["inputs"].items()
                    ),
                    output_ports=tuple(
                        DataPort(name, kind)
                        for name, kind in contract["outputs"].items()
                    ),
                    placement_constraints=PlacementConstraints(
                        pinned_node_id=pinned,
                        allowed_node_kinds=(NodeKind.ROBOT, NodeKind.EDGE),
                        required_capabilities=("hil_navigation_v1",),
                        allow_fallback=False,
                    ),
                ),
                dependency_task_ids=tuple(
                    dict.fromkeys(
                        source for source, _, target, _ in EDGES if target == task_id
                    )
                ),
                stage_index=index,
                deadline_time_ms=120_000,
            )
        )
    return WorkflowSpec(
        workflow_id,
        tuple(tasks),
        deadline_time_ms=120_000,
        failure_policy=FailurePolicy.FAIL_FAST,
        metadata={
            "purpose": "offline_hardware_validation",
            "sensor_source": "synthetic",
            "placement": placement,
        },
        data_edges=tuple(
            DataEdge(
                source,
                output,
                target,
                input_port,
                PORT_TYPES[TASKS[source]]["outputs"][output],
            )
            for source, output, target, input_port in EDGES
        ),
    )


def initial_profiles() -> ProfileCatalog:
    """Bootstrap scheduling estimates, explicitly NOT hardware measurements."""
    return ProfileCatalog(
        [
            ExecutionProfile(
                task_type=task_type,
                task_class=TaskClass.REALTIME_OFFLOADABLE,
                node_kind=kind,
                model_variant="classical_navigation_cpu_v1",
                input_shape="bounded_synthetic_range_survey",
                precision="python_float64",
                batch_size=1,
                p50_ms=100.0,
                p95_ms=200.0,
                p99_ms=500.0,
                throughput_per_s=10.0,
                peak_memory_mb=32.0,
                energy_j=0.0,
                output_size_mb=0.2,
                cpu_units=1.0,
                gpu_units=0.0,
                provenance="unmeasured_hil_bootstrap_prior",
            )
            for task_type in TASKS.values()
            for kind in (NodeKind.ROBOT, NodeKind.EDGE)
        ]
    )


async def run_hardware_loop(
    endpoints: dict[str, str],
    *,
    placement: str = "split",
    seed: int = 19,
    artifact_directory: str | Path = ".mars-hil/received",
    workflow_timeout_seconds: float = 120.0,
    require_distinct_hosts: bool = False,
) -> dict:
    if not math.isfinite(workflow_timeout_seconds) or workflow_timeout_seconds <= 0:
        raise ValueError("workflow timeout must be finite and positive")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    required = {"robot_1"} if placement == "orin" else {"robot_1", "edge_pc"}
    if not required.issubset(endpoints) or not set(endpoints).issubset(
        {"robot_1", "edge_pc"}
    ):
        raise ValueError(f"placement {placement} requires endpoints {sorted(required)}")
    if require_distinct_hosts and placement in {"orin", "edge"}:
        raise ValueError("distinct executing hosts requires split or auto placement")
    workflow = navigation_workflow(placement)
    runtime = GrpcRuntimeAdapter(
        endpoints,
        completion_timeout_seconds=min(60.0, workflow_timeout_seconds),
    )
    files = ArtifactFiles(artifact_directory)
    evidence = {
        "schema": "mars.hil.run.v1",
        "workflow_id": workflow.workflow_id,
        "seed": seed,
        "placement": placement,
        "endpoints": endpoints,
        "status": "failed",
        "scope": "unverified",
        "sensor_source": "synthetic_known_pose_range_survey",
        "business_execution": "real_cpu",
        "physical_actuation": False,
        "gpu_tested": False,
        "energy_j": None,
        "planning_assumptions": {
            "profiles": "unmeasured_hil_bootstrap_prior",
            "link_bandwidth_mbps": 100.0,
            "link_latency_ms": 0.0,
            "scheduler_timestamps": "logical_dispatch_anchor_plus_measured_elapsed",
            "scheduler_communication_metrics": "estimated_not_measured",
        },
        "dag": {"tasks": TASKS, "edges": EDGES},
        "artifacts": [],
        "error": None,
    }
    started = perf_counter()
    try:

        async def execute():
            inventory = await runtime.start(0)
            if any(
                "hil_navigation_v1" not in node.capabilities for node in inventory.nodes
            ):
                raise ValueError(
                    "all endpoints must run --executor navigation, not Mock Agents"
                )
            for node in inventory.nodes:
                expected_kind = (
                    NodeKind.ROBOT if node.node_id == "robot_1" else NodeKind.EDGE
                )
                if node.kind != expected_kind:
                    raise ValueError(f"{node.node_id} has wrong node kind")
            links = tuple(
                LinkSpec(f"hil:{source}:{target}", source, target, 100.0, 0.0)
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

        report = await asyncio.wait_for(execute(), timeout=workflow_timeout_seconds)
        evidence["workflow_wall_elapsed_ms"] = (perf_counter() - started) * 1000
        evidence["coordinator_report"] = report.as_dict()
        evidence["final_node_observations"] = [
            {
                "node_id": snapshot.node_id,
                "cpu_utilization_ratio": snapshot.cpu_util,
                "memory_utilization_ratio": snapshot.memory_util,
                "online": snapshot.online,
            }
            for snapshot in runtime.snapshots
        ]
        if report.workflow["state"] != "succeeded":
            failures = []
            for task in report.task_results:
                if task["state"] == "succeeded":
                    continue
                attempts = task.get("attempts", ())
                reason = attempts[-1].get("error_code", "") if attempts else ""
                failures.append(f"{task['task_id']}: {task['state']} ({reason})")
            raise RuntimeError("workflow did not succeed: " + "; ".join(failures))
        for task in report.task_results:
            for output in task["outputs"]:
                reference = ArtifactRef(**output)
                envelope, _ = await fetch_artifact(
                    reference,
                    agent_id="coordinator",
                    files=files,
                    peers=endpoints,
                )
                evidence["artifacts"].append(
                    {"reference": output, "envelope": envelope}
                )
        records = {
            item["envelope"]["producer_task_id"]: item["envelope"]["execution"]
            for item in evidence["artifacts"]
        }
        if set(records) != set(TASKS):
            raise RuntimeError("missing real execution evidence for one or more stages")
        executions = list(records.values())
        host_keys = {
            (item["host"]["hostname"], item["host"]["architecture"])
            for item in executions
        }
        evidence["executions"] = executions
        evidence["executing_host_count"] = len(host_keys)
        evidence["executing_node_ids"] = sorted(
            {item["agent_id"] for item in executions}
        )
        evidence["scope"] = (
            "cross_host_cpu_execution"
            if len(host_keys) > 1
            else "same_host_cpu_execution"
        )
        evidence["remote_input_bytes"] = sum(
            item["remote_input_bytes"] for item in executions
        )
        evidence["worker_elapsed_ms"] = sum(
            item["worker_elapsed_ms"] for item in executions
        )
        validations = [
            item["envelope"]["payload"]
            for item in evidence["artifacts"]
            if item["reference"]["producer_task_id"] == "validate"
        ]
        if len(validations) != 1 or validations[0].get("valid") is not True:
            raise RuntimeError("missing successful business trajectory validation")
        evidence["validation"] = validations[0]
        if placement == "split" and (
            evidence["executing_node_ids"] != ["edge_pc", "robot_1"]
            or records["map"]["remote_input_bytes"] <= 0
            or records["validate"]["remote_input_bytes"] <= 0
        ):
            raise RuntimeError(
                "split run did not prove bidirectional artifact transfer"
            )
        if require_distinct_hosts and len(host_keys) < 2:
            raise RuntimeError(
                "both Agents executed on the same reported host; physical two-host test not proven"
            )
        evidence["status"] = "succeeded"
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        await runtime.close()
        evidence["total_wall_elapsed_ms"] = (perf_counter() - started) * 1000
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", action="append", required=True, metavar="NODE=HOST:PORT"
    )
    parser.add_argument(
        "--placement", choices=("split", "orin", "edge", "auto"), default="split"
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow-timeout", type=float, default=120.0)
    parser.add_argument("--require-distinct-hosts", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "output already exists; choose a new run filename to preserve evidence"
        )
    try:
        endpoints = parse_endpoints(args.agent)
        report = asyncio.run(
            run_hardware_loop(
                endpoints,
                placement=args.placement,
                seed=args.seed,
                artifact_directory=args.output.parent / "received-artifacts",
                workflow_timeout_seconds=args.workflow_timeout,
                require_distinct_hosts=args.require_distinct_hosts,
            )
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"invalid HIL configuration: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidentally overwriting a prior run.
    with args.output.open("xb") as output:
        output.write(canonical_json(report) + b"\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "report": str(args.output),
                "error": report["error"],
                "remote_input_bytes": report.get("remote_input_bytes", 0),
            }
        )
    )
    raise SystemExit(0 if report["status"] == "succeeded" else 1)


if __name__ == "__main__":
    main()
