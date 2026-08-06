"""Run the fixed Stage 4-6 MARS benchmark and write every result under doc/."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys
from statistics import mean, stdev
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from backend.app.schemas import BenchmarkScene
from mars.domain.task import TaskClass
from mars.domain.execution import task_resource_demand
from mars.domain.topology import NodeKind
from mars.engine import run_workflow_simulation
from mars.optimizers import BinaryOffloadOptimizer, OptimizerRegistry
from mars.synthetic_workloads import load_default_synthetic_workloads


DOC = ROOT / "doc"
SEED = 20260731
SEEDS = (7, 17, 27, 37, 47)
FORMAL_BETA = 0.01
BETA_SENSITIVITY = (0.0001, 0.001, 0.01, 0.1)

HARDWARE = {
    "jetson": {
        "count": 4,
        "label": "Jetson AGX Orin 32GB",
        "power_mode": "40W fixed",
        "cpu_capacity": 8.0,
        "gpu_capacity": 1.0,
        "memory_gb": 32.0,
        "max_concurrency": 1,
        "initial_utilization": {"cpu": 0.10, "gpu": 0.08, "memory": 0.12},
        "field_provenance": {
            "label": "reference_data: NVIDIA Jetson AGX Orin 32GB technical brief",
            "power_mode": "reference_data: NVIDIA lists 15W-40W; experiment fixes 40W",
            "memory_gb": "reference_data: NVIDIA lists 32GB LPDDR5",
            "cpu_capacity": "synthetic_data: MARS capacity unit; numerically aligned to the official 8-core CPU",
            "gpu_capacity": "synthetic_data: normalized MARS unit, not CUDA-core count or utilization percentage",
            "max_concurrency": "synthetic_data: conservative experiment setting",
            "initial_utilization": "synthetic_data: experiment initial condition"
        },
    },
    "edge": {
        "count": 1,
        "label": "x86 workstation + discrete NVIDIA GPU",
        "power_mode": "mains powered",
        "cpu_capacity": 16.0,
        "gpu_capacity": 4.0,
        "memory_gb": 64.0,
        "max_concurrency": 4,
        "initial_utilization": {"cpu": 0.20, "gpu": 0.25, "memory": 0.18},
        "field_provenance": {
            "all_fields": "synthetic_data: the project does not specify an Edge PC CPU/GPU model"
        },
    },
    "network": {
        "robot_edge": "Wi-Fi 6 / LAN simulation",
        "bandwidth_mbps": 300.0,
        "base_latency_ms": 4.0,
        "available_bandwidth_mbps": 240.0,
        "jitter_ms": 1.0,
        "packet_loss_rate": 0.002,
        "field_provenance": {
            "all_fields": "synthetic_data: fixed LAN/Wi-Fi experiment assumptions, not measured values"
        },
    },
}

SCENARIOS = (
    {
        "id": "warehouse_navigation",
        "name": "仓库导航",
        "description": "4台机器人同时做急停、定位、目标检测、环境理解和局部规划。",
        "tasks_per_robot": (
            ("emergency_stop", ()),
            ("localization", ("emergency_stop",)),
            ("object_detection", ("localization",)),
            ("environment_understanding", ("localization", "object_detection")),
            ("local_planning", ("environment_understanding",)),
        ),
        "deadline_ms": 850.0,
    },
    {
        "id": "multi_robot_mapping",
        "name": "多机器人地图构建",
        "description": "每台机器人完成定位、语义分割和压缩，再各自执行地图融合候选。",
        "tasks_per_robot": (
            ("localization", ()),
            ("semantic_segmentation", ("localization",)),
            ("data_compression", ("semantic_segmentation",)),
            ("map_fusion", ("data_compression",)),
        ),
        "deadline_ms": 1200.0,
    },
    {
        "id": "visual_inspection",
        "name": "视觉巡检",
        "description": "检测后进行结果复核，最后的本地控制禁止卸载。",
        "tasks_per_robot": (
            ("object_detection", ()),
            ("result_verification", ("object_detection",)),
            ("local_control", ("result_verification",)),
        ),
        "deadline_ms": 900.0,
    },
    {
        "id": "high_load",
        "name": "高负载压力",
        "description": "4台机器人同时提交两条视觉链，Edge保持较高初始利用率。",
        "tasks_per_robot": (
            ("object_detection", ()),
            ("semantic_segmentation", ("object_detection",)),
            ("environment_understanding", ("object_detection", "semantic_segmentation")),
            ("result_verification", ("environment_understanding",)),
        ),
        "deadline_ms": 650.0,
        "edge_utilization": {"cpu": 0.45, "gpu": 0.55, "memory": 0.35},
        "available_bandwidth_mbps": 120.0,
    },
)

METHODS = (
    ("binary_offload", None, "精确枚举：冻结的二元目标函数"),
    ("heuristic", "local_first", "启发式：本地优先"),
    ("heuristic", "edge_first", "启发式：Edge优先"),
    ("heuristic", "dag_deadline", "启发式：DAG截止时间策略"),
)

def build_scene(
    scenario: dict[str, object],
    catalog,
) -> BenchmarkScene:
    """Expand one experiment template into the platform's web scene model."""

    node_rows = [
        {
            "id": f"jetson-{index}",
            "kind": "robot",
            "display_name": f"Jetson AGX Orin {index}",
            "architecture": "aarch64-jetson-agx-orin-32gb-40w",
            "cpu_capacity": 8.0,
            "gpu_capacity": 1.0,
            "memory_gb": 32.0,
            "bandwidth_mbps": 300.0,
            "base_latency_ms": 2.0,
            "battery_wh": 100.0,
            "safety_capable": True,
            "capabilities": ["cuda", "tensorrt", "local_safety"],
            "max_concurrency": 1,
        }
        for index in range(1, 5)
    ]
    node_rows.append(
        {
            "id": "edge-1",
            "kind": "edge",
            "display_name": "Edge PC",
            "architecture": "x86_64-discrete-nvidia-gpu",
            "cpu_capacity": 16.0,
            "gpu_capacity": 4.0,
            "memory_gb": 64.0,
            "bandwidth_mbps": 1000.0,
            "base_latency_ms": 1.0,
            "safety_capable": False,
            "capabilities": ["cuda", "tensorrt"],
            "max_concurrency": 4,
        }
    )
    edge_util = scenario.get(
        "edge_utilization", HARDWARE["edge"]["initial_utilization"]
    )
    resource_rows = [
        {
            "node_id": node["id"],
            "cpu_util": edge_util["cpu"] if node["kind"] == "edge" else 0.10,
            "gpu_util": edge_util["gpu"] if node["kind"] == "edge" else 0.08,
            "memory_util": edge_util["memory"] if node["kind"] == "edge" else 0.12,
            "temperature_c": 45.0 if node["kind"] == "edge" else 52.0,
            "power_w": 110.0 if node["kind"] == "edge" else 40.0,
            "network_latency_ms": 1.0 if node["kind"] == "edge" else 4.0,
            "online": True,
        }
        for node in node_rows
    ]
    available_bandwidth = float(
        scenario.get(
            "available_bandwidth_mbps",
            HARDWARE["network"]["available_bandwidth_mbps"],
        )
    )
    link_rows = []
    link_snapshot_rows = []
    for source in node_rows:
        for target in node_rows:
            if source["id"] == target["id"]:
                continue
            robot_edge = "edge" in {source["kind"], target["kind"]}
            link_id = f"{source['id']}-to-{target['id']}"
            link_rows.append(
                {
                    "id": link_id,
                    "source_node_id": source["id"],
                    "target_node_id": target["id"],
                    "bandwidth_mbps": 300.0 if robot_edge else 150.0,
                    "base_latency_ms": 4.0 if robot_edge else 7.0,
                }
            )
            link_snapshot_rows.append(
                {
                    "link_id": link_id,
                    "available_bandwidth_mbps": (
                        available_bandwidth
                        if robot_edge
                        else min(100.0, available_bandwidth)
                    ),
                    "latency_ms": 4.0 if robot_edge else 7.0,
                    "jitter_ms": 1.0,
                    "packet_loss_rate": 0.002,
                    "online": True,
                }
            )

    task_rows = []
    workflow_id = f"{scenario['id']}--concurrent-batch"
    for robot_index in range(1, 5):
        source = f"jetson-{robot_index}"
        for instance_index, arrival_ms in enumerate(
            (0.0, float(scenario["deadline_ms"]) * 0.4),
            start=1,
        ):
            logical_workflow_id = (
                f"{scenario['id']}--robot-{robot_index}--run-{instance_index}"
            )
            logical_prefix = f"wf{robot_index}_{instance_index}"
            created: dict[str, str] = {}
            for stage, (task_type, dependency_names) in enumerate(
                scenario["tasks_per_robot"]
            ):
                workload = catalog.get(task_type)
                spec = workload.to_task_spec()
                if workload.task_class is TaskClass.LOCAL_SAFETY:
                    placement = {
                        "pin_to_source": True,
                        "allowed_node_kinds": ["robot"],
                        "preferred_node_kinds": ["robot"],
                        "required_capabilities": ["local_safety"],
                        "safety_required": True,
                        "allow_fallback": False,
                        "stateful": True,
                        "idempotent": False,
                    }
                else:
                    placement = {
                        "allowed_node_kinds": ["edge"],
                        "allow_source_node": True,
                        "allow_other_robots": False,
                        "allow_fallback": True,
                    }
                task_id = f"{logical_prefix}--{task_type}"
                dependencies = tuple(created[name] for name in dependency_names)
                task_rows.append(
                    {
                        "id": task_id,
                        "name": (
                            f"{workload.display_name} R{robot_index} "
                            f"[{logical_workflow_id}]"
                        ),
                        "source_robot_id": source,
                        "task_type": task_type,
                        "task_class": workload.task_class,
                        "priority": (
                            5 if workload.task_class is TaskClass.LOCAL_SAFETY else 3
                        ),
                        "compute_demand": spec.compute_demand,
                        "gpu_demand": spec.gpu_demand,
                        "latency_budget_ms": spec.latency_budget_ms,
                        "model_requirement": spec.model_requirement,
                        "data_size_mb": spec.input_size_mb,
                        "output_size_mb": spec.output_size_mb,
                        "bandwidth_requirement_mbps": spec.bandwidth_requirement_mbps,
                        "energy_budget_j": spec.energy_budget_j,
                        "placement_constraints": placement,
                        "result_verification": "synthetic profile outcome",
                        "arrival_time_ms": arrival_ms,
                        "deadline_ms": arrival_ms + float(scenario["deadline_ms"]),
                        "dependencies": list(dependencies),
                        "stage_index": stage,
                        "expected_accuracy": workload.profile_for("orin").accuracy.typical,
                        "input_ports": [
                            {"name": port.name, "message_type": port.message_type}
                            for port in spec.input_ports
                        ],
                        "output_ports": [
                            {"name": port.name, "message_type": port.message_type}
                            for port in spec.output_ports
                        ],
                    }
                )
                created[task_type] = task_id
    return BenchmarkScene(
        id=str(scenario["id"]),
        title=str(scenario["name"]),
        natural_language_description=str(scenario["description"]),
        scenario_type=str(scenario["id"]),
        difficulty="stress" if scenario["id"] == "high_load" else "medium",
        nodes=node_rows,
        initial_resources=resource_rows,
        links=link_rows,
        link_snapshots=link_snapshot_rows,
        tasks=task_rows,
        workflow_id=workflow_id,
        workflow_deadline_ms=float(scenario["deadline_ms"]),
        generation_source="deterministic",
        generation_note=(
            "Eight independent DAGs share one MARS WorkflowSpec because "
            "CentralCoordinator accepts one workflow per run."
        ),
    )


def profile_summary(catalog):
    model_names = {
        "object_detection": "YOLOv8n TensorRT FP16, batch=1, 640x640 JPEG",
        "semantic_segmentation": "lightweight segmentation TensorRT FP16, batch=1, 640x640 JPEG",
        "localization": "synthetic visual/sensor localization",
        "environment_understanding": "synthetic scene encoder TensorRT FP16",
        "local_planning": "synthetic local planner",
        "map_fusion": "synthetic multi-map fusion",
        "data_compression": "synthetic image/map codec",
        "result_verification": "synthetic VLM verifier",
        "emergency_stop": "rule/state-machine safety guard",
        "local_control": "local motion controller",
    }
    result = []
    used = sorted(
        {task_type for item in SCENARIOS for task_type, _ in item["tasks_per_robot"]}
    )
    for task_type in used:
        workload = catalog.get(task_type)
        targets = {}
        for target in ("orin", "edge"):
            profile = workload.profile_for(target)
            targets[target] = {
                "supported": profile.supported,
                "latency_ms": asdict(profile.latency),
                "cpu_units": profile.resources.cpu_cores,
                "gpu_units": profile.resources.gpu_units,
                "memory_mb": profile.resources.memory_mb,
                "input_size_mb": asdict(profile.input_size_mb),
                "output_size_mb": asdict(profile.output_size_mb),
                "energy_j": asdict(profile.energy_j),
                "failure_rate": profile.failure_rate,
                "success_probability": 1.0 - profile.failure_rate,
                "accuracy": asdict(profile.accuracy),
            }
        result.append(
            {
                "task_type": task_type,
                "experiment_model_assumption": model_names[task_type],
                "provenance": "synthetic_placeholder_not_measured",
                "targets": targets,
            }
        )
    return result


def expected_success_reward(report, workflow, nodes, catalog) -> float:
    """Compute sum_i p_i*q_i for the assignments actually selected."""

    task_by_id = {task.task_id: task for task in workflow.tasks}
    kind_by_id = {node.node_id: node.kind for node in nodes}
    reward = 0.0
    for record in report.task_results:
        if not record.target_node_id:
            continue
        task = task_by_id[record.task_id]
        target = (
            "edge"
            if kind_by_id[record.target_node_id] is NodeKind.EDGE
            else "orin"
        )
        failure_rate = catalog.get(task.spec.task_type).profile_for(
            target
        ).failure_rate
        reward += task.priority * (1.0 - failure_rate)
    return round(reward, 6)


def peak_resource_utilization(
    report,
    workflow,
    nodes,
    snapshots,
    catalog,
) -> dict[str, float]:
    """Reconstruct peak CPU/GPU/memory use from task execution intervals."""

    task_by_id = {task.task_id: task for task in workflow.tasks}
    node_by_id = {node.node_id: node for node in nodes}
    snapshot_by_id = {item.node_id: item for item in snapshots}
    peaks = {"cpu": 0.0, "gpu": 0.0, "memory": 0.0, "maximum": 0.0}
    for node_id, node in node_by_id.items():
        records = [
            item
            for item in report.task_results
            if item.target_node_id == node_id and item.finish_time_ms > item.start_time_ms
        ]
        boundaries = sorted(
            {
                point
                for item in records
                for point in (
                    round(
                        max(
                            item.start_time_ms,
                            item.finish_time_ms - item.compute_time_ms,
                        ),
                        2,
                    ),
                    item.finish_time_ms,
                )
            }
        )
        snapshot = snapshot_by_id[node_id]
        for point in boundaries:
            active = [
                item
                for item in records
                if (
                    round(
                        max(
                            item.start_time_ms,
                            item.finish_time_ms - item.compute_time_ms,
                        ),
                        2,
                    )
                    <= point
                    and point < item.finish_time_ms - 0.011
                )
            ]
            reserved_cpu = 0.0
            reserved_gpu = 0.0
            reserved_memory = 0.0
            for record in active:
                task = task_by_id[record.task_id]
                demand_cpu, demand_gpu, demand_memory = (
                    task_resource_demand(task, node)
                )
                reserved_cpu += demand_cpu
                reserved_gpu += demand_gpu
                reserved_memory += demand_memory
            cpu = max(
                snapshot.cpu_util,
                reserved_cpu / node.cpu_capacity,
            )
            gpu = snapshot.gpu_util
            if node.gpu_capacity > 0:
                gpu = max(
                    snapshot.gpu_util,
                    reserved_gpu / node.gpu_capacity,
                )
            memory = max(
                snapshot.memory_util,
                reserved_memory / node.memory_gb,
            )
            peaks["cpu"] = max(peaks["cpu"], cpu)
            peaks["gpu"] = max(peaks["gpu"], gpu)
            peaks["memory"] = max(peaks["memory"], memory)
    peaks["maximum"] = max(peaks["cpu"], peaks["gpu"], peaks["memory"])
    return {key: round(value, 6) for key, value in peaks.items()}


def main() -> None:
    DOC.mkdir(exist_ok=True)
    catalog = load_default_synthetic_workloads()
    profiles = profile_summary(catalog)
    benchmark = {
        "schema_version": "mars.binary-offload-benchmark.v1",
        "formal_experiment_seeds": list(SEEDS),
        "provenance": "synthetic_placeholder_not_hardware_measurement",
        "warning": "这些数值是可复现实验假设，不是真机测量结果。",
        "reference_sources": [
            {
                "id": "nvidia_agx_orin_technical_brief",
                "url": "https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf",
                "supports": [
                    "Jetson AGX Orin 32GB model",
                    "8-core Arm Cortex-A78AE CPU",
                    "32GB LPDDR5 memory",
                    "15W-40W power range"
                ]
            },
            {
                "id": "repository_synthetic_workloads",
                "path": "configs/mars/workloads.synthetic.json",
                "supports": [
                    "synthetic task latency",
                    "synthetic task resources",
                    "synthetic energy, failure rate and accuracy"
                ],
                "classification": "synthetic_data"
            }
        ],
        "frozen_objective": "min[-alpha*sum_i(p_i*q_i(x_i)) + beta*sum_i(T_comm_i(x_i)) + gamma*U_max(x)]",
        "weights": {"alpha": 1.0, "beta": FORMAL_BETA, "gamma": 2.0},
        "beta_sensitivity": {
            "values": list(BETA_SENSITIVITY),
            "separate_from_formal_results": True,
        },
        "hardware": HARDWARE,
        "profiles": profiles,
        "scenarios": [
            {
                key: value
                for key, value in scenario.items()
                if key != "tasks_per_robot"
            }
            | {
                "tasks_per_robot": [
                    {"task_type": name, "depends_on": list(deps)}
                    for name, deps in scenario["tasks_per_robot"]
                ]
            }
            for scenario in SCENARIOS
        ],
        "workflow_layout": {
            "logical_workflows_per_scene": 8,
            "two_per_robot": True,
            "arrival_pattern": "first wave at 0ms; second wave at 40% of the scene deadline",
            "dependencies_between_workflows": False,
            "shared_resources": ["edge-1", "network_links"],
            "platform_packaging": (
                "The eight independent DAGs are submitted in one WorkflowSpec "
                "because CentralCoordinator is single-workflow."
            ),
        },
        "methods": [
            {"optimizer": optimizer, "policy": policy, "description": description}
            for optimizer, policy, description in METHODS
        ],
    }
    (DOC / "benchmark.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_rows = []
    record_rows = []
    optimizer_epoch_rows = []
    for scenario in SCENARIOS:
        scene = build_scene(scenario, catalog)
        workflow = build_workflow(scene)
        nodes = build_node_specs(scene)
        snapshots = build_node_snapshots(scene)
        links = build_link_specs(scene)
        link_snapshots = build_link_snapshots(scene)
        for seed in SEEDS:
            for optimizer, policy, description in METHODS:
                algorithm = optimizer if policy is None else policy
                binary_optimizer = BinaryOffloadOptimizer(
                    alpha=1.0,
                    beta=FORMAL_BETA,
                    gamma=2.0,
                )
                registry = OptimizerRegistry()
                registry.register(binary_optimizer)
                started = perf_counter()
                report = run_workflow_simulation(
                    workflow,
                    nodes,
                    snapshots,
                    algorithm=algorithm,
                    seed=seed,
                    network_jitter=0.0,
                    resource_noise=0.05,
                    link_specs=links,
                    link_snapshots=link_snapshots,
                    optimizer_registry=registry,
                    fallback_optimizer=None,
                )
                elapsed_ms = (perf_counter() - started) * 1000.0
                modes: dict[str, int] = {}
                for record in report.task_results:
                    modes[record.mode] = modes.get(record.mode, 0) + 1
                    logical_workflow = record.task_id.split("--", 1)[0]
                    record_rows.append(
                        {
                            "scenario": scenario["id"],
                            "seed": seed,
                            "logical_workflow": logical_workflow,
                            "method": algorithm,
                            **asdict(record),
                        }
                    )
                utilization = peak_resource_utilization(
                    report,
                    workflow,
                    nodes,
                    snapshots,
                    catalog,
                )
                if optimizer == "binary_offload":
                    optimizer_epoch_rows.extend(
                        {
                            "experiment": "formal_beta_0.01",
                            "scenario": scenario["id"],
                            "seed": seed,
                            **item,
                        }
                        for item in binary_optimizer.solve_history
                    )
                success_reward = expected_success_reward(
                    report, workflow, nodes, catalog
                )
                communication_time = round(
                    sum(
                        item.communication_time_ms
                        for item in report.task_results
                    ),
                    3,
                )
                aggregate_objective = (
                    -success_reward
                    + FORMAL_BETA * communication_time
                    + 2.0 * utilization["maximum"]
                )
                metric_rows.append(
                    {
                        **report.metrics,
                        "experiment": "formal_beta_0.01",
                        "scenario": scenario["id"],
                        "seed": seed,
                        "logical_workflows": 8,
                        "method": algorithm,
                        "optimizer": optimizer,
                        "policy": policy or "frozen_binary_objective",
                        "beta": FORMAL_BETA if optimizer == "binary_offload" else "",
                        "expected_success_reward": success_reward,
                        "peak_cpu_utilization": utilization["cpu"],
                        "peak_gpu_utilization": utilization["gpu"],
                        "peak_memory_utilization": utilization["memory"],
                        "maximum_resource_utilization": utilization["maximum"],
                        "workflow_evaluation_objective": round(
                            aggregate_objective, 6
                        ),
                        "communication_time_ms": communication_time,
                        "local_tasks": modes.get("local", 0) + modes.get("fallback_local", 0),
                        "edge_tasks": modes.get("edge", 0),
                        "wall_clock_ms": round(elapsed_ms, 3),
                        "fallback_disabled": True,
                        "description": description,
                    }
                )

    columns = list(metric_rows[0])
    with (DOC / "step3_evaluation_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metric_rows)
    (DOC / "step3_evaluation_records.json").write_text(
        json.dumps(record_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_rows = []
    summary_metrics = (
        "expected_success_reward",
        "communication_time_ms",
        "avg_latency_ms",
        "p95_latency_ms",
        "deadline_miss_rate",
        "maximum_resource_utilization",
        "workflow_evaluation_objective",
        "total_solver_time_ms",
    )
    for scenario in SCENARIOS:
        for optimizer, policy, _ in METHODS:
            method = optimizer if policy is None else policy
            selected = [
                row for row in metric_rows
                if row["scenario"] == scenario["id"] and row["method"] == method
            ]
            summary = {"scenario": scenario["id"], "method": method, "runs": len(selected)}
            for metric in summary_metrics:
                values = [float(row[metric]) for row in selected]
                summary[f"{metric}_mean"] = round(mean(values), 6)
                summary[f"{metric}_std"] = round(stdev(values), 6)
            summary_rows.append(summary)
    with (DOC / "step3_evaluation_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    sensitivity_rows = []
    for beta in BETA_SENSITIVITY:
        for scenario in SCENARIOS:
            scene = build_scene(scenario, catalog)
            workflow = build_workflow(scene)
            nodes = build_node_specs(scene)
            snapshots = build_node_snapshots(scene)
            links = build_link_specs(scene)
            link_snapshots = build_link_snapshots(scene)
            for seed in SEEDS:
                beta_optimizer = BinaryOffloadOptimizer(
                    alpha=1.0, beta=beta, gamma=2.0
                )
                beta_registry = OptimizerRegistry()
                beta_registry.register(beta_optimizer)
                report = run_workflow_simulation(
                    workflow,
                    nodes,
                    snapshots,
                    algorithm="binary_offload",
                    seed=seed,
                    network_jitter=0.0,
                    resource_noise=0.05,
                    link_specs=links,
                    link_snapshots=link_snapshots,
                    optimizer_registry=beta_registry,
                    fallback_optimizer=None,
                )
                utilization = peak_resource_utilization(
                    report, workflow, nodes, snapshots, catalog
                )
                optimizer_epoch_rows.extend(
                    {
                        "experiment": "beta_sensitivity_not_formal",
                        "scenario": scenario["id"],
                        "seed": seed,
                        **item,
                    }
                    for item in beta_optimizer.solve_history
                )
                local_tasks = sum(
                    item.mode in {"local", "fallback_local"}
                    for item in report.task_results
                )
                edge_tasks = sum(
                    item.mode == "edge" for item in report.task_results
                )
                sensitivity_rows.append(
                    {
                        **report.metrics,
                        "experiment": "beta_sensitivity_not_formal",
                        "scenario": scenario["id"],
                        "seed": seed,
                        "beta": beta,
                        "expected_success_reward": expected_success_reward(
                            report, workflow, nodes, catalog
                        ),
                        "communication_time_ms": round(
                            sum(
                                item.communication_time_ms
                                for item in report.task_results
                            ),
                            3,
                        ),
                        "maximum_resource_utilization": utilization["maximum"],
                        "local_tasks": local_tasks,
                        "edge_tasks": edge_tasks,
                    }
                )
    with (DOC / "step3_beta_sensitivity.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    sensitivity_summary = []
    for beta in BETA_SENSITIVITY:
        selected = [row for row in sensitivity_rows if row["beta"] == beta]
        sensitivity_summary.append(
            {
                "beta": beta,
                "runs": len(selected),
                "edge_tasks_mean": round(
                    mean(float(row["edge_tasks"]) for row in selected), 4
                ),
                "success_reward_mean": round(
                    mean(float(row["expected_success_reward"]) for row in selected), 6
                ),
                "communication_ms_mean": round(
                    mean(float(row["communication_time_ms"]) for row in selected), 6
                ),
                "latency_ms_mean": round(
                    mean(float(row["avg_latency_ms"]) for row in selected), 6
                ),
                "deadline_miss_rate_mean": round(
                    mean(float(row["deadline_miss_rate"]) for row in selected), 6
                ),
            }
        )
    with (DOC / "step3_beta_sensitivity_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(sensitivity_summary[0])
        )
        writer.writeheader()
        writer.writerows(sensitivity_summary)
    (DOC / "step3_optimizer_epoch_metrics.json").write_text(
        json.dumps(optimizer_epoch_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote {len(metric_rows)} runs and {len(record_rows)} task records to {DOC}")


if __name__ == "__main__":
    main()
