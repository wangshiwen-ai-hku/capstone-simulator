"""Run the fixed Stage 4-6 MARS benchmark and write every result under doc/."""

# ruff: noqa: E402 -- direct execution adds the repository root to sys.path.

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping
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
from mars.engine import run_workflow_simulation
from mars.optimizers import BinaryOffloadOptimizer, OptimizerRegistry
from mars.optimizers.policy import SolveLimits
from mars.synthetic_workloads import load_default_synthetic_workloads
from mars.workflow_metrics import WorkflowEvaluationWeights


DOC = ROOT / "doc"
SEED = 20260731
SEEDS = (7, 17, 27, 37, 47)
FORMAL_WEIGHTS = WorkflowEvaluationWeights(
    success=1.0,
    communication=1.0,
    utilization=2.0,
)
FORMAL_BETA = FORMAL_WEIGHTS.communication
BETA_SENSITIVITY = (0.0, 0.25, 1.0, 4.0)
FORMAL_EXPERIMENT = "formal_normalized_beta_1"
SENSITIVITY_EXPERIMENT = "normalized_beta_sensitivity_not_formal"
FALLBACK_OPTIMIZER = "heuristic"
DEFAULT_SOLVE_LIMITS = SolveLimits()

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
    (
        "binary_offload",
        None,
        "滚动时域（receding-horizon）的逐 epoch 限时穷举 placement",
    ),
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
        workflow_deadline_ms=max(
            float(task["deadline_ms"]) for task in task_rows
        ),
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


def _json_counter(values) -> str:
    return json.dumps(
        dict(sorted(Counter(str(value) for value in values).items())),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_mapping(value: Mapping[object, object]) -> str:
    return json.dumps(
        {str(key): value[key] for key in sorted(value, key=str)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def scheduling_audit(report) -> dict[str, object]:
    """Read effective scheduling facts emitted by the production coordinator."""

    scheduling = report.workflow.get("scheduling")
    if not isinstance(scheduling, Mapping):
        raise RuntimeError(
            "simulation report does not expose workflow.scheduling; refusing "
            "to infer effective optimizers or fallback from task placements"
        )
    effective_optimizers = scheduling.get("effective_optimizers")
    effective_policies = scheduling.get("effective_policies")
    if not isinstance(effective_optimizers, Mapping):
        raise RuntimeError("workflow.scheduling.effective_optimizers is missing")
    if not isinstance(effective_policies, Mapping):
        raise RuntimeError("workflow.scheduling.effective_policies is missing")
    requested = str(scheduling.get("requested_algorithm", ""))
    if not requested:
        raise RuntimeError("workflow.scheduling.requested_algorithm is missing")
    solve_limits = scheduling.get("solve_limits")
    if not isinstance(solve_limits, Mapping):
        raise RuntimeError("workflow.scheduling.solve_limits is missing")
    fallback_count = int(scheduling.get("fallback_count", 0))
    return {
        "requested_algorithm": requested,
        "effective_algorithm": (
            next(iter(effective_optimizers))
            if len(effective_optimizers) == 1
            else "mixed"
        ),
        "effective_optimizers": _json_mapping(effective_optimizers),
        "effective_policies": _json_mapping(effective_policies),
        "effective_solver_statuses": (
            _json_mapping(scheduling["solve_statuses"])
            if isinstance(scheduling.get("solve_statuses"), Mapping)
            else "not_exposed_by_simulation_report"
        ),
        "effective_termination_reasons": (
            _json_mapping(scheduling["termination_reasons"])
            if isinstance(scheduling.get("termination_reasons"), Mapping)
            else "not_exposed_by_simulation_report"
        ),
        "fallback_enabled": True,
        "fallback_optimizer": FALLBACK_OPTIMIZER,
        "fallback_count": fallback_count,
        "fallback_used": fallback_count > 0,
        "requested_seed": int(scheduling.get("requested_seed", 0)),
        "deterministic_execution": bool(scheduling.get("deterministic", False)),
        "execution_seed": int(scheduling.get("execution_seed", 0)),
        "solve_budget_ms": float(solve_limits["solve_budget_ms"]),
        "max_iterations": int(solve_limits["max_iterations"]),
        "solver_deterministic": bool(solve_limits["deterministic"]),
        "solver_random_seed": int(solve_limits["random_seed"]),
    }


def solver_audit(
    requested_algorithm: str,
    binary_optimizer: BinaryOffloadOptimizer,
) -> dict[str, object]:
    """Describe requested binary solves without confusing them with fallback."""

    history = (
        tuple(binary_optimizer.solve_history)
        if requested_algorithm == "binary_offload"
        else ()
    )
    budgets = sorted({float(item["solve_budget_ms"]) for item in history})
    iteration_limits = sorted({int(item["max_iterations"]) for item in history})
    if history:
        statuses = _json_counter(item["solve_status"] for item in history)
        reasons = _json_counter(item["termination_reason"] for item in history)
    elif requested_algorithm == "binary_offload":
        statuses = "{}"
        reasons = "{}"
    else:
        statuses = "not_exposed_for_policy_alias"
        reasons = "not_exposed_for_policy_alias"
    return {
        "requested_solver_history_epochs": len(history),
        "requested_solver_statuses": statuses,
        "requested_termination_reasons": reasons,
        "observed_solve_budgets_ms": json.dumps(
            budgets,
            separators=(",", ":"),
        ),
        "observed_iteration_limits": json.dumps(
            iteration_limits,
            separators=(",", ":"),
        ),
        "requested_placement_search_exhaustive": (
            all(bool(item["placement_search_exhaustive"]) for item in history)
            if history
            else ""
        ),
        "search_scope": (
            "receding_horizon_ready_epoch"
            if requested_algorithm == "binary_offload"
            else "policy_alias_per_ready_epoch"
        ),
        "global_workflow_exact": False,
    }


def run_benchmark_case(
    *,
    experiment: str,
    scenario_id: str,
    workflow,
    nodes,
    snapshots,
    links,
    link_snapshots,
    seed: int,
    optimizer: str,
    policy: str | None,
    description: str,
    beta: float = FORMAL_BETA,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Run one method/seed case and return auditable rows without writing files."""

    requested_algorithm = optimizer if policy is None else policy
    binary_optimizer = BinaryOffloadOptimizer(
        alpha=FORMAL_WEIGHTS.success,
        beta=beta,
        gamma=FORMAL_WEIGHTS.utilization,
    )
    registry = OptimizerRegistry()
    registry.register(binary_optimizer)
    case_weights = WorkflowEvaluationWeights(
        success=FORMAL_WEIGHTS.success,
        communication=beta,
        utilization=FORMAL_WEIGHTS.utilization,
    )
    started = perf_counter()
    report = run_workflow_simulation(
        workflow,
        nodes,
        snapshots,
        algorithm=requested_algorithm,
        seed=seed,
        network_jitter=0.0,
        resource_noise=0.05,
        evaluation_weights=case_weights,
        link_specs=links,
        link_snapshots=link_snapshots,
        optimizer_registry=registry,
        fallback_optimizer=FALLBACK_OPTIMIZER,
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    audit = scheduling_audit(report)
    if audit["requested_algorithm"] != requested_algorithm:
        raise RuntimeError(
            "coordinator scheduling audit does not match requested algorithm: "
            f"{audit['requested_algorithm']!r} != {requested_algorithm!r}"
        )
    if audit["requested_seed"] != seed:
        raise RuntimeError(
            "coordinator scheduling audit does not match requested seed: "
            f"{audit['requested_seed']!r} != {seed!r}"
        )
    modes = Counter(record.mode for record in report.task_results)
    effective_optimizers = str(audit["effective_optimizers"])
    records = [
        {
            "experiment": experiment,
            "scenario": scenario_id,
            "seed": seed,
            "logical_workflow": record.task_id.split("--", 1)[0],
            "requested_algorithm": requested_algorithm,
            "effective_optimizers": effective_optimizers,
            "fallback_count": audit["fallback_count"],
            **asdict(record),
        }
        for record in report.task_results
    ]
    epoch_rows = [
        {
            "experiment": experiment,
            "scenario": scenario_id,
            "seed": seed,
            "requested_algorithm": requested_algorithm,
            "fallback_enabled": True,
            **item,
        }
        for item in binary_optimizer.solve_history
    ]
    row = {
        **report.metrics,
        "experiment": experiment,
        "scenario": scenario_id,
        "seed": seed,
        "logical_workflows": 8,
        "method": requested_algorithm,
        "nominal_optimizer": optimizer,
        "nominal_policy": policy or "binary_offload",
        **audit,
        **solver_audit(requested_algorithm, binary_optimizer),
        "evaluation_success_weight": case_weights.success,
        "evaluation_communication_weight": case_weights.communication,
        "evaluation_utilization_weight": case_weights.utilization,
        "optimizer_success_weight": (
            FORMAL_WEIGHTS.success if optimizer == "binary_offload" else ""
        ),
        "optimizer_communication_weight": (
            beta if optimizer == "binary_offload" else ""
        ),
        "optimizer_utilization_weight": (
            FORMAL_WEIGHTS.utilization if optimizer == "binary_offload" else ""
        ),
        "local_tasks": modes.get("local", 0) + modes.get("fallback_local", 0),
        "edge_tasks": modes.get("edge", 0),
        "wall_clock_ms": round(elapsed_ms, 3),
        "description": description,
    }
    return row, records, epoch_rows


def main() -> None:
    DOC.mkdir(exist_ok=True)
    catalog = load_default_synthetic_workloads()
    profiles = profile_summary(catalog)
    benchmark = {
        "schema_version": "mars.binary-offload-benchmark.v2",
        "formal_experiment": FORMAL_EXPERIMENT,
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
        "objective": (
            "At each ready-task scheduling epoch, min["
            "-alpha*weighted_success_ratio + beta*normalized_communication "
            "+ gamma*U_max] subject to the shared placement constraints."
        ),
        "weights": {
            "alpha": FORMAL_WEIGHTS.success,
            "beta": FORMAL_BETA,
            "gamma": FORMAL_WEIGHTS.utilization,
        },
        "evaluation": {
            "implementation": "mars.workflow_metrics.evaluate_workflow_metrics",
            "components_are_shared_with_production": True,
            "case_weights_are_passed_to_the_shared_evaluator": True,
            "deadline_reporting": {
                "legacy_deadline_miss_rate": "all task records",
                "executed_deadline_miss_rate": "executed tasks only",
                "required_task_on_time_rate": (
                    "all required tasks; failures, drops, and skips are not on time"
                ),
            },
        },
        "solver": {
            "scope": "receding_horizon_ready_epoch",
            "search": "bounded_exhaustive_one_hot_placement",
            "global_workflow_exact": False,
            "solve_budget_ms": DEFAULT_SOLVE_LIMITS.solve_budget_ms,
            "max_iterations": DEFAULT_SOLVE_LIMITS.max_iterations,
            "deterministic": DEFAULT_SOLVE_LIMITS.deterministic,
            "random_seed": {
                "source": "case_seed",
                "recorded_as": "solver_random_seed",
            },
            "fallback_optimizer": FALLBACK_OPTIMIZER,
        },
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
                row, records, epochs = run_benchmark_case(
                    experiment=FORMAL_EXPERIMENT,
                    scenario_id=str(scenario["id"]),
                    workflow=workflow,
                    nodes=nodes,
                    snapshots=snapshots,
                    links=links,
                    link_snapshots=link_snapshots,
                    seed=seed,
                    optimizer=optimizer,
                    policy=policy,
                    description=description,
                )
                metric_rows.append(row)
                record_rows.extend(records)
                optimizer_epoch_rows.extend(epochs)

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
        "executed_deadline_miss_rate",
        "required_task_on_time_rate",
        "skipped_task_count",
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
                row, _, epochs = run_benchmark_case(
                    experiment=SENSITIVITY_EXPERIMENT,
                    scenario_id=str(scenario["id"]),
                    workflow=workflow,
                    nodes=nodes,
                    snapshots=snapshots,
                    links=links,
                    link_snapshots=link_snapshots,
                    seed=seed,
                    optimizer="binary_offload",
                    policy=None,
                    description=METHODS[0][2],
                    beta=beta,
                )
                row["beta"] = beta
                sensitivity_rows.append(row)
                optimizer_epoch_rows.extend(epochs)
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
                "success_ratio_mean": round(
                    mean(float(row["expected_success_ratio"]) for row in selected),
                    6,
                ),
                "communication_ms_mean": round(
                    mean(float(row["communication_time_ms"]) for row in selected), 6
                ),
                "normalized_communication_mean": round(
                    mean(
                        float(row["normalized_communication"])
                        for row in selected
                    ),
                    6,
                ),
                "latency_ms_mean": round(
                    mean(float(row["avg_latency_ms"]) for row in selected), 6
                ),
                "deadline_miss_rate_mean": round(
                    mean(float(row["deadline_miss_rate"]) for row in selected), 6
                ),
                "executed_deadline_miss_rate_mean": round(
                    mean(
                        float(row["executed_deadline_miss_rate"])
                        for row in selected
                    ),
                    6,
                ),
                "required_task_on_time_rate_mean": round(
                    mean(
                        float(row["required_task_on_time_rate"])
                        for row in selected
                    ),
                    6,
                ),
                "skipped_task_count_mean": round(
                    mean(float(row["skipped_task_count"]) for row in selected),
                    6,
                ),
                "maximum_resource_utilization_mean": round(
                    mean(
                        float(row["maximum_resource_utilization"])
                        for row in selected
                    ),
                    6,
                ),
                "workflow_evaluation_objective_mean": round(
                    mean(
                        float(row["workflow_evaluation_objective"])
                        for row in selected
                    ),
                    6,
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
