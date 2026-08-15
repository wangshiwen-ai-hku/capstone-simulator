"""Fixed experiment definition and scene construction for binary offloading."""

from __future__ import annotations

from dataclasses import asdict

from evals.workflow import WorkflowEvaluationWeights
from backend.app.schemas import BenchmarkScene
from mars.domain.task import TaskClass
from mars.optimizers.policy import SolveLimits


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
        "initial_utilization": {
            "cpu": 0.10,
            "gpu": 0.08,
            "memory": 0.12,
        },
        "field_provenance": {
            "label": (
                "reference_data: NVIDIA Jetson AGX Orin 32GB "
                "technical brief"
            ),
            "power_mode": (
                "reference_data: NVIDIA lists 15W-40W; experiment fixes 40W"
            ),
            "memory_gb": "reference_data: NVIDIA lists 32GB LPDDR5",
            "cpu_capacity": (
                "synthetic_data: MARS capacity unit; numerically aligned "
                "to the official 8-core CPU"
            ),
            "gpu_capacity": (
                "synthetic_data: normalized MARS unit, not CUDA-core count "
                "or utilization percentage"
            ),
            "max_concurrency": (
                "synthetic_data: conservative experiment setting"
            ),
            "initial_utilization": (
                "synthetic_data: experiment initial condition"
            ),
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
        "initial_utilization": {
            "cpu": 0.20,
            "gpu": 0.25,
            "memory": 0.18,
        },
        "field_provenance": {
            "all_fields": (
                "synthetic_data: the project does not specify an Edge PC "
                "CPU/GPU model"
            )
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
            "all_fields": (
                "synthetic_data: fixed LAN/Wi-Fi experiment assumptions, "
                "not measured values"
            )
        },
    },
}

SCENARIOS = (
    {
        "id": "warehouse_navigation",
        "name": "仓库导航",
        "description": (
            "4台机器人同时做急停、定位、目标检测、环境理解和局部规划。"
        ),
        "tasks_per_robot": (
            ("emergency_stop", ()),
            ("localization", ("emergency_stop",)),
            ("object_detection", ("localization",)),
            (
                "environment_understanding",
                ("localization", "object_detection"),
            ),
            ("local_planning", ("environment_understanding",)),
        ),
        "deadline_ms": 850.0,
    },
    {
        "id": "multi_robot_mapping",
        "name": "多机器人地图构建",
        "description": (
            "每台机器人完成定位、语义分割和压缩，再各自执行地图融合候选。"
        ),
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
        "description": (
            "4台机器人同时提交两条视觉链，Edge保持较高初始利用率。"
        ),
        "tasks_per_robot": (
            ("object_detection", ()),
            ("semantic_segmentation", ("object_detection",)),
            (
                "environment_understanding",
                ("object_detection", "semantic_segmentation"),
            ),
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
        "edge_utilization",
        HARDWARE["edge"]["initial_utilization"],
    )
    resource_rows = [
        {
            "node_id": node["id"],
            "cpu_util": edge_util["cpu"] if node["kind"] == "edge" else 0.10,
            "gpu_util": edge_util["gpu"] if node["kind"] == "edge" else 0.08,
            "memory_util": (
                edge_util["memory"] if node["kind"] == "edge" else 0.12
            ),
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
                task_spec = workload.to_task_spec()
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
                dependencies = tuple(
                    created[name] for name in dependency_names
                )
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
                            5
                            if workload.task_class is TaskClass.LOCAL_SAFETY
                            else 3
                        ),
                        "compute_demand": task_spec.compute_demand,
                        "gpu_demand": task_spec.gpu_demand,
                        "latency_budget_ms": task_spec.latency_budget_ms,
                        "model_requirement": task_spec.model_requirement,
                        "data_size_mb": task_spec.input_size_mb,
                        "output_size_mb": task_spec.output_size_mb,
                        "bandwidth_requirement_mbps": (
                            task_spec.bandwidth_requirement_mbps
                        ),
                        "energy_budget_j": task_spec.energy_budget_j,
                        "placement_constraints": placement,
                        "result_verification": "synthetic profile outcome",
                        "arrival_time_ms": arrival_ms,
                        "deadline_ms": (
                            arrival_ms + float(scenario["deadline_ms"])
                        ),
                        "dependencies": list(dependencies),
                        "stage_index": stage,
                        "expected_accuracy": (
                            workload.profile_for("orin").accuracy.typical
                        ),
                        "input_ports": [
                            {
                                "name": port.name,
                                "message_type": port.message_type,
                            }
                            for port in task_spec.input_ports
                        ],
                        "output_ports": [
                            {
                                "name": port.name,
                                "message_type": port.message_type,
                            }
                            for port in task_spec.output_ports
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


def profile_summary(catalog) -> list[dict[str, object]]:
    """Describe every workload profile referenced by the fixed scenarios."""

    model_names = {
        "object_detection": "YOLOv8n TensorRT FP16, batch=1, 640x640 JPEG",
        "semantic_segmentation": (
            "lightweight segmentation TensorRT FP16, batch=1, 640x640 JPEG"
        ),
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
        {
            task_type
            for item in SCENARIOS
            for task_type, _ in item["tasks_per_robot"]
        }
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


def build_benchmark_manifest(catalog) -> dict[str, object]:
    """Build the stable metadata document written as ``benchmark.json``."""

    return {
        "schema_version": "mars.binary-offload-benchmark.v2",
        "formal_experiment": FORMAL_EXPERIMENT,
        "formal_experiment_seeds": list(SEEDS),
        "provenance": "synthetic_placeholder_not_hardware_measurement",
        "warning": "这些数值是可复现实验假设，不是真机测量结果。",
        "reference_sources": [
            {
                "id": "nvidia_agx_orin_technical_brief",
                "url": (
                    "https://www.nvidia.com/content/dam/en-zz/Solutions/"
                    "gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-"
                    "brief.pdf"
                ),
                "supports": [
                    "Jetson AGX Orin 32GB model",
                    "8-core Arm Cortex-A78AE CPU",
                    "32GB LPDDR5 memory",
                    "15W-40W power range",
                ],
            },
            {
                "id": "repository_synthetic_workloads",
                "path": "configs/mars/workloads.synthetic.json",
                "supports": [
                    "synthetic task latency",
                    "synthetic task resources",
                    "synthetic energy, failure rate and accuracy",
                ],
                "classification": "synthetic_data",
            },
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
            "implementation": "evals.workflow.evaluate_run_artifact",
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
        "profiles": profile_summary(catalog),
        "scenarios": [
            {
                key: value
                for key, value in scenario.items()
                if key != "tasks_per_robot"
            }
            | {
                "tasks_per_robot": [
                    {"task_type": name, "depends_on": list(dependencies)}
                    for name, dependencies in scenario["tasks_per_robot"]
                ]
            }
            for scenario in SCENARIOS
        ],
        "workflow_layout": {
            "logical_workflows_per_scene": 8,
            "two_per_robot": True,
            "arrival_pattern": (
                "first wave at 0ms; second wave at 40% of the scene deadline"
            ),
            "dependencies_between_workflows": False,
            "shared_resources": ["edge-1", "network_links"],
            "platform_packaging": (
                "The eight independent DAGs are submitted in one WorkflowSpec "
                "because CentralCoordinator is single-workflow."
            ),
        },
        "methods": [
            {
                "optimizer": optimizer,
                "policy": policy,
                "description": description,
            }
            for optimizer, policy, description in METHODS
        ],
    }


__all__ = [
    "BETA_SENSITIVITY",
    "DEFAULT_SOLVE_LIMITS",
    "FALLBACK_OPTIMIZER",
    "FORMAL_BETA",
    "FORMAL_EXPERIMENT",
    "FORMAL_WEIGHTS",
    "HARDWARE",
    "METHODS",
    "SCENARIOS",
    "SEEDS",
    "SENSITIVITY_EXPERIMENT",
    "build_benchmark_manifest",
    "build_scene",
    "profile_summary",
]
