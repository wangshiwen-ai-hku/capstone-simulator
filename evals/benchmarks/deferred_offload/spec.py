"""Fixed configuration and scene construction for deferred-offload benchmarks."""

from __future__ import annotations

from backend.app.schemas import BenchmarkScene
from evals.benchmarks.binary_offload.spec import (
    DEFAULT_SOLVE_LIMITS,
    FORMAL_WEIGHTS,
    SEEDS,
    build_benchmark_manifest,
    build_scene,
)
from mars.domain import TaskClass


EXPERIMENT_ID = "deferred_offload_cross_robot_v1"
DEFERRED_WEIGHTS = {
    "alpha": FORMAL_WEIGHTS.success,
    "beta": FORMAL_WEIGHTS.communication,
    "gamma": FORMAL_WEIGHTS.utilization,
    "delta": 1.0,
}
DEFERRED_METHODS = (
    ("deferred_offload", None, "CP-SAT assign-or-defer"),
    ("binary_offload", None, "bounded exhaustive one-hot placement"),
    ("heuristic", "local_first", "heuristic local-first"),
    ("heuristic", "edge_first", "heuristic edge-first"),
    ("heuristic", "dag_deadline", "heuristic DAG-deadline"),
)

ASYMMETRIC_PEER_SCENARIO = {
    "id": "asymmetric_peer_burst",
    "name": "Asymmetric robot burst",
    "description": (
        "Jetson-1 receives two vision pipelines while Jetson-2 is lightly "
        "loaded, Jetson-3 and Jetson-4 are busy, and the Edge uplink is "
        "degraded. Peer placement remains a choice rather than a pin."
    ),
    "tasks_per_robot": (
        ("object_detection", ()),
        ("localization", ()),
        ("semantic_segmentation", ("object_detection",)),
        (
            "environment_understanding",
            ("localization", "semantic_segmentation"),
        ),
    ),
    "task_priorities": {
        "object_detection": 2,
        "localization": 3,
        "semantic_segmentation": 5,
        "environment_understanding": 4,
    },
    "deadline_ms": 900.0,
    "simultaneous_workflow_arrivals": True,
    "task_source_node_ids": ("jetson-1",),
    "node_initial_utilization": {
        "jetson-1": {"cpu": 0.45, "gpu": 0.30, "memory": 0.35},
        "jetson-2": {"cpu": 0.05, "gpu": 0.05, "memory": 0.10},
        "jetson-3": {"cpu": 0.90, "gpu": 0.95, "memory": 0.80},
        "jetson-4": {"cpu": 0.90, "gpu": 0.95, "memory": 0.80},
        "edge-1": {"cpu": 0.90, "gpu": 0.98, "memory": 0.85},
    },
    "link_snapshot_overrides": {
        "jetson-1-to-jetson-2": {
            "available_bandwidth_mbps": 300.0,
            "latency_ms": 2.0,
        },
        "jetson-2-to-jetson-1": {
            "available_bandwidth_mbps": 300.0,
            "latency_ms": 2.0,
        },
        **{
            link_id: {
                "available_bandwidth_mbps": 25.0,
                "latency_ms": 30.0,
            }
            for robot_id in range(1, 5)
            for link_id in (
                f"jetson-{robot_id}-to-edge-1",
                f"edge-1-to-jetson-{robot_id}",
            )
        },
    },
}

PEER_REGRESSION_SCENARIO = {
    "id": "peer_regression",
    "name": "Peer regression",
    "description": "Jetson-2 is the only immediately feasible target.",
    "tasks_per_robot": (("object_detection", ()),),
    "deadline_ms": 500.0,
    "task_source_node_ids": ("jetson-1",),
    "node_initial_utilization": {
        "jetson-1": {"cpu": 0.50, "gpu": 0.90, "memory": 0.35},
        "jetson-2": {"cpu": 0.05, "gpu": 0.05, "memory": 0.10},
        "jetson-3": {"cpu": 0.80, "gpu": 0.90, "memory": 0.50},
        "jetson-4": {"cpu": 0.80, "gpu": 0.90, "memory": 0.50},
        "edge-1": {"cpu": 0.90, "gpu": 0.95, "memory": 0.70},
    },
    "offline_node_ids": ("edge-1",),
}

DEFERRED_SCENARIOS = (ASYMMETRIC_PEER_SCENARIO,)


def build_deferred_scene(
    scenario: dict[str, object],
    catalog,
) -> BenchmarkScene:
    """Build a binary benchmark scene with peer placement enabled."""

    scene = build_scene(scenario, catalog)
    selected_sources = frozenset(
        str(node_id)
        for node_id in scenario.get("task_source_node_ids", ())
    )
    tasks = []
    priorities = scenario.get("task_priorities", {})
    for task in scene.tasks:
        if selected_sources and task.source_robot_id not in selected_sources:
            continue
        placement = task.placement_constraints
        if task.task_class is TaskClass.LOCAL_SAFETY:
            tasks.append(task)
            continue
        assert placement is not None
        timing_update = (
            {
                "arrival_time_ms": 0.0,
                "deadline_ms": float(scenario["deadline_ms"]),
            }
            if scenario.get("simultaneous_workflow_arrivals", False)
            else {}
        )
        tasks.append(
            task.model_copy(
                update={
                    **timing_update,
                    "priority": priorities.get(task.task_type, task.priority),
                    "placement_constraints": placement.model_copy(
                        update={
                            "allowed_node_kinds": ["robot", "edge"],
                            "allow_other_robots": True,
                        }
                    )
                }
            )
        )
    utilization_by_node = scenario.get("node_initial_utilization", {})
    offline_node_ids = frozenset(scenario.get("offline_node_ids", ()))
    initial_resources = []
    for snapshot in scene.initial_resources:
        utilization = utilization_by_node.get(snapshot.node_id, {})
        initial_resources.append(
            snapshot.model_copy(
                update={
                    "cpu_util": utilization.get("cpu", snapshot.cpu_util),
                    "gpu_util": utilization.get("gpu", snapshot.gpu_util),
                    "memory_util": utilization.get(
                        "memory",
                        snapshot.memory_util,
                    ),
                    "online": snapshot.node_id not in offline_node_ids,
                }
            )
        )
    link_overrides = scenario.get("link_snapshot_overrides", {})
    link_snapshots = [
        snapshot.model_copy(update=link_overrides.get(snapshot.link_id, {}))
        for snapshot in scene.link_snapshots
    ]
    return scene.model_copy(
        update={
            "tasks": tasks,
            "initial_resources": initial_resources,
            "link_snapshots": link_snapshots,
        }
    )


def build_deferred_benchmark_manifest(catalog) -> dict[str, object]:
    """Describe the actual deferred benchmark configuration and solver."""

    shared = build_benchmark_manifest(catalog)
    return {
        "schema_version": "mars.deferred-offload-benchmark.v1",
        "formal_experiment": EXPERIMENT_ID,
        "formal_experiment_seeds": list(SEEDS),
        "provenance": shared["provenance"],
        "warning": shared["warning"],
        "reference_sources": shared["reference_sources"],
        "base_configuration": "mars.binary-offload-benchmark.v2",
        "objective": (
            "At each ready-task scheduling epoch, minimize "
            "[-alpha*expected_chain_weighted_success_ratio + "
            "beta*normalized_communication_ratio + "
            "gamma*maximum_resource_utilization + "
            "delta*deferred_priority_penalty]."
        ),
        "weights": dict(DEFERRED_WEIGHTS),
        "evaluation": shared["evaluation"],
        "solver": {
            "optimizer": "deferred_offload",
            "formulation": "assign_or_defer",
            "backend": "OR-Tools CP-SAT",
            "search": "cp_sat_assign_or_defer",
            "scope": "receding_horizon_ready_epoch",
            "solve_budget_ms": DEFAULT_SOLVE_LIMITS.solve_budget_ms,
            "deterministic": DEFAULT_SOLVE_LIMITS.deterministic,
        },
        "hardware": shared["hardware"],
        "profiles": shared["profiles"],
        "placement_change": {
            "ordinary_allowed_node_kinds": ["robot", "edge"],
            "ordinary_allow_other_robots": True,
            "local_safety_unchanged": True,
        },
        "deferred_weight": DEFERRED_WEIGHTS["delta"],
        "scenarios": [
            {
                key: value
                for key, value in scenario.items()
                if key != "tasks_per_robot"
            }
            | {
                "tasks_per_robot": [
                    {
                        "task_type": name,
                        "depends_on": list(dependencies),
                    }
                    for name, dependencies in scenario["tasks_per_robot"]
                ]
            }
            for scenario in DEFERRED_SCENARIOS
        ],
        "workflow_layout": {
            "logical_workflows_per_scene": 2,
            "source_robots": ["jetson-1"],
            "arrival_pattern": (
                "first pipeline at 0ms; second at 40% of the deadline"
            ),
            "dependencies_between_workflows": False,
            "shared_resources": ["peer_robots", "edge-1", "network_links"],
            "platform_packaging": (
                "The two independent DAGs share one MARS WorkflowSpec."
            ),
        },
        "methods": [
            {
                "optimizer": optimizer,
                "policy": policy,
                "description": description,
            }
            for optimizer, policy, description in DEFERRED_METHODS
        ],
    }


__all__ = [
    "ASYMMETRIC_PEER_SCENARIO",
    "DEFERRED_METHODS",
    "DEFERRED_SCENARIOS",
    "DEFERRED_WEIGHTS",
    "EXPERIMENT_ID",
    "PEER_REGRESSION_SCENARIO",
    "SEEDS",
    "build_deferred_benchmark_manifest",
    "build_deferred_scene",
]
