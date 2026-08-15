import json
from pathlib import Path
import subprocess
import sys

import pytest

from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from evals.benchmarks.binary_offload.runner import run_benchmark_case
from evals.benchmarks.binary_offload.spec import (
    DEFAULT_SOLVE_LIMITS,
    FORMAL_BETA,
    FORMAL_EXPERIMENT,
    build_scene,
)
from mars.synthetic_workloads import load_default_synthetic_workloads


ROOT = Path(__file__).resolve().parents[1]


def test_package_import_does_not_mutate_sys_path():
    probe = """
import importlib
import json
import sys

before = list(sys.path)
module = importlib.import_module("evals.benchmarks.binary_offload.runner")
after_import = list(sys.path)
importlib.reload(module)
after_reload = list(sys.path)

print(json.dumps({
    "import_unchanged": after_import == before,
    "reload_unchanged": after_reload == after_import,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "import_unchanged": True,
        "reload_unchanged": True,
    }


def test_single_case_uses_production_metrics_and_auditable_solver_metadata():
    catalog = load_default_synthetic_workloads()
    scenario = {
        "id": "benchmark_smoke",
        "name": "Benchmark smoke",
        "description": "One edge-only task per logical workflow.",
        "tasks_per_robot": (("object_detection", ()),),
        "deadline_ms": 500.0,
    }
    scene = build_scene(scenario, catalog)
    for task in scene.tasks:
        task.placement_constraints.allow_source_node = False

    row, records, epochs = run_benchmark_case(
        experiment="smoke_contract",
        scenario_id="benchmark_smoke",
        workflow=build_workflow(scene),
        nodes=build_node_specs(scene),
        snapshots=build_node_snapshots(scene),
        links=build_link_specs(scene),
        link_snapshots=build_link_snapshots(scene),
        seed=7,
        optimizer="binary_offload",
        policy=None,
        description="smoke",
        beta=4.0,
    )

    assert FORMAL_BETA == 1.0
    assert "0.01" not in FORMAL_EXPERIMENT
    assert scene.workflow_deadline_ms == max(task.deadline_ms for task in scene.tasks)

    assert row["requested_algorithm"] == "binary_offload"
    assert row["effective_algorithm"] == "binary_offload"
    assert json.loads(row["effective_optimizers"]) == {"binary_offload": 2}
    assert row["fallback_enabled"] is True
    assert row["fallback_count"] == 0
    assert row["global_workflow_exact"] is False
    assert row["search_scope"] == "receding_horizon_ready_epoch"
    assert row["requested_placement_search_exhaustive"] is True
    assert row["requested_solver_invocations"] == 2
    assert json.loads(row["requested_solver_statuses"]) == {"optimal": 2}
    assert json.loads(row["effective_solver_statuses"]) == {"optimal": 2}
    assert json.loads(row["effective_termination_reasons"]) == {
        "exhaustive_one_hot_search_complete": 2,
    }
    assert row["solve_budget_ms"] == DEFAULT_SOLVE_LIMITS.solve_budget_ms
    assert row["max_iterations"] == DEFAULT_SOLVE_LIMITS.max_iterations
    assert row["solver_deterministic"] is True
    assert row["requested_seed"] == 7
    assert row["solver_random_seed"] == 7
    assert row["deterministic_execution"] is True
    assert row["evaluation_communication_weight"] == 4.0
    assert row["optimizer_communication_weight"] == 4.0

    for key in (
        "expected_success_reward",
        "expected_success_ratio",
        "communication_time_ms",
        "normalized_communication",
        "peak_cpu_utilization",
        "peak_gpu_utilization",
        "peak_memory_utilization",
        "maximum_resource_utilization",
        "workflow_evaluation_objective",
    ):
        assert key in row
    assert row["normalized_communication"] > 0
    assert row["workflow_evaluation_objective"] == pytest.approx(
        -row["expected_success_ratio"]
        + 4.0 * row["normalized_communication"]
        + 2.0 * row["maximum_resource_utilization"],
        abs=2e-6,
    )

    assert len(records) == len(scene.tasks) == 8
    assert {item["mode"] for item in records} == {"edge"}
    assert len(epochs) == 2
    assert all(item["solve_status"] == "optimal" for item in epochs)
    assert all(
        item["solve_budget_ms"] == DEFAULT_SOLVE_LIMITS.solve_budget_ms
        for item in epochs
    )
