"""Deferred benchmark must reuse platform configuration and execution paths."""

import csv
import json

from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from evals.benchmarks.binary_offload.spec import HARDWARE
from evals.benchmarks.deferred_offload.reporting import (
    ARTIFACT_FILENAMES,
    DEFERRED_SUMMARY_METRICS,
    write_deferred_benchmark_artifacts,
)
from evals.benchmarks.deferred_offload.runner import (
    DeferredBenchmarkResults,
    run_deferred_benchmark_case,
)
from evals.benchmarks.deferred_offload.spec import (
    ASYMMETRIC_PEER_SCENARIO,
    DEFERRED_METHODS,
    DEFERRED_SCENARIOS,
    PEER_REGRESSION_SCENARIO,
    build_deferred_benchmark_manifest,
    build_deferred_scene,
)
from mars.domain import TaskClass
from mars.dag import TaskManager
from mars.scheduler import chain_priority_weights
from mars.synthetic_workloads import load_default_synthetic_workloads


def test_manifest_describes_the_actual_deferred_solver() -> None:
    manifest = build_deferred_benchmark_manifest(
        load_default_synthetic_workloads()
    )

    assert manifest["formal_experiment"] == "deferred_offload_cross_robot_v1"
    assert manifest["solver"] == {
        "optimizer": "deferred_offload",
        "formulation": "assign_or_defer",
        "backend": "OR-Tools CP-SAT",
        "search": "cp_sat_assign_or_defer",
        "scope": "receding_horizon_ready_epoch",
        "solve_budget_ms": 50.0,
        "deterministic": True,
    }
    assert "deferred_priority_penalty" in manifest["objective"]
    assert "beta_sensitivity" not in manifest


def test_asymmetric_scene_declares_peer_placement_and_load() -> None:
    catalog = load_default_synthetic_workloads()
    deferred = build_deferred_scene(ASYMMETRIC_PEER_SCENARIO, catalog)

    assert len(deferred.nodes) == HARDWARE["jetson"]["count"] + 1
    assert {task.source_robot_id for task in deferred.tasks} == {"jetson-1"}
    assert len(deferred.tasks) == 2 * len(
        ASYMMETRIC_PEER_SCENARIO["tasks_per_robot"]
    )
    snapshots = {item.node_id: item for item in deferred.initial_resources}
    assert snapshots["jetson-2"].gpu_util < snapshots["jetson-1"].gpu_util
    for task in deferred.tasks:
        assert task.task_class is not TaskClass.LOCAL_SAFETY
        assert task.placement_constraints is not None
        assert task.placement_constraints.allowed_node_kinds == [
                "robot",
                "edge",
        ]
        assert task.placement_constraints.allow_other_robots is True


def test_asymmetric_scene_propagates_descendant_priority_to_roots() -> None:
    scene = build_deferred_scene(
        ASYMMETRIC_PEER_SCENARIO,
        load_default_synthetic_workloads(),
    )
    workflow = build_workflow(scene)
    index = TaskManager().submit(workflow)
    weights = chain_priority_weights(workflow.tasks, index)

    assert weights["wf1_1--object_detection"] == 52.0
    assert weights["wf1_1--localization"] == 24.0
    assert weights["wf1_1--semantic_segmentation"] == 48.0
    assert weights["wf1_1--environment_understanding"] == 16.0


def test_peer_regression_runs_on_other_jetson_through_production_engine() -> None:
    catalog = load_default_synthetic_workloads()
    scene = build_deferred_scene(PEER_REGRESSION_SCENARIO, catalog)

    row, records, _ = run_deferred_benchmark_case(
        scenario_id="peer_regression",
        workflow=build_workflow(scene),
        nodes=build_node_specs(scene),
        snapshots=build_node_snapshots(scene),
        links=build_link_specs(scene),
        link_snapshots=build_link_snapshots(scene),
        seed=7,
        optimizer="deferred_offload",
        policy=None,
        description="peer regression",
    )

    peer_records = [record for record in records if record["mode"] == "peer"]
    assert row["fallback_count"] == 0
    assert row["peer_tasks"] == len(scene.tasks)
    assert {record["target_node_id"] for record in peer_records} == {
        "jetson-2"
    }


def test_asymmetric_peer_case_uses_production_engine() -> None:
    catalog = load_default_synthetic_workloads()
    scenario = ASYMMETRIC_PEER_SCENARIO
    scene = build_deferred_scene(scenario, catalog)
    row, records, epochs = run_deferred_benchmark_case(
        scenario_id=str(scenario["id"]),
        workflow=build_workflow(scene),
        nodes=build_node_specs(scene),
        snapshots=build_node_snapshots(scene),
        links=build_link_specs(scene),
        link_snapshots=build_link_snapshots(scene),
        seed=7,
        optimizer="deferred_offload",
        policy=None,
        description="asymmetric peer benchmark",
    )

    assert row["requested_algorithm"] == "deferred_offload"
    assert row["requested_seed"] == 7
    assert row["effective_algorithm"] == "deferred_offload"
    assert row["fallback_count"] == 0
    assert row["requested_solver_invocations"] > 0
    assert "maximum_resource_utilization" in row
    assert "deferred_decision_count" in row
    assert row["peer_tasks"] > 0
    assert row["deferred_decision_count"] > 0
    assert row["unique_deferred_task_count"] > 0
    assert any(record["target_node_id"] == "jetson-2" for record in records)
    assert len(records) == len(scene.tasks)
    assert epochs
    assert all(item["optimizer_id"] == "deferred_offload" for item in epochs)
    assert all(item["fallback_enabled"] is True for item in epochs)
    assert all("logical_workflow" in record for record in records)


def test_deferred_reporter_writes_five_artifacts(tmp_path) -> None:
    metric_rows = []
    for scenario in DEFERRED_SCENARIOS:
        for optimizer, policy, _ in DEFERRED_METHODS:
            method = optimizer if policy is None else policy
            for value in (1.0, 3.0):
                metric_rows.append(
                    {
                        "scenario": scenario["id"],
                        "method": method,
                        "case_status": "succeeded",
                        **{
                            metric: value
                            for metric in DEFERRED_SUMMARY_METRICS
                        },
                    }
                )
    results = DeferredBenchmarkResults(
        manifest={"schema_version": "test"},
        metric_rows=metric_rows,
        record_rows=[{"task_id": "task"}],
        optimizer_epoch_rows=[{"epoch_id": "epoch"}],
    )

    paths = write_deferred_benchmark_artifacts(results, tmp_path)

    assert tuple(path.name for path in paths) == ARTIFACT_FILENAMES
    assert json.loads((tmp_path / "benchmark.json").read_text())["schema_version"] == "test"
    with (tmp_path / "evaluation_summary.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == len(DEFERRED_SCENARIOS) * len(DEFERRED_METHODS)
