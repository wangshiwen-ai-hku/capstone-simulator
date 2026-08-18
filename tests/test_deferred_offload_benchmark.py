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
from evals.benchmarks.binary_offload.spec import HARDWARE, SCENARIOS, build_scene
from evals.benchmarks.deferred_offload.reporting import (
    ARTIFACT_FILENAMES,
    DEFERRED_SUMMARY_METRICS,
    write_deferred_benchmark_artifacts,
)
from evals.benchmarks.deferred_offload.runner import (
    DEFERRED_METHODS,
    DeferredBenchmarkResults,
    build_deferred_scene,
    run_deferred_benchmark_case,
)
from mars.domain import TaskClass
from mars.synthetic_workloads import load_default_synthetic_workloads


def test_deferred_scene_only_changes_ordinary_placement() -> None:
    catalog = load_default_synthetic_workloads()
    original = build_scene(SCENARIOS[0], catalog)
    deferred = build_deferred_scene(SCENARIOS[0], catalog)

    assert deferred.nodes == original.nodes
    assert deferred.initial_resources == original.initial_resources
    assert deferred.links == original.links
    assert deferred.link_snapshots == original.link_snapshots
    assert len(deferred.nodes) == HARDWARE["jetson"]["count"] + 1
    assert [task.arrival_time_ms for task in deferred.tasks] == [
        task.arrival_time_ms for task in original.tasks
    ]
    for before, after in zip(original.tasks, deferred.tasks, strict=True):
        assert after.id == before.id
        if after.task_class is TaskClass.LOCAL_SAFETY:
            assert after.placement_constraints == before.placement_constraints
        else:
            assert after.placement_constraints is not None
            assert after.placement_constraints.allowed_node_kinds == [
                "robot",
                "edge",
            ]
            assert after.placement_constraints.allow_other_robots is True


def test_deferred_smoke_case_uses_production_engine() -> None:
    catalog = load_default_synthetic_workloads()
    scenario = {
        "id": "deferred_smoke",
        "name": "Deferred smoke",
        "description": "One ordinary task per logical workflow.",
        "tasks_per_robot": (("object_detection", ()),),
        "deadline_ms": 500.0,
    }
    scene = build_deferred_scene(scenario, catalog)
    row, records, epochs = run_deferred_benchmark_case(
        scenario_id="deferred_smoke",
        workflow=build_workflow(scene),
        nodes=build_node_specs(scene),
        snapshots=build_node_snapshots(scene),
        links=build_link_specs(scene),
        link_snapshots=build_link_snapshots(scene),
        seed=7,
        optimizer="deferred_offload",
        policy=None,
        description="smoke",
    )

    assert row["requested_algorithm"] == "deferred_offload"
    assert row["fallback_count"] == 0
    assert row["requested_solver_invocations"] > 0
    assert "maximum_resource_utilization" in row
    assert "deferred_decision_count" in row
    assert "peer_tasks" in row
    assert len(records) == len(scene.tasks)
    assert epochs


def test_deferred_reporter_writes_five_artifacts(tmp_path) -> None:
    metric_rows = []
    for scenario in SCENARIOS:
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
    assert len(summaries) == len(SCENARIOS) * len(DEFERRED_METHODS)
