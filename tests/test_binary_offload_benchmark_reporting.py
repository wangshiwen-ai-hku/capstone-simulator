"""Output-contract coverage for the importable benchmark reporter."""

import csv
import json

from evals.benchmarks.binary_offload.reporting import (
    ARTIFACT_FILENAMES,
    SUMMARY_METRICS,
    write_benchmark_artifacts,
)
from evals.benchmarks.binary_offload.runner import BenchmarkResults
from evals.benchmarks.binary_offload.spec import (
    BETA_SENSITIVITY,
    METHODS,
    SCENARIOS,
)


def _formal_row(
    scenario_id: str,
    method: str,
    value: float,
) -> dict[str, object]:
    return {
        "scenario": scenario_id,
        "method": method,
        **{metric: value for metric in SUMMARY_METRICS},
    }


def _sensitivity_row(beta: float) -> dict[str, object]:
    return {
        "beta": beta,
        "edge_tasks": 3.0,
        "expected_success_reward": 0.8,
        "expected_success_ratio": 0.9,
        "communication_time_ms": 12.0,
        "normalized_communication": 0.1,
        "avg_latency_ms": 42.0,
        "deadline_miss_rate": 0.2,
        "executed_deadline_miss_rate": 0.1,
        "required_task_on_time_rate": 0.7,
        "skipped_task_count": 1.0,
        "maximum_resource_utilization": 0.6,
        "workflow_evaluation_objective": 0.5,
    }


def test_reporter_writes_the_seven_stable_artifacts(tmp_path) -> None:
    metric_rows = []
    for scenario in SCENARIOS:
        for optimizer, policy, _ in METHODS:
            method = optimizer if policy is None else policy
            metric_rows.extend(
                (
                    _formal_row(str(scenario["id"]), method, 1.0),
                    _formal_row(str(scenario["id"]), method, 3.0),
                )
            )
    results = BenchmarkResults(
        manifest={"schema_version": "test"},
        metric_rows=metric_rows,
        record_rows=[{"task_id": "task-1"}],
        sensitivity_rows=[
            _sensitivity_row(beta) for beta in BETA_SENSITIVITY
        ],
        optimizer_epoch_rows=[{"epoch": 0}],
    )

    paths = write_benchmark_artifacts(results, tmp_path)

    assert tuple(path.name for path in paths) == ARTIFACT_FILENAMES
    assert {path.name for path in tmp_path.iterdir()} == set(
        ARTIFACT_FILENAMES
    )
    assert json.loads((tmp_path / "benchmark.json").read_text()) == {
        "schema_version": "test"
    }
    assert json.loads(
        (tmp_path / "step3_evaluation_records.json").read_text()
    ) == [{"task_id": "task-1"}]

    with (tmp_path / "step3_evaluation_summary.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        summary_rows = list(csv.DictReader(stream))
    assert len(summary_rows) == len(SCENARIOS) * len(METHODS)
    assert {row["runs"] for row in summary_rows} == {"2"}
    assert {row["expected_success_reward_mean"] for row in summary_rows} == {
        "2.0"
    }

    with (tmp_path / "step3_beta_sensitivity_summary.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        sensitivity_summary = list(csv.DictReader(stream))
    assert len(sensitivity_summary) == len(BETA_SENSITIVITY)
