"""Aggregate and serialize binary-offload benchmark results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

from .runner import BenchmarkResults
from .spec import BETA_SENSITIVITY, METHODS, SCENARIOS


ARTIFACT_FILENAMES = (
    "benchmark.json",
    "step3_evaluation_metrics.csv",
    "step3_evaluation_records.json",
    "step3_evaluation_summary.csv",
    "step3_beta_sensitivity.csv",
    "step3_beta_sensitivity_summary.csv",
    "step3_optimizer_epoch_metrics.json",
)

SUMMARY_METRICS = (
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


def summarize_formal_results(
    metric_rows: list[dict[str, object]],
    *,
    scenarios=SCENARIOS,
    methods=METHODS,
) -> list[dict[str, object]]:
    """Compute the same per-scenario/method mean and sample deviation table."""

    summary_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for optimizer, policy, _ in methods:
            method = optimizer if policy is None else policy
            selected = [
                row
                for row in metric_rows
                if row["scenario"] == scenario["id"]
                and row["method"] == method
            ]
            summary: dict[str, object] = {
                "scenario": scenario["id"],
                "method": method,
                "runs": len(selected),
            }
            for metric in SUMMARY_METRICS:
                values = [float(row[metric]) for row in selected]
                summary[f"{metric}_mean"] = round(mean(values), 6)
                summary[f"{metric}_std"] = round(stdev(values), 6)
            summary_rows.append(summary)
    return summary_rows


def summarize_sensitivity_results(
    sensitivity_rows: list[dict[str, object]],
    *,
    beta_values=BETA_SENSITIVITY,
) -> list[dict[str, object]]:
    """Compute the fixed beta-sensitivity aggregate table."""

    sensitivity_summary: list[dict[str, object]] = []
    for beta in beta_values:
        selected = [
            row for row in sensitivity_rows if row["beta"] == beta
        ]
        sensitivity_summary.append(
            {
                "beta": beta,
                "runs": len(selected),
                "edge_tasks_mean": round(
                    mean(float(row["edge_tasks"]) for row in selected),
                    4,
                ),
                "success_reward_mean": round(
                    mean(
                        float(row["expected_success_reward"])
                        for row in selected
                    ),
                    6,
                ),
                "success_ratio_mean": round(
                    mean(
                        float(row["expected_success_ratio"])
                        for row in selected
                    ),
                    6,
                ),
                "communication_ms_mean": round(
                    mean(
                        float(row["communication_time_ms"])
                        for row in selected
                    ),
                    6,
                ),
                "normalized_communication_mean": round(
                    mean(
                        float(row["normalized_communication"])
                        for row in selected
                    ),
                    6,
                ),
                "latency_ms_mean": round(
                    mean(
                        float(row["avg_latency_ms"])
                        for row in selected
                    ),
                    6,
                ),
                "deadline_miss_rate_mean": round(
                    mean(
                        float(row["deadline_miss_rate"])
                        for row in selected
                    ),
                    6,
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
                    mean(
                        float(row["skipped_task_count"])
                        for row in selected
                    ),
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
    return sensitivity_summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_benchmark_artifacts(
    results: BenchmarkResults,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write the benchmark's seven stable output artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_formal_results(results.metric_rows)
    sensitivity_summary = summarize_sensitivity_results(
        results.sensitivity_rows
    )

    _write_json(output_dir / ARTIFACT_FILENAMES[0], results.manifest)
    _write_csv(output_dir / ARTIFACT_FILENAMES[1], results.metric_rows)
    _write_json(output_dir / ARTIFACT_FILENAMES[2], results.record_rows)
    _write_csv(output_dir / ARTIFACT_FILENAMES[3], summary_rows)
    _write_csv(output_dir / ARTIFACT_FILENAMES[4], results.sensitivity_rows)
    _write_csv(output_dir / ARTIFACT_FILENAMES[5], sensitivity_summary)
    _write_json(
        output_dir / ARTIFACT_FILENAMES[6],
        results.optimizer_epoch_rows,
    )
    return tuple(output_dir / name for name in ARTIFACT_FILENAMES)


__all__ = [
    "ARTIFACT_FILENAMES",
    "SUMMARY_METRICS",
    "summarize_formal_results",
    "summarize_sensitivity_results",
    "write_benchmark_artifacts",
]
