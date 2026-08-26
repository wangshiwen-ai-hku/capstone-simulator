"""Serialize deferred-offload benchmark evidence without recomputing metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

from evals.benchmarks.binary_offload.reporting import SUMMARY_METRICS
from .runner import DeferredBenchmarkResults
from .spec import DEFERRED_METHODS, DEFERRED_SCENARIOS


ARTIFACT_FILENAMES = (
    "benchmark.json",
    "evaluation_metrics.csv",
    "evaluation_summary.csv",
    "task_records.json",
    "optimizer_epoch_metrics.json",
)
DEFERRED_SUMMARY_METRICS = (
    *SUMMARY_METRICS,
    "peer_tasks",
    "deferred_decision_count",
    "unique_deferred_task_count",
    "deferred_priority_penalty",
    "wall_clock_ms",
)


def summarize_results(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for scenario in DEFERRED_SCENARIOS:
        for optimizer, policy, _ in DEFERRED_METHODS:
            method = optimizer if policy is None else policy
            selected = [
                row
                for row in rows
                if row["scenario"] == scenario["id"]
                and row["method"] == method
            ]
            summary: dict[str, object] = {
                "scenario": scenario["id"],
                "method": method,
                "runs": len(selected),
            }
            for metric in DEFERRED_SUMMARY_METRICS:
                values = [float(row[metric]) for row in selected]
                summary[f"{metric}_mean"] = round(mean(values), 6)
                summary[f"{metric}_std"] = round(stdev(values), 6)
            summaries.append(summary)
    return summaries


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


def write_deferred_benchmark_artifacts(
    results: DeferredBenchmarkResults,
    output_dir: Path,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_results(results.metric_rows)
    _write_json(output_dir / ARTIFACT_FILENAMES[0], results.manifest)
    _write_csv(output_dir / ARTIFACT_FILENAMES[1], results.metric_rows)
    _write_csv(output_dir / ARTIFACT_FILENAMES[2], summary_rows)
    _write_json(output_dir / ARTIFACT_FILENAMES[3], results.record_rows)
    _write_json(
        output_dir / ARTIFACT_FILENAMES[4],
        results.optimizer_epoch_rows,
    )
    return tuple(output_dir / name for name in ARTIFACT_FILENAMES)


__all__ = [
    "ARTIFACT_FILENAMES",
    "DEFERRED_SUMMARY_METRICS",
    "summarize_results",
    "write_deferred_benchmark_artifacts",
]
