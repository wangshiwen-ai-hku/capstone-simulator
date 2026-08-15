"""Run the binary-offload benchmark and write its seven artifacts to doc/."""

# ruff: noqa: E402 -- direct script execution bootstraps repository imports.

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    root_path = str(ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

from evals.benchmarks.binary_offload.reporting import (
    write_benchmark_artifacts,
)
from evals.benchmarks.binary_offload.runner import (
    run_binary_offload_benchmark,
)


DOC = ROOT / "doc"


def main() -> None:
    results = run_binary_offload_benchmark()
    write_benchmark_artifacts(results, DOC)
    print(
        f"wrote {len(results.metric_rows)} runs and "
        f"{len(results.record_rows)} task records to {DOC}"
    )


if __name__ == "__main__":
    main()
