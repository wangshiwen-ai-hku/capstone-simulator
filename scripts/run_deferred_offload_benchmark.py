"""Run deferred-offload comparisons and write evidence to doc/phobos/."""

# ruff: noqa: E402 -- direct script execution bootstraps repository imports.

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    root_path = str(ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

from evals.benchmarks.deferred_offload import (
    run_deferred_offload_benchmark,
    write_deferred_benchmark_artifacts,
)


OUTPUT_DIR = ROOT / "doc" / "phobos"


def main() -> None:
    results = run_deferred_offload_benchmark()
    paths = write_deferred_benchmark_artifacts(results, OUTPUT_DIR)
    print(
        f"wrote {len(results.metric_rows)} runs and "
        f"{len(results.record_rows)} task records to {OUTPUT_DIR}: "
        f"{', '.join(path.name for path in paths)}"
    )


if __name__ == "__main__":
    main()
