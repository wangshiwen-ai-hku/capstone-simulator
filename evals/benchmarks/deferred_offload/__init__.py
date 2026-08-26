"""Importable deferred-offload benchmark runner and reporter."""

from .audit import (
    optimizer_invocation_summaries,
    scheduling_audit,
    solver_audit,
)
from .reporting import (
    ARTIFACT_FILENAMES,
    DEFERRED_SUMMARY_METRICS,
    summarize_results,
    write_deferred_benchmark_artifacts,
)
from .runner import (
    DeferredBenchmarkResults,
    run_deferred_benchmark_case,
    run_deferred_offload_benchmark,
)
from .spec import (
    ASYMMETRIC_PEER_SCENARIO,
    DEFERRED_METHODS,
    DEFERRED_SCENARIOS,
    DEFERRED_WEIGHTS,
    PEER_REGRESSION_SCENARIO,
    build_deferred_benchmark_manifest,
    build_deferred_scene,
)

__all__ = [
    "ASYMMETRIC_PEER_SCENARIO",
    "ARTIFACT_FILENAMES",
    "DEFERRED_METHODS",
    "DEFERRED_SCENARIOS",
    "DEFERRED_SUMMARY_METRICS",
    "DEFERRED_WEIGHTS",
    "DeferredBenchmarkResults",
    "PEER_REGRESSION_SCENARIO",
    "build_deferred_benchmark_manifest",
    "build_deferred_scene",
    "optimizer_invocation_summaries",
    "run_deferred_benchmark_case",
    "run_deferred_offload_benchmark",
    "scheduling_audit",
    "solver_audit",
    "summarize_results",
    "write_deferred_benchmark_artifacts",
]
