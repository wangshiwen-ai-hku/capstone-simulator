"""Importable binary-offload benchmark definition, runner, and reporting."""

from .audit import (
    optimizer_invocation_summaries,
    scheduling_audit,
    solver_audit,
)
from .runner import (
    BenchmarkResults,
    run_benchmark_case,
    run_binary_offload_benchmark,
)
from .reporting import (
    ARTIFACT_FILENAMES,
    SUMMARY_METRICS,
    summarize_formal_results,
    summarize_sensitivity_results,
    write_benchmark_artifacts,
)
from .spec import (
    BETA_SENSITIVITY,
    DEFAULT_SOLVE_LIMITS,
    FALLBACK_OPTIMIZER,
    FORMAL_BETA,
    FORMAL_EXPERIMENT,
    FORMAL_WEIGHTS,
    HARDWARE,
    METHODS,
    SCENARIOS,
    SEEDS,
    SENSITIVITY_EXPERIMENT,
    build_benchmark_manifest,
    build_scene,
    profile_summary,
)

__all__ = [
    "ARTIFACT_FILENAMES",
    "BETA_SENSITIVITY",
    "BenchmarkResults",
    "DEFAULT_SOLVE_LIMITS",
    "FALLBACK_OPTIMIZER",
    "FORMAL_BETA",
    "FORMAL_EXPERIMENT",
    "FORMAL_WEIGHTS",
    "HARDWARE",
    "METHODS",
    "SCENARIOS",
    "SEEDS",
    "SENSITIVITY_EXPERIMENT",
    "SUMMARY_METRICS",
    "build_benchmark_manifest",
    "build_scene",
    "optimizer_invocation_summaries",
    "profile_summary",
    "run_benchmark_case",
    "run_binary_offload_benchmark",
    "scheduling_audit",
    "solver_audit",
    "summarize_formal_results",
    "summarize_sensitivity_results",
    "write_benchmark_artifacts",
]
