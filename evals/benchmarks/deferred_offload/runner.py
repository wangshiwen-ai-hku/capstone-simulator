"""Run deferred-offload comparisons through the production MARS engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from time import perf_counter

from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from .audit import (
    optimizer_invocation_summaries,
    scheduling_audit,
    solver_audit,
)
from evals.benchmarks.binary_offload.spec import (
    FALLBACK_OPTIMIZER,
)
from evals.workflow import WorkflowEvaluationWeights
from mars.engine import project_run_artifact, run_workflow_artifact
from mars.optimizers import (
    BinaryOffloadOptimizer,
    DeferredOffloadOptimizer,
    OptimizerRegistry,
)
from mars.synthetic_workloads import load_default_synthetic_workloads
from .spec import (
    DEFERRED_METHODS,
    DEFERRED_SCENARIOS,
    DEFERRED_WEIGHTS,
    EXPERIMENT_ID,
    SEEDS,
    build_deferred_benchmark_manifest,
    build_deferred_scene,
)


@dataclass(frozen=True)
class DeferredBenchmarkResults:
    manifest: dict[str, object]
    metric_rows: list[dict[str, object]]
    record_rows: list[dict[str, object]]
    optimizer_epoch_rows: list[dict[str, object]]


def run_deferred_benchmark_case(
    *,
    scenario_id: str,
    workflow,
    nodes,
    snapshots,
    links,
    link_snapshots,
    seed: int,
    optimizer: str,
    policy: str | None,
    description: str,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Run one comparison case and project only production evidence."""

    requested_algorithm = optimizer if policy is None else policy
    registry = OptimizerRegistry()
    registry.register(
        BinaryOffloadOptimizer(
            alpha=DEFERRED_WEIGHTS["alpha"],
            beta=DEFERRED_WEIGHTS["beta"],
            gamma=DEFERRED_WEIGHTS["gamma"],
        )
    )
    registry.register(
        DeferredOffloadOptimizer(
            alpha=DEFERRED_WEIGHTS["alpha"],
            beta=DEFERRED_WEIGHTS["beta"],
            gamma=DEFERRED_WEIGHTS["gamma"],
            delta=DEFERRED_WEIGHTS["delta"],
        )
    )
    case_weights = WorkflowEvaluationWeights(
        success=DEFERRED_WEIGHTS["alpha"],
        communication=DEFERRED_WEIGHTS["beta"],
        utilization=DEFERRED_WEIGHTS["gamma"],
    )
    started = perf_counter()
    artifact = run_workflow_artifact(
        workflow,
        nodes,
        snapshots,
        algorithm=requested_algorithm,
        seed=seed,
        network_jitter=0.0,
        resource_noise=0.05,
        link_specs=links,
        link_snapshots=link_snapshots,
        optimizer_registry=registry,
        fallback_optimizer=FALLBACK_OPTIMIZER,
        run_id=(
            f"benchmark:{EXPERIMENT_ID}:{scenario_id}:"
            f"{requested_algorithm}:{seed}"
        ),
    )
    report = project_run_artifact(artifact, evaluation_weights=case_weights)
    elapsed_ms = (perf_counter() - started) * 1_000.0
    audit = scheduling_audit(report)
    summaries = optimizer_invocation_summaries(report)
    if audit["requested_algorithm"] != requested_algorithm:
        raise RuntimeError(
            "coordinator scheduling audit does not match requested algorithm: "
            f"{audit['requested_algorithm']!r} != {requested_algorithm!r}"
        )
    if audit["requested_seed"] != seed:
        raise RuntimeError(
            "coordinator scheduling audit does not match requested seed: "
            f"{audit['requested_seed']!r} != {seed!r}"
        )
    effective_optimizers = str(audit["effective_optimizers"])
    plans = artifact.scheduling_plans
    deferred_decisions = sum(len(plan.deferred_task_ids) for plan in plans)
    unique_deferred = {
        task_id for plan in plans for task_id in plan.deferred_task_ids
    }
    task_by_id = {task.task_id: task for task in workflow.tasks}
    deferred_penalty = sum(
        2 ** task_by_id[task_id].priority
        for plan in plans
        for task_id in plan.deferred_task_ids
    )
    modes = Counter(record.mode for record in report.task_results)
    records = [
        {
            "experiment": EXPERIMENT_ID,
            "scenario": scenario_id,
            "seed": seed,
            "logical_workflow": record.task_id.split("--", 1)[0],
            "requested_algorithm": requested_algorithm,
            "effective_optimizers": effective_optimizers,
            "fallback_count": audit["fallback_count"],
            **asdict(record),
        }
        for record in report.task_results
    ]
    epochs = [
        {
            "experiment": EXPERIMENT_ID,
            "scenario": scenario_id,
            "seed": seed,
            "requested_algorithm": requested_algorithm,
            "fallback_enabled": True,
            **dict(item),
        }
        for item in summaries
        if item.get("optimizer_id") == optimizer
    ]
    row = {
        **report.metrics,
        "experiment": EXPERIMENT_ID,
        "scenario": scenario_id,
        "seed": seed,
        "logical_workflows": 2,
        "method": requested_algorithm,
        "nominal_optimizer": optimizer,
        "nominal_policy": policy or optimizer,
        **audit,
        **solver_audit(requested_algorithm, summaries),
        "evaluation_success_weight": DEFERRED_WEIGHTS["alpha"],
        "evaluation_communication_weight": DEFERRED_WEIGHTS["beta"],
        "evaluation_utilization_weight": DEFERRED_WEIGHTS["gamma"],
        "optimizer_success_weight": (
            DEFERRED_WEIGHTS["alpha"] if optimizer == "deferred_offload" else ""
        ),
        "optimizer_communication_weight": (
            DEFERRED_WEIGHTS["beta"] if optimizer == "deferred_offload" else ""
        ),
        "optimizer_utilization_weight": (
            DEFERRED_WEIGHTS["gamma"] if optimizer == "deferred_offload" else ""
        ),
        "optimizer_deferred_weight": (
            DEFERRED_WEIGHTS["delta"] if optimizer == "deferred_offload" else ""
        ),
        "deferred_decision_count": deferred_decisions,
        "unique_deferred_task_count": len(unique_deferred),
        "deferred_priority_penalty": deferred_penalty,
        "local_tasks": modes.get("local", 0)
        + modes.get("fallback_local", 0),
        "peer_tasks": modes.get("peer", 0),
        "edge_tasks": modes.get("edge", 0),
        "wall_clock_ms": round(elapsed_ms, 3),
        "description": description,
    }
    return row, records, epochs


def run_deferred_offload_benchmark() -> DeferredBenchmarkResults:
    """Run the asymmetric peer scene with deferred CP-SAT and baselines."""

    catalog = load_default_synthetic_workloads()
    metric_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    optimizer_epoch_rows: list[dict[str, object]] = []
    for scenario in DEFERRED_SCENARIOS:
        scene = build_deferred_scene(scenario, catalog)
        workflow = build_workflow(scene)
        nodes = build_node_specs(scene)
        snapshots = build_node_snapshots(scene)
        links = build_link_specs(scene)
        link_snapshots = build_link_snapshots(scene)
        for seed in SEEDS:
            for optimizer, policy, description in DEFERRED_METHODS:
                row, records, epochs = run_deferred_benchmark_case(
                    scenario_id=str(scenario["id"]),
                    workflow=workflow,
                    nodes=nodes,
                    snapshots=snapshots,
                    links=links,
                    link_snapshots=link_snapshots,
                    seed=seed,
                    optimizer=optimizer,
                    policy=policy,
                    description=description,
                )
                metric_rows.append(row)
                record_rows.extend(records)
                optimizer_epoch_rows.extend(epochs)
    return DeferredBenchmarkResults(
        manifest=build_deferred_benchmark_manifest(catalog),
        metric_rows=metric_rows,
        record_rows=record_rows,
        optimizer_epoch_rows=optimizer_epoch_rows,
    )


__all__ = [
    "DEFERRED_METHODS",
    "DEFERRED_SCENARIOS",
    "DEFERRED_WEIGHTS",
    "DeferredBenchmarkResults",
    "run_deferred_benchmark_case",
    "run_deferred_offload_benchmark",
]
