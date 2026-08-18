"""Run deferred-offload comparisons through the production MARS engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from time import perf_counter

from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from backend.app.schemas import BenchmarkScene
from evals.benchmarks.binary_offload.audit import (
    optimizer_invocation_summaries,
    scheduling_audit,
)
from evals.benchmarks.binary_offload.spec import (
    FALLBACK_OPTIMIZER,
    FORMAL_WEIGHTS,
    SCENARIOS,
    SEEDS,
    build_benchmark_manifest,
    build_scene,
)
from evals.workflow import WorkflowEvaluationWeights
from mars.domain import TaskClass
from mars.engine import project_run_artifact, run_workflow_artifact
from mars.optimizers import (
    BinaryOffloadOptimizer,
    DeferredOffloadOptimizer,
    OptimizerRegistry,
)
from mars.synthetic_workloads import load_default_synthetic_workloads


EXPERIMENT_ID = "deferred_offload_cross_robot_v1"
DEFERRED_WEIGHTS = {
    "alpha": FORMAL_WEIGHTS.success,
    "beta": FORMAL_WEIGHTS.communication,
    "gamma": FORMAL_WEIGHTS.utilization,
    "delta": 1.0,
}
DEFERRED_METHODS = (
    ("deferred_offload", None, "CP-SAT assign-or-defer"),
    ("binary_offload", None, "bounded exhaustive one-hot placement"),
    ("heuristic", "local_first", "heuristic local-first"),
    ("heuristic", "edge_first", "heuristic edge-first"),
    ("heuristic", "dag_deadline", "heuristic DAG-deadline"),
)


@dataclass(frozen=True)
class DeferredBenchmarkResults:
    manifest: dict[str, object]
    metric_rows: list[dict[str, object]]
    record_rows: list[dict[str, object]]
    optimizer_epoch_rows: list[dict[str, object]]


def build_deferred_scene(
    scenario: dict[str, object],
    catalog,
) -> BenchmarkScene:
    """Reuse the fixed binary scene and only enable peer robot candidates."""

    scene = build_scene(scenario, catalog)
    tasks = []
    for task in scene.tasks:
        placement = task.placement_constraints
        if task.task_class is TaskClass.LOCAL_SAFETY:
            tasks.append(task)
            continue
        assert placement is not None
        tasks.append(
            task.model_copy(
                update={
                    "placement_constraints": placement.model_copy(
                        update={
                            "allowed_node_kinds": ["robot", "edge"],
                            "allow_other_robots": True,
                        }
                    )
                }
            )
        )
    return scene.model_copy(update={"tasks": tasks})


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
    requested_summaries = tuple(
        item
        for item in summaries
        if item.get("optimizer_id") == optimizer
    )
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
            "requested_algorithm": requested_algorithm,
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
            **dict(item),
        }
        for item in summaries
    ]
    row = {
        **report.metrics,
        "case_status": "succeeded",
        "error_type": "",
        "error_message": "",
        "experiment": EXPERIMENT_ID,
        "scenario": scenario_id,
        "seed": seed,
        "method": requested_algorithm,
        "nominal_optimizer": optimizer,
        "nominal_policy": policy or optimizer,
        **audit,
        "requested_solver_invocations": len(requested_summaries),
        "requested_solver_statuses": json.dumps(
            dict(
                sorted(
                    Counter(
                        str(item.get("solve_status", ""))
                        for item in requested_summaries
                    ).items()
                )
            ),
            separators=(",", ":"),
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


def _manifest(catalog) -> dict[str, object]:
    manifest = build_benchmark_manifest(catalog)
    return {
        **manifest,
        "schema_version": "mars.deferred-offload-benchmark.v1",
        "experiment": EXPERIMENT_ID,
        "base_configuration": "mars.binary-offload-benchmark.v2",
        "placement_change": {
            "ordinary_allowed_node_kinds": ["robot", "edge"],
            "ordinary_allow_other_robots": True,
            "local_safety_unchanged": True,
        },
        "deferred_weight": DEFERRED_WEIGHTS["delta"],
        "methods": [
            {
                "optimizer": optimizer,
                "policy": policy,
                "description": description,
            }
            for optimizer, policy, description in DEFERRED_METHODS
        ],
    }


def run_deferred_offload_benchmark() -> DeferredBenchmarkResults:
    """Run the fixed binary scenes with peer placement and deferred CP-SAT."""

    catalog = load_default_synthetic_workloads()
    metric_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    optimizer_epoch_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        scene = build_deferred_scene(scenario, catalog)
        workflow = build_workflow(scene)
        nodes = build_node_specs(scene)
        snapshots = build_node_snapshots(scene)
        links = build_link_specs(scene)
        link_snapshots = build_link_snapshots(scene)
        for seed in SEEDS:
            for optimizer, policy, description in DEFERRED_METHODS:
                requested = optimizer if policy is None else policy
                try:
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
                except Exception as exc:
                    row = {
                        "case_status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "experiment": EXPERIMENT_ID,
                        "scenario": str(scenario["id"]),
                        "seed": seed,
                        "method": requested,
                        "nominal_optimizer": optimizer,
                        "nominal_policy": policy or optimizer,
                        "description": description,
                    }
                    records = []
                    epochs = []
                metric_rows.append(row)
                record_rows.extend(records)
                optimizer_epoch_rows.extend(epochs)
    return DeferredBenchmarkResults(
        manifest=_manifest(catalog),
        metric_rows=metric_rows,
        record_rows=record_rows,
        optimizer_epoch_rows=optimizer_epoch_rows,
    )


__all__ = [
    "DEFERRED_METHODS",
    "DEFERRED_WEIGHTS",
    "DeferredBenchmarkResults",
    "build_deferred_scene",
    "run_deferred_benchmark_case",
    "run_deferred_offload_benchmark",
]
