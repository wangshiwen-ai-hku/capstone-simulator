"""Execute binary-offload benchmark cases through the production engine."""

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
from evals.workflow import WorkflowEvaluationWeights
from mars.engine import project_run_artifact, run_workflow_artifact
from mars.optimizers import BinaryOffloadOptimizer, OptimizerRegistry
from mars.synthetic_workloads import load_default_synthetic_workloads

from .audit import (
    optimizer_invocation_summaries,
    scheduling_audit,
    solver_audit,
)
from .spec import (
    BETA_SENSITIVITY,
    FALLBACK_OPTIMIZER,
    FORMAL_BETA,
    FORMAL_EXPERIMENT,
    FORMAL_WEIGHTS,
    METHODS,
    SCENARIOS,
    SEEDS,
    SENSITIVITY_EXPERIMENT,
    build_benchmark_manifest,
    build_scene,
)


@dataclass(frozen=True)
class BenchmarkResults:
    """In-memory rows produced by one complete benchmark invocation."""

    manifest: dict[str, object]
    metric_rows: list[dict[str, object]]
    record_rows: list[dict[str, object]]
    sensitivity_rows: list[dict[str, object]]
    optimizer_epoch_rows: list[dict[str, object]]


def run_benchmark_case(
    *,
    experiment: str,
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
    beta: float = FORMAL_BETA,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Run one method/seed case and return auditable rows without writing files."""

    requested_algorithm = optimizer if policy is None else policy
    binary_optimizer = BinaryOffloadOptimizer(
        alpha=FORMAL_WEIGHTS.success,
        beta=beta,
        gamma=FORMAL_WEIGHTS.utilization,
    )
    registry = OptimizerRegistry()
    registry.register(binary_optimizer)
    case_weights = WorkflowEvaluationWeights(
        success=FORMAL_WEIGHTS.success,
        communication=beta,
        utilization=FORMAL_WEIGHTS.utilization,
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
            f"benchmark:{experiment}:{scenario_id}:"
            f"{requested_algorithm}:{seed}:beta={beta}"
        ),
    )
    report = project_run_artifact(
        artifact,
        evaluation_weights=case_weights,
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    audit = scheduling_audit(report)
    invocation_summaries = optimizer_invocation_summaries(report)
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
    modes = Counter(record.mode for record in report.task_results)
    effective_optimizers = str(audit["effective_optimizers"])
    records = [
        {
            "experiment": experiment,
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
    epoch_rows = [
        {
            "experiment": experiment,
            "scenario": scenario_id,
            "seed": seed,
            "requested_algorithm": requested_algorithm,
            "fallback_enabled": True,
            **item,
        }
        for item in invocation_summaries
        if item.get("optimizer_id") == "binary_offload"
    ]
    row = {
        **report.metrics,
        "experiment": experiment,
        "scenario": scenario_id,
        "seed": seed,
        "logical_workflows": 8,
        "method": requested_algorithm,
        "nominal_optimizer": optimizer,
        "nominal_policy": policy or "binary_offload",
        **audit,
        **solver_audit(requested_algorithm, invocation_summaries),
        "evaluation_success_weight": case_weights.success,
        "evaluation_communication_weight": case_weights.communication,
        "evaluation_utilization_weight": case_weights.utilization,
        "optimizer_success_weight": (
            FORMAL_WEIGHTS.success if optimizer == "binary_offload" else ""
        ),
        "optimizer_communication_weight": (
            beta if optimizer == "binary_offload" else ""
        ),
        "optimizer_utilization_weight": (
            FORMAL_WEIGHTS.utilization
            if optimizer == "binary_offload"
            else ""
        ),
        "local_tasks": modes.get("local", 0)
        + modes.get("fallback_local", 0),
        "edge_tasks": modes.get("edge", 0),
        "wall_clock_ms": round(elapsed_ms, 3),
        "description": description,
    }
    return row, records, epoch_rows


def run_binary_offload_benchmark() -> BenchmarkResults:
    """Run the fixed formal matrix and its separate beta sensitivity study."""

    catalog = load_default_synthetic_workloads()
    metric_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    optimizer_epoch_rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        scene = build_scene(scenario, catalog)
        workflow = build_workflow(scene)
        nodes = build_node_specs(scene)
        snapshots = build_node_snapshots(scene)
        links = build_link_specs(scene)
        link_snapshots = build_link_snapshots(scene)
        for seed in SEEDS:
            for optimizer, policy, description in METHODS:
                row, records, epochs = run_benchmark_case(
                    experiment=FORMAL_EXPERIMENT,
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

    for beta in BETA_SENSITIVITY:
        for scenario in SCENARIOS:
            scene = build_scene(scenario, catalog)
            workflow = build_workflow(scene)
            nodes = build_node_specs(scene)
            snapshots = build_node_snapshots(scene)
            links = build_link_specs(scene)
            link_snapshots = build_link_snapshots(scene)
            for seed in SEEDS:
                row, _, epochs = run_benchmark_case(
                    experiment=SENSITIVITY_EXPERIMENT,
                    scenario_id=str(scenario["id"]),
                    workflow=workflow,
                    nodes=nodes,
                    snapshots=snapshots,
                    links=links,
                    link_snapshots=link_snapshots,
                    seed=seed,
                    optimizer="binary_offload",
                    policy=None,
                    description=METHODS[0][2],
                    beta=beta,
                )
                row["beta"] = beta
                for epoch in epochs:
                    epoch["beta"] = beta
                sensitivity_rows.append(row)
                optimizer_epoch_rows.extend(epochs)

    return BenchmarkResults(
        manifest=build_benchmark_manifest(catalog),
        metric_rows=metric_rows,
        record_rows=record_rows,
        sensitivity_rows=sensitivity_rows,
        optimizer_epoch_rows=optimizer_epoch_rows,
    )


__all__ = [
    "BenchmarkResults",
    "run_benchmark_case",
    "run_binary_offload_benchmark",
]
