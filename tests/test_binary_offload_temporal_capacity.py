"""Regression coverage for time-aware binary-offload capacity."""

from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from evals.benchmarks.binary_offload.spec import (
    FORMAL_BETA,
    SCENARIOS,
    build_scene,
)
from mars.engine import run_workflow_simulation
from mars.optimizers import OptimizerRegistry
from mars.optimizers.binary_offload import BinaryOffloadOptimizer
from mars.synthetic_workloads import load_default_synthetic_workloads


def test_high_load_binary_offload_can_use_future_capacity() -> None:
    scenario = next(
        item
        for item in SCENARIOS
        if item["id"] == "high_load"
    )
    catalog = load_default_synthetic_workloads()
    scene = build_scene(scenario, catalog)
    workflow = build_workflow(scene)
    nodes = build_node_specs(scene)
    snapshots = build_node_snapshots(scene)
    links = build_link_specs(scene)
    link_snapshots = build_link_snapshots(scene)
    optimizer = BinaryOffloadOptimizer(
        alpha=1.0,
        beta=FORMAL_BETA,
        gamma=2.0,
    )
    registry = OptimizerRegistry()
    registry.register(optimizer)

    report = run_workflow_simulation(
        workflow,
        nodes,
        snapshots,
        algorithm="binary_offload",
        seed=27,
        network_jitter=0.0,
        resource_noise=0.05,
        link_specs=links,
        link_snapshots=link_snapshots,
        optimizer_registry=registry,
        fallback_optimizer=None,
    )

    assert len(report.task_results) == 32
    assert report.metrics["scheduling_epoch_count"] > 1
