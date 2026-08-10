"""Regression coverage for time-aware binary-offload capacity."""

from mars.engine import run_workflow_simulation
from mars.optimizers import OptimizerRegistry
from mars.optimizers.binary_offload import BinaryOffloadOptimizer
from scripts import run_binary_offload_benchmark as benchmark


def test_high_load_binary_offload_can_use_future_capacity() -> None:
    scenario = next(
        item
        for item in benchmark.SCENARIOS
        if item["id"] == "high_load"
    )
    catalog = benchmark.load_default_synthetic_workloads()
    scene = benchmark.build_scene(scenario, catalog)
    workflow = benchmark.build_workflow(scene)
    nodes = benchmark.build_node_specs(scene)
    snapshots = benchmark.build_node_snapshots(scene)
    links = benchmark.build_link_specs(scene)
    link_snapshots = benchmark.build_link_snapshots(scene)
    optimizer = BinaryOffloadOptimizer(
        alpha=1.0,
        beta=benchmark.FORMAL_BETA,
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
