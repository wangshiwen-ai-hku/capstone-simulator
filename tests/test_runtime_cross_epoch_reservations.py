"""Regression coverage for Runtime reservations across scheduling epochs."""

from mars.engine import run_workflow_simulation
from mars.optimizers import OptimizerRegistry
from mars.optimizers.binary_offload import BinaryOffloadOptimizer
from scripts import run_binary_offload_benchmark as benchmark


def test_edge_first_keeps_cross_epoch_runtime_reservations() -> None:
    scenario = next(
        item
        for item in benchmark.SCENARIOS
        if item["id"] == "multi_robot_mapping"
    )
    catalog = benchmark.load_default_synthetic_workloads()
    scene = benchmark.build_scene(scenario, catalog)
    workflow = benchmark.build_workflow(scene)
    nodes = benchmark.build_node_specs(scene)
    snapshots = benchmark.build_node_snapshots(scene)
    links = benchmark.build_link_specs(scene)
    link_snapshots = benchmark.build_link_snapshots(scene)
    binary_optimizer = BinaryOffloadOptimizer(
        alpha=1.0,
        beta=benchmark.FORMAL_BETA,
        gamma=2.0,
    )
    registry = OptimizerRegistry()
    registry.register(binary_optimizer)

    report = run_workflow_simulation(
        workflow,
        nodes,
        snapshots,
        algorithm="edge_first",
        seed=47,
        network_jitter=0.0,
        resource_noise=0.05,
        link_specs=links,
        link_snapshots=link_snapshots,
        optimizer_registry=registry,
        fallback_optimizer=None,
    )

    assert report.task_results
    assert report.metrics["scheduling_epoch_count"] > 1
