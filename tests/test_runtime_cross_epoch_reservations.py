"""Regression coverage for Runtime reservations across scheduling epochs."""

from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from evals.benchmarks.binary_offload.spec import SCENARIOS, build_scene
from mars.engine import run_workflow_simulation
from mars.synthetic_workloads import load_default_synthetic_workloads


def test_edge_first_keeps_cross_epoch_runtime_reservations() -> None:
    scenario = next(
        item
        for item in SCENARIOS
        if item["id"] == "multi_robot_mapping"
    )
    catalog = load_default_synthetic_workloads()
    scene = build_scene(scenario, catalog)
    workflow = build_workflow(scene)
    nodes = build_node_specs(scene)
    snapshots = build_node_snapshots(scene)
    links = build_link_specs(scene)
    link_snapshots = build_link_snapshots(scene)
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
        fallback_optimizer=None,
    )

    assert report.task_results
    assert report.metrics["scheduling_epoch_count"] > 1
