"""Run Web simulations through the shared MARS runtime boundary."""

from __future__ import annotations

from dataclasses import replace

from mars.engine import project_coordinator_report
from mars.domain.topology import LinkSnapshot
from mars.workflow_metrics import evaluate_workflow_metrics

from .mars_adapter import (
    build_link_snapshots,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from .runtime import coordinator_for_scene
from .scheduling import configure_scheduling
from .schemas import SimulateRequest, SimulationResponse


def run_simulation(req: SimulateRequest) -> SimulationResponse:
    """Execute one Web scenario through CentralCoordinator and RuntimePort."""

    workflow = build_workflow(req.scene)
    scheduling = configure_scheduling(
        req.algorithm,
        req.optimizer_options,
        legacy_beta=req.model_dump(include={"beta"}).get("beta"),
    )
    coordinator = coordinator_for_scene(
        req.scene,
        execution_noise=req.resource_noise,
        # The runtime samples the same target-specific success probability
        # used by the optimizer; task accuracy remains a separate quality fact.
        respect_expected_accuracy=True,
        link_snapshots=_with_network_jitter(
            build_link_snapshots(req.scene),
            req.network_jitter,
        ),
        optimizer_registry=scheduling.registry,
        fallback_optimizer=scheduling.fallback_optimizer,
    )
    report = coordinator.run(
        workflow,
        algorithm=req.algorithm,
        seed=req.seed,
        max_attempts=1,
        deterministic=True,
    )
    report = replace(
        report,
        metrics={
            **report.metrics,
            **evaluate_workflow_metrics(
                report.task_results,
                workflow,
                build_node_specs(req.scene),
                build_node_snapshots(req.scene),
                coordinator.profile_catalog,
                weights=scheduling.evaluation_weights,
            ),
        },
        workflow={
            **report.workflow,
            "requested_algorithm": req.algorithm,
            "optimizer_options": dict(scheduling.optimizer_options),
            "metric_schema_version": "mars.workflow-metrics.v1",
        },
    )
    projected = project_coordinator_report(
        report,
        workflow,
        build_node_specs(req.scene),
        algorithm=req.algorithm,
        profiles=coordinator.profile_catalog,
        network_jitter=req.network_jitter,
        resource_noise=req.resource_noise,
    )
    return SimulationResponse.model_validate(projected.as_dict())


def _with_network_jitter(
    snapshots: list[LinkSnapshot],
    jitter_ratio: float,
) -> tuple[LinkSnapshot, ...]:
    """Apply the declared deterministic network disturbance to link facts."""

    return tuple(
        replace(
            snapshot,
            jitter_ms=(
                snapshot.jitter_ms
                + max(1.0, snapshot.latency_ms) * jitter_ratio
            ),
        )
        for snapshot in snapshots
    )
