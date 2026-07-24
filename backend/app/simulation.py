"""Run Web simulations through the canonical MARS runtime boundary."""

from __future__ import annotations

from dataclasses import replace

from mars.engine import project_coordinator_report
from mars.domain.topology import LinkSnapshot

from .mars_adapter import (
    build_link_snapshots,
    build_node_specs,
    build_workflow,
)
from .runtime import coordinator_for_scene
from .schemas import SimulateRequest, SimulationResponse


def run_simulation(req: SimulateRequest) -> SimulationResponse:
    """Execute one Web scenario through CentralCoordinator and RuntimePort."""

    workflow = build_workflow(req.scene)
    coordinator = coordinator_for_scene(
        req.scene,
        execution_noise=req.resource_noise,
        respect_expected_accuracy=True,
        link_snapshots=_with_network_jitter(
            build_link_snapshots(req.scene),
            req.network_jitter,
        ),
    )
    report = coordinator.run(
        workflow,
        algorithm=req.algorithm,
        seed=req.seed,
        max_attempts=1,
        deterministic=True,
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
