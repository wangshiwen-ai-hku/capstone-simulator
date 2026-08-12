"""Run Web simulations through the shared MARS runtime boundary."""

from __future__ import annotations

from dataclasses import replace

from mars.domain.topology import LinkSnapshot
from mars.engine import project_run_artifact
from mars.run_artifact import RunArtifact, build_run_artifact

from .mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from .runtime import coordinator_for_scene
from .scheduling import configure_scheduling
from .schemas import SimulateRequest, SimulationResponse


def run_simulation(req: SimulateRequest) -> SimulationResponse:
    """Execute and return the existing Web response contract."""

    response, _ = run_simulation_with_artifact(req)
    return response


def run_simulation_with_artifact(
    req: SimulateRequest,
) -> tuple[SimulationResponse, RunArtifact]:
    """Execute one Web scenario through CentralCoordinator and RuntimePort."""

    workflow = build_workflow(req.scene)
    node_specs = build_node_specs(req.scene)
    node_snapshots = build_node_snapshots(req.scene)
    link_specs = build_link_specs(req.scene)
    link_snapshots = _with_network_jitter(
        build_link_snapshots(req.scene),
        req.network_jitter,
    )
    scheduling = configure_scheduling(
        req.algorithm,
        req.optimizer_options,
        formulation=req.formulation,
        legacy_beta=req.model_dump(include={"beta"}).get("beta"),
    )
    coordinator = coordinator_for_scene(
        req.scene,
        execution_noise=req.resource_noise,
        # The runtime samples the same target-specific success probability
        # used by the optimizer; task accuracy remains a separate quality fact.
        respect_expected_accuracy=True,
        link_snapshots=link_snapshots,
        optimizer_registry=scheduling.registry,
        fallback_optimizer=scheduling.fallback_optimizer,
    )
    report = coordinator.run(
        workflow,
        algorithm=req.algorithm,
        formulation=scheduling.formulation,
        seed=req.seed,
        max_attempts=1,
        deterministic=True,
    )
    artifact = build_run_artifact(
        run_id=f"simulation:{req.scene.id}:{req.seed}",
        workflow=workflow,
        node_specs=node_specs,
        node_snapshots=node_snapshots,
        link_specs=link_specs,
        link_snapshots=link_snapshots,
        profiles=coordinator.profile_catalog.profiles,
        raw_report=report,
        algorithm=req.algorithm,
        formulation=scheduling.formulation,
        seed=req.seed,
        deterministic=True,
        max_attempts=1,
        network_jitter=req.network_jitter,
        resource_noise=req.resource_noise,
    )
    projected = project_run_artifact(
        artifact,
        evaluation_weights=scheduling.evaluation_weights,
    )
    projected = replace(
        projected,
        workflow={
            **projected.workflow,
            "requested_algorithm": req.algorithm,
            "formulation": scheduling.formulation,
            "optimizer_options": dict(scheduling.optimizer_options),
        },
    )
    return SimulationResponse.model_validate(projected.as_dict()), artifact


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
