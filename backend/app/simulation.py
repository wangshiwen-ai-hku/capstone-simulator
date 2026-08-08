"""Run Web simulations through the shared MARS runtime boundary."""

from __future__ import annotations

from dataclasses import replace

from mars.engine import project_coordinator_report
from mars.domain.topology import LinkSnapshot
from mars.optimizers import BinaryOffloadOptimizer, OptimizerRegistry

from .mars_adapter import (
    build_link_snapshots,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from .runtime import coordinator_for_scene
from .schemas import SimulateRequest, SimulationResponse


def run_simulation(req: SimulateRequest) -> SimulationResponse:
    """Execute one Web scenario through CentralCoordinator and RuntimePort."""

    workflow = build_workflow(req.scene)
    registry = None
    fallback_optimizer = "heuristic"
    if req.algorithm == "binary_offload":
        registry = OptimizerRegistry()
        registry.register(BinaryOffloadOptimizer(beta=req.beta))
        fallback_optimizer = None
    coordinator_kwargs = {
        "execution_noise": req.resource_noise,
        "respect_expected_accuracy": True,
        "link_snapshots": _with_network_jitter(
            build_link_snapshots(req.scene),
            req.network_jitter,
        ),
    }
    if registry is not None:
        coordinator_kwargs.update(
            optimizer_registry=registry,
            fallback_optimizer=fallback_optimizer,
        )
    coordinator = coordinator_for_scene(req.scene, **coordinator_kwargs)
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
    from scripts.run_binary_offload_benchmark import (
        expected_success_reward,
        peak_resource_utilization,
    )

    utilization = peak_resource_utilization(
        projected,
        workflow,
        build_node_specs(req.scene),
        build_node_snapshots(req.scene),
        coordinator.workload_catalog,
    )
    success_reward = expected_success_reward(
        projected,
        workflow,
        build_node_specs(req.scene),
        coordinator.workload_catalog,
    )
    communication_time = round(
        sum(item.communication_time_ms for item in projected.task_results),
        6,
    )
    projected.metrics.update(
        {
            "expected_success_reward": success_reward,
            "communication_time_ms": communication_time,
            "peak_cpu_utilization": utilization["cpu"],
            "peak_gpu_utilization": utilization["gpu"],
            "peak_memory_utilization": utilization["memory"],
            "maximum_resource_utilization": utilization["maximum"],
            "workflow_evaluation_objective": round(
                -success_reward
                + req.beta * communication_time
                + 2.0 * utilization["maximum"],
                6,
            ),
        }
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
