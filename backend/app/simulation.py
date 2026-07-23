"""Run MARS simulations for the FastAPI adapter."""

from mars.engine import run_workflow_simulation

from .mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from .schemas import SimulateRequest, SimulationResponse


def run_simulation(req: SimulateRequest) -> SimulationResponse:
    report = run_workflow_simulation(
        build_workflow(req.scene),
        build_node_specs(req.scene),
        build_node_snapshots(req.scene),
        link_specs=build_link_specs(req.scene),
        link_snapshots=build_link_snapshots(req.scene),
        algorithm=req.algorithm,
        seed=req.seed,
        network_jitter=req.network_jitter,
        resource_noise=req.resource_noise,
    )
    return SimulationResponse.model_validate(report.as_dict())
