"""Run MARS simulations for the FastAPI adapter."""

from mars.engine import run_workflow_simulation

from .mars_adapter import build_nodes, build_workflow
from .schemas import SimulateRequest, SimulationResponse


async def run_simulation(req: SimulateRequest) -> SimulationResponse:
    report = run_workflow_simulation(
        build_workflow(req.scene),
        build_nodes(req.scene),
        algorithm=req.algorithm,
        seed=req.seed,
        network_jitter=req.network_jitter,
        resource_noise=req.resource_noise,
    )
    return SimulationResponse.model_validate(report.as_dict())
