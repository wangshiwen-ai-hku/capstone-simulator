from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
from dataclasses import asdict

from mars import __version__ as mars_version
from mars.models import TASK_CLASS_LABELS, TaskClass
from mars.optimizers import (
    algorithm_aliases,
    built_in_policy_ids,
    built_in_registry,
)
from mars.synthetic_workloads import load_default_synthetic_workloads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .config import get_settings
from .llm_client import generate_scene_with_llm
from .mars_adapter import SceneValidationError, validate_scene
from .runtime import runtime_service
from .schemas import (
    BenchmarkScene,
    GenerateSceneRequest,
    RuntimeWorkflowRequest,
    SimulateRequest,
)
from .simulation import run_simulation

settings = get_settings()
synthetic_workloads = load_default_synthetic_workloads()

app = FastAPI(
    title="MARS Simulator API",
    description="Web adapter for DAG-aware MARS scheduling and benchmark simulation.",
    version=mars_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_scene_request(scene: BenchmarkScene):
    try:
        return validate_scene(scene)
    except SceneValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/health")
def health():
    cfg = settings.public_llm()
    return {
        "status": "ok",
        "provider": cfg["provider"],
        "model": cfg["model"],
        "llm_configured": cfg["configured"],
        "system": "MARS",
        "mars_version": mars_version,
    }


@app.get("/api/providers")
def providers():
    return {
        "current": settings.public_llm(),
        "available": ["openai", "doubao", "glm", "gemini", "custom"],
        "note": "Use backend/.env to switch provider. The app calls the selected endpoint through the OpenAI-compatible client.",
    }


@app.get("/api/architecture")
def architecture():
    return {
        "system": "MARS",
        "core_version": mars_version,
        "workflow": "validated DAG with blocked/ready/running/terminal lifecycle",
        "runtime": "central_scheduler_with_async_runtime_port",
        "runtime_adapters": ["in_process"],
        "network_adapters": [],
        "network_model": "directed_link_topology",
        "planning_pipeline": [
            "hard_constraint_filtering",
            "candidate_estimation",
            "immutable_scheduling_snapshot",
            "scheduling_policy",
            "solve_limits",
            "scheduling_problem",
            "optimizer",
            "shared_objective_constraint_evaluation",
            "plan_validation_or_fallback",
            "reservation_commit",
        ],
        "optimizers": list(built_in_registry().ids()),
        "policies": list(built_in_policy_ids()),
        "algorithm_aliases": algorithm_aliases(),
        "task_classes": [
            {"id": task_class.value, "label": TASK_CLASS_LABELS[task_class]}
            for task_class in TaskClass
        ],
    }


@app.get("/api/workload-catalog")
def workload_catalog():
    return {
        "provenance": "synthetic_placeholder",
        "warning": "Local simulation values; replace them with measured deployment profiles before production use.",
        "workloads": [asdict(workload) for workload in synthetic_workloads],
    }


@app.post("/api/runtime/bootstrap")
def bootstrap_runtime():
    return runtime_service.bootstrap()


@app.get("/api/runtime")
def runtime_status():
    return runtime_service.status()


@app.get("/api/agents")
def runtime_agents():
    return {"agents": runtime_service.status()["agents"]}


@app.post("/api/runtime/workflows", status_code=202)
def submit_runtime_workflow(req: RuntimeWorkflowRequest):
    _validate_scene_request(req.scene)
    try:
        return runtime_service.submit(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/runtime/workflows/{run_id}")
def get_runtime_workflow(run_id: str):
    payload = runtime_service.get_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="runtime workflow not found")
    return payload


@app.get("/api/runtime/workflows/{run_id}/events")
def get_runtime_events(
    run_id: str,
    after_sequence: int = Query(default=0, ge=0),
):
    payload = runtime_service.events(run_id, after_sequence)
    if payload is None:
        raise HTTPException(status_code=404, detail="runtime workflow not found")
    return payload


@app.post("/api/validate-workflow")
def validate_scene_workflow(scene: BenchmarkScene):
    index = _validate_scene_request(scene)
    return {
        "valid": True,
        "workflow_id": scene.workflow_id,
        "topological_order": list(index.topological_order),
        "levels": index.levels,
        "edges": [
            {"from": parent, "to": child}
            for child in index.topological_order
            for parent in index.parents[child]
        ],
    }


@app.post("/api/generate-scene")
def generate_scene(req: GenerateSceneRequest):
    logger.info(f"Received request to generate scene: {req.scenario_type.value} with difficulty {req.difficulty.value}")
    scene = generate_scene_with_llm(settings, req)
    _validate_scene_request(scene)
    return scene


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    logger.info(f"Received request to run simulation with policy preset: {req.algorithm}")
    _validate_scene_request(req.scene)
    return await run_simulation(req)
