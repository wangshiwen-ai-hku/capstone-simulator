from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from mars import __version__ as mars_version
from mars.models import TASK_CLASS_LABELS, TaskClass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .config import get_settings
from .llm_client import generate_scene_with_llm
from .mars_adapter import SceneValidationError, validate_scene
from .schemas import BenchmarkScene, GenerateSceneRequest, SimulateRequest
from .simulation import run_simulation

settings = get_settings()

app = FastAPI(
    title="MARS Simulator API",
    description="Web adapter for DAG-aware MARS scheduling and benchmark simulation.",
    version="0.2.0",
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
    cfg = settings.current_llm()
    return {
        "status": "ok",
        "provider": cfg["provider"],
        "model": cfg["model"],
        "llm_configured": bool(cfg.get("api_key")),
        "system": "MARS",
        "mars_version": mars_version,
    }


@app.get("/api/providers")
def providers():
    return {
        "current": settings.current_llm(),
        "available": ["openai", "doubao", "glm", "gemini", "custom"],
        "note": "Use backend/.env to switch provider. The app calls the selected endpoint through the OpenAI-compatible client.",
    }


@app.get("/api/architecture")
def architecture():
    return {
        "system": "MARS",
        "core_version": mars_version,
        "schema_version": "mars.v1",
        "workflow": "validated DAG with blocked/ready/running/terminal lifecycle",
        "runtime": "in_process_simulation",
        "transport_interfaces": ["in_memory"],
        "task_classes": [
            {"id": task_class.value, "label": TASK_CLASS_LABELS[task_class]}
            for task_class in TaskClass
        ],
    }


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
    logger.info(f"Received request to run simulation with algorithm: {req.algorithm}")
    _validate_scene_request(req.scene)
    return await run_simulation(req)
