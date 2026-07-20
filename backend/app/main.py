from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from edgesched import __version__ as edgesched_version
from edgesched.dag import validate_workflow
from edgesched.models import TASK_CLASS_LABELS, TaskClass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .config import get_settings
from .llm_client import generate_scene_with_llm
from .schemas import BenchmarkScene, GenerateSceneRequest, SimulateRequest
from .simulator import _to_workflow, run_simulation

settings = get_settings()

app = FastAPI(
    title="Unified EdgeSched Simulator API",
    description="DAG-aware robot edge scheduling and benchmark simulation API.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    cfg = settings.current_llm()
    return {
        "status": "ok",
        "provider": cfg["provider"],
        "model": cfg["model"],
        "llm_configured": bool(cfg.get("api_key")),
        "architecture": "unified-edgesched-v2",
        "edgesched_version": edgesched_version,
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
        "core": "edgesched.v2",
        "workflow": "validated DAG with blocked/ready/running/terminal lifecycle",
        "transport": "in_memory",
        "task_classes": [
            {"id": task_class.value, "label": TASK_CLASS_LABELS[task_class]}
            for task_class in TaskClass
        ],
        "deprecated": [
            "backend.app.schedulers v1 isolated-task placement",
            "external scheduler v1 callback without workflow topology",
        ],
    }


@app.post("/api/validate-workflow")
def validate_scene_workflow(scene: BenchmarkScene):
    # Validation and simulation share the same domain conversion.
    request = SimulateRequest(scene=scene)
    index = validate_workflow(_to_workflow(request))
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
    return generate_scene_with_llm(settings, req)


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    logger.info(f"Received request to run simulation with algorithm: {req.algorithm}")
    return await run_simulation(req)
