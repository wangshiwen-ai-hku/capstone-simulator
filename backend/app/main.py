from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .config import get_settings
from .llm_client import generate_scene_with_llm
from .schemas import GenerateSceneRequest, SimulateRequest
from .simulator import run_simulation

settings = get_settings()

app = FastAPI(
    title="Capstone MARS Benchmark API",
    description="Multi-agent robot scheduling benchmark and simulation generator.",
    version="0.1.0",
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
    }


@app.get("/api/providers")
def providers():
    return {
        "current": settings.current_llm(),
        "available": ["openai", "doubao", "glm", "gemini", "custom"],
        "note": "Use backend/.env to switch provider. The app calls the selected endpoint through the OpenAI-compatible client.",
    }


@app.post("/api/generate-scene")
def generate_scene(req: GenerateSceneRequest):
    logger.info(f"Received request to generate scene: {req.scenario_type.value} with difficulty {req.difficulty.value}")
    return generate_scene_with_llm(settings, req)


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    logger.info(f"Received request to run simulation with algorithm: {req.algorithm}")
    return await run_simulation(req)
