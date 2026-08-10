from dataclasses import asdict
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from mars import __version__ as mars_version
from mars.domain.task import TASK_CLASS_LABELS, TaskClass
from mars.optimizers import (
    algorithm_aliases,
    built_in_policy_ids,
    built_in_registry,
)
from mars.synthetic_workloads import load_default_synthetic_workloads

from .config import get_settings
from .llm_client import generate_scene_with_llm
from .mars_adapter import SceneValidationError, validate_scene
from .runtime import runtime_service
from .scheduling import scheduling_capabilities
from .scene_generator import (
    TASK_TYPE_TEMPLATES,
    placement_constraints_for,
)
from .schemas import (
    BenchmarkScene,
    GenerateSceneRequest,
    RuntimeWorkflowRequest,
    SimulateRequest,
)
from .simulation import run_simulation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
synthetic_workloads = load_default_synthetic_workloads()

app = FastAPI(
    title="MARS Simulator API",
    description=(
        "HTTP interface for MARS workflow validation, simulation, and "
        "runtime control."
    ),
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
        "available": [
            "openai",
            "deepseek",
            "doubao",
            "glm",
            "gemini",
            "custom",
        ],
        "note": (
            "Provider configuration is loaded from backend/.env through an "
            "OpenAI-compatible client."
        ),
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
        "scheduling_capabilities": scheduling_capabilities(),
        "task_class_role": "reporting_compatibility",
        "task_classes": [
            {"id": task_class.value, "label": TASK_CLASS_LABELS[task_class]}
            for task_class in TaskClass
        ],
        "placement_contract": {
            "authority": "task.placement_constraints",
            "dimensions": [
                "node_eligibility",
                "node_preference",
                "required_capabilities",
                "source_and_robot_locality",
                "safety",
                "fallback",
                "statefulness",
                "idempotence",
                "splitting",
                "replication",
            ],
        },
    }


@app.get("/api/workload-catalog")
def workload_catalog():
    workloads = []
    for workload in synthetic_workloads:
        item = asdict(workload)
        if workload.task_type in TASK_TYPE_TEMPLATES:
            item["placement_constraints"] = placement_constraints_for(
                workload.task_type
            ).model_dump(mode="json")
        item["task_class_role"] = "reporting_compatibility"
        workloads.append(item)
    return {
        "provenance": "synthetic_placeholder",
        "warning": (
            "Synthetic profiles are local simulation inputs. Deployment "
            "profiles require measured telemetry."
        ),
        "workloads": workloads,
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
    task_by_id = {task.id: task for task in scene.tasks}
    dependency_edges = [
        {
            "from": parent,
            "to": child,
            "source": parent,
            "target": child,
            "kind": "dependency",
        }
        for child in index.topological_order
        for parent in index.parents[child]
    ]
    data_edges = [
        {
            "id": (
                f"{edge.producer_task}.{edge.producer_port}->"
                f"{edge.consumer_task}.{edge.consumer_port}"
            ),
            "from": edge.producer_task,
            "to": edge.consumer_task,
            "source": edge.producer_task,
            "target": edge.consumer_task,
            "kind": "data",
            "producer_task": edge.producer_task,
            "producer_port": edge.producer_port,
            "consumer_task": edge.consumer_task,
            "consumer_port": edge.consumer_port,
            "message_type": edge.message_type,
        }
        for edge in index.data_edges
    ]
    tasks = [
        {
            "id": task_id,
            "task_id": task_id,
            "name": task_by_id[task_id].name,
            "task_type": task_by_id[task_id].task_type,
            "task_class": task_by_id[task_id].task_class.value,
            "level": index.levels[task_id],
            "dependencies": list(index.parents[task_id]),
            "children": list(index.children[task_id]),
            "input_ports": [
                port.model_dump(mode="json")
                for port in task_by_id[task_id].input_ports
            ],
            "output_ports": [
                port.model_dump(mode="json")
                for port in task_by_id[task_id].output_ports
            ],
            "placement_constraints": (
                task_by_id[task_id].placement_constraints.model_dump(
                    mode="json"
                )
                if task_by_id[task_id].placement_constraints is not None
                else None
            ),
        }
        for task_id in index.topological_order
    ]
    level_numbers = sorted(set(index.levels.values()))
    return {
        "valid": True,
        "workflow_id": scene.workflow_id,
        "topological_order": list(index.topological_order),
        "levels": index.levels,
        "level_groups": [
            {
                "level": level,
                "task_ids": [
                    task_id
                    for task_id in index.topological_order
                    if index.levels[task_id] == level
                ],
            }
            for level in level_numbers
        ],
        "roots": [
            task_id
            for task_id in index.topological_order
            if not index.parents[task_id]
        ],
        "leaves": [
            task_id
            for task_id in index.topological_order
            if not index.children[task_id]
        ],
        "depth": max(index.levels.values(), default=-1) + 1,
        "tasks": tasks,
        "edges": dependency_edges,
        "data_edges": data_edges,
    }


@app.post("/api/generate-scene")
def generate_scene(req: GenerateSceneRequest):
    logger.info(
        "Received scene-generation request: scenario=%s difficulty=%s",
        req.scenario_type.value,
        req.difficulty.value,
    )
    scene = generate_scene_with_llm(settings, req)
    _validate_scene_request(scene)
    return scene


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    logger.info(
        "Received simulation request with policy preset=%s",
        req.algorithm,
    )
    _validate_scene_request(req.scene)
    try:
        return run_simulation(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
