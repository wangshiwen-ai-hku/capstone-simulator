import json
import logging
import re
from typing import Any, Dict

from openai import OpenAI

from .config import Settings
from .mars_adapter import validate_scene
from .scene_generator import build_deterministic_scene
from .schemas import BenchmarkScene, GenerateSceneRequest


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Generate one multi-robot scheduling scene for MARS.
Return strict JSON without Markdown.
The JSON schema must match the following high-level fields:
{
  "id": string,
  "title": string,
  "natural_language_description": string,
  "scenario_type": string,
  "difficulty": "easy" | "medium" | "hard" | "stress",
  "workflow_id": string,
  "workflow_deadline_ms": number,
  "failure_policy": "skip_descendants" | "fail_fast",
  "nodes": [
    {"id": string, "kind": "robot"|"edge"|"cloud", "display_name": string, "architecture": string,
     "cpu_capacity": number, "gpu_capacity": number, "memory_gb": number,
     "bandwidth_mbps": number, "base_latency_ms": number, "battery_wh": number|null,
     "safety_capable": boolean, "capabilities": [string],
     "supported_models": [string], "max_concurrency": integer}
  ],
  "initial_resources": [
    {"node_id": string, "cpu_util": number, "gpu_util": number, "memory_util": number,
     "temperature_c": number, "power_w": number, "network_latency_ms": number, "online": boolean}
  ],
  "tasks": [
    {"id": string, "name": string, "source_robot_id": string, "task_type": string,
     "task_class": "local_safety"|"realtime_offloadable"|"edge_heavy"|null,
     "priority": integer 1-5, "compute_demand": number, "gpu_demand": number,
     "latency_budget_ms": number, "safety_level": integer 1-5,
     "model_requirement": string, "data_size_mb": number, "output_size_mb": number,
     "bandwidth_requirement_mbps": number, "energy_budget_j": number,
     "allow_local_fallback": boolean,
     "placement_constraints": {
       "pinned_node_id": string, "pin_to_source": boolean,
       "allowed_node_kinds": ["robot"|"edge"|"cloud"],
       "preferred_node_kinds": ["robot"|"edge"|"cloud"],
       "required_capabilities": [string],
       "allow_source_node": boolean, "allow_other_robots": boolean,
       "safety_required": boolean, "allow_fallback": boolean,
       "stateful": boolean, "idempotent": boolean,
       "splittable": boolean, "replicable": boolean
     },
     "result_verification": string, "arrival_time_ms": number, "deadline_ms": number,
     "dependencies": [string], "stage_index": integer, "expected_accuracy": number,
     "input_ports": [{"name": string, "message_type": string}],
     "output_ports": [{"name": string, "message_type": string}]}
  ],
  "data_edges": [
    {"producer_task": string, "producer_port": string,
     "consumer_task": string, "consumer_port": string, "message_type": string}
  ],
  "stressors": [string],
  "success_criteria": [string]
}
Include similar-task resource conflicts, priority differences, long dependency
chains, network bottlenecks, and local fallback conditions.
Dependencies and data_edges must reference tasks in the same response and form a valid directed acyclic graph.
Every data edge must connect declared ports with the same message_type. One producer output may fan out to multiple consumers.
Every task must include placement_constraints. These constraints are the
authoritative scheduling contract and must be chosen from the task's concrete
execution requirements. preferred_node_kinds must be a subset of
allowed_node_kinds. A source-pinned safety task must set pin_to_source=true,
safety_required=true, allow_fallback=false, and require the local_safety
capability. Perception and model inference tasks may allow both robot and edge
nodes when both satisfy their capabilities. VLA, LLM, map, and data-intensive
tasks may prefer edge nodes without making the reporting cohort a placement
rule. task_class is optional compatibility metadata used only for aggregate
reporting.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
        raise


def _request_prompt(req: GenerateSceneRequest) -> str:
    return f"""
Scene controls:
- scenario_type: {req.scenario_type.value}
- custom_scene: {req.custom_scene or "N/A"}
- robot_count: {req.robot_count}
- edge_count: {req.edge_count}
- task_categories: {[x.value for x in req.task_categories]}
- difficulty: {req.difficulty.value}
- seed: {req.seed}

Domain constraints:
- Robot nodes are Jetson Orin-like and can execute local inference and safety tasks.
- Edge nodes are PC/control-plane-like and can run heavier VLA/LLM/VLM workloads.
- Use workload abstraction fields: task_type, compute_demand, latency_budget, safety_level, model_requirement, data_size, bandwidth_requirement, energy_budget, allow_local_fallback, result_verification.
- Build per-robot DAG pipelines with typed input/output ports, explicit data_edges, dependencies and stage_index. Never emit a cycle or a missing dependency.
- Define explicit placement_constraints for every task from its task_type, safety, state, capabilities, and data-locality requirements.
- Treat task_class as optional reporting metadata, not as the source of placement constraints.
- Ensure every required capability is declared by every node type eligible to execute the task.
- Include initial CPU/GPU/memory utilization, temperature, power, and network latency.
- Keep utilization values within [0, 1], physical measurements non-negative, and capacities and demands internally consistent.
"""


def _deterministic_result(
    req: GenerateSceneRequest,
    *,
    fallback: bool = False,
    note: str = "",
) -> BenchmarkScene:
    scene = build_deterministic_scene(req)
    if fallback:
        scene.generation_source = "deterministic_fallback"
        scene.generation_note = note
    return scene


def _validate_llm_contract(
    scene: BenchmarkScene,
    req: GenerateSceneRequest,
) -> None:
    missing = [
        task.id
        for task in scene.tasks
        if task.placement_constraints is None
    ]
    if missing:
        raise ValueError(
            "LLM scene omitted placement_constraints for tasks: "
            + ", ".join(missing)
        )

    robot_count = sum(node.kind == "robot" for node in scene.nodes)
    edge_count = sum(node.kind == "edge" for node in scene.nodes)
    if (robot_count, edge_count) != (req.robot_count, req.edge_count):
        raise ValueError(
            "LLM scene topology does not match requested robot/edge counts"
        )


def generate_scene_with_llm(settings: Settings, req: GenerateSceneRequest) -> BenchmarkScene:
    cfg = settings.current_llm()
    api_key = cfg.get("api_key")
    model = cfg.get("model")
    base_url = cfg.get("base_url")
    logger.info(
        "Scene generation requested with use_llm=%s provider=%s model=%s",
        req.use_llm,
        cfg.get("provider"),
        model,
    )
    if not req.use_llm:
        return _deterministic_result(req)
    if not api_key or not model:
        logger.warning(
            "LLM generation requested without a complete provider configuration"
        )
        return _deterministic_result(
            req,
            fallback=True,
            note="LLM provider configuration is incomplete",
        )

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=settings.llm_timeout_seconds)
    try:
        user_prompt = _request_prompt(req)
        logger.info(
            "Sending scene-generation request to provider=%s model=%s",
            cfg.get("provider"),
            model,
        )

        resp = client.chat.completions.create(
            model=model,
            temperature=settings.llm_temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        logger.info("Received scene-generation response (%d bytes)", len(content))

        data = _extract_json(content)
        scene = BenchmarkScene.model_validate(data)
        _validate_llm_contract(scene, req)
        validate_scene(scene)
        scene.generation_source = "llm"
        scene.generation_note = ""
        return scene
    except Exception as exc:
        logger.error(
            "LLM generation failed with %s: %s",
            type(exc).__name__,
            exc,
        )
        return _deterministic_result(
            req,
            fallback=True,
            note=(
                f"{type(exc).__name__}: LLM scene generation or validation "
                "failed"
            ),
        )
