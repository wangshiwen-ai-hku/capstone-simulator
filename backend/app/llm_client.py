import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

from openai import OpenAI
from .config import Settings
from .mars_adapter import validate_scene
from .scene_generator import build_deterministic_scene
from .schemas import BenchmarkScene, GenerateSceneRequest


SYSTEM_PROMPT = """
Generate one benchmark scene for stress-testing a multi-robot
cloud-edge-device scheduling platform.
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
     "safety_capable": boolean, "capabilities": [string], "supported_models": [string]}
  ],
  "initial_resources": [
    {"node_id": string, "cpu_util": number, "gpu_util": number, "memory_util": number,
     "temperature_c": number, "power_w": number, "network_latency_ms": number, "online": boolean}
  ],
  "tasks": [
    {"id": string, "name": string, "source_robot_id": string, "task_type": string,
     "task_class": "local_safety"|"realtime_offloadable"|"edge_heavy",
     "priority": integer 1-5, "compute_demand": number, "gpu_demand": number,
     "latency_budget_ms": number, "safety_level": integer 1-5,
     "model_requirement": string, "data_size_mb": number, "output_size_mb": number,
     "bandwidth_requirement_mbps": number, "energy_budget_j": number,
     "allow_local_fallback": boolean,
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
The scene must differentiate DAG-deadline, rule-based, greedy-cost,
local-first, and edge-first scheduling behavior.
Include similar-task resource conflicts, priority differences, long dependency
chains, network bottlenecks, and local fallback.
Dependencies and data_edges must reference tasks in the same response and form a valid directed acyclic graph.
Every data edge must connect declared ports with the same message_type. One producer output may fan out to multiple consumers.
Use exactly these workload semantics: local_safety must stay on its safety-capable source robot;
realtime_offloadable includes YOLO/perception and may run on its source robot or edge;
edge_heavy prefers the edge for VLA/LLM/map/data-heavy work.
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
Benchmark controls:
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
- Classify every task into local_safety, realtime_offloadable, or edge_heavy using the semantics above.
- Include initial CPU/GPU/memory utilization, temperature, power, and network latency.
- Keep utilization values within [0, 1], physical measurements non-negative, and capacities and demands internally consistent.
"""


def generate_scene_with_llm(settings: Settings, req: GenerateSceneRequest) -> BenchmarkScene:
    cfg = settings.current_llm()
    api_key = cfg.get("api_key")
    model = cfg.get("model")
    base_url = cfg.get("base_url")
    logger.info(f"Generating scene with LLM enabled: {req.use_llm}, provider: {cfg.get('provider')}, model: {model}")
    if not req.use_llm or not api_key or not model:
        logger.info("Using deterministic fallback because LLM is disabled or missing config.")
        return build_deterministic_scene(req)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=settings.llm_timeout_seconds)
    try:
        user_prompt = _request_prompt(req)
        logger.info(f"Sending request to LLM base_url: {base_url} model: {model}")
        logger.info(f"LLM request prompt: {user_prompt}")

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
        logger.info(f"Received LLM response, length: {len(content)}")
        logger.info(f"LLM response content: {content}")

        data = _extract_json(content)
        logger.info("Extracted JSON from LLM response")
        scene = BenchmarkScene.model_validate(data)
        validate_scene(scene)
        return scene
    except Exception as exc:
        logger.error(f"LLM generation failed with exception: {type(exc).__name__} - {str(exc)}")
        fallback = build_deterministic_scene(req)
        fallback.stressors.append(f"LLM generation failed, deterministic fallback used: {type(exc).__name__}")
        return fallback
