import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

from openai import OpenAI

from .config import Settings
from .generators import build_deterministic_scene
from .schemas import BenchmarkScene, GenerateSceneRequest


SYSTEM_PROMPT = """
You are a benchmark designer for a multi-robot cloud-edge-device scheduling platform.
Generate one realistic simulation scene for stress-testing scheduling algorithms.
Return STRICT JSON only. Do not include markdown.
The JSON schema must match the following high-level fields:
{
  "id": string,
  "title": string,
  "natural_language_description": string,
  "scenario_type": string,
  "difficulty": "easy" | "medium" | "hard" | "stress",
  "nodes": [
    {"id": string, "kind": "robot"|"edge"|"cloud", "display_name": string,
     "cpu_capacity": number, "gpu_capacity": number, "memory_gb": number,
     "bandwidth_mbps": number, "base_latency_ms": number, "battery_wh": number|null,
     "safety_capable": boolean}
  ],
  "initial_resources": [
    {"node_id": string, "cpu_util": number, "gpu_util": number, "memory_util": number,
     "temperature_c": number, "power_w": number, "network_latency_ms": number}
  ],
  "tasks": [
    {"id": string, "name": string, "source_robot_id": string, "task_type": string,
     "priority": integer 1-5, "compute_demand": number, "gpu_demand": number,
     "latency_budget_ms": number, "safety_level": integer 1-5,
     "model_requirement": string, "data_size_mb": number,
     "bandwidth_requirement_mbps": number, "energy_budget_j": number,
     "fallback_policy": "local_only"|"edge_preferred"|"local_preferred"|"any",
     "result_verification": string, "arrival_time_ms": number, "deadline_ms": number,
     "dependencies": [string], "expected_accuracy": number}
  ],
  "stressors": [string],
  "success_criteria": [string]
}
Make the benchmark useful for comparing rule-based, greedy-cost, ADMM or primal-dual scheduling.
Include heavy test cases: similar-task conflicts, priority differences, long chains, network bottlenecks, local fallback.
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
Generate a benchmark scene with these controls:
- scenario_type: {req.scenario_type.value}
- custom_scene: {req.custom_scene or "N/A"}
- robot_count: {req.robot_count}
- edge_count: {req.edge_count}
- task_categories: {[x.value for x in req.task_categories]}
- difficulty: {req.difficulty.value}
- seed: {req.seed}

Important domain requirements:
- Robot nodes are Jetson Orin-like and can execute local inference and safety tasks.
- Edge nodes are PC/control-plane-like and can run heavier VLA/LLM/VLM workloads.
- Use workload abstraction fields: task_type, compute_demand, latency_budget, safety_level, model_requirement, data_size, bandwidth_requirement, energy_budget, fallback_policy, result_verification.
- Include realistic initial resources: CPU/GPU/memory utilization, temperature, power, network latency.
- Keep numeric values plausible and internally consistent.
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
        logger.info(f"LLM User Prompt: {user_prompt}")
        
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
        logger.info(f"LLM Response Content: {content}")
        
        data = _extract_json(content)
        logger.info(f"Successfully extracted JSON from LLM response")
        return BenchmarkScene.model_validate(data)
    except Exception as exc:
        logger.error(f"LLM generation failed with exception: {type(exc).__name__} - {str(exc)}")
        fallback = build_deterministic_scene(req)
        fallback.stressors.append(f"LLM generation failed, deterministic fallback used: {type(exc).__name__}")
        return fallback
