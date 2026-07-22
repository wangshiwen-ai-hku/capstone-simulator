from __future__ import annotations

import unittest
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app import main as api_main
from backend.app.llm_client import generate_scene_with_llm
from backend.app.mars_adapter import validate_scene
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import GenerateSceneRequest


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_main.app)

    def test_provider_status_never_exposes_api_key(self):
        secret = "sentinel-secret-that-must-not-leak"
        with (
            patch.object(api_main.settings, "model_provider", "openai"),
            patch.object(api_main.settings, "openai_api_key", secret),
        ):
            response = self.client.get("/api/providers")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(secret, response.text)
        self.assertEqual(
            response.json()["current"],
            {
                "provider": "openai",
                "model": api_main.settings.openai_model,
                "configured": True,
            },
        )

    def test_generate_validate_and_simulate_flow(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["system"], "MARS")

        generated = self.client.post(
            "/api/generate-scene",
            json={"robot_count": 2, "edge_count": 1, "use_llm": False, "seed": 11},
        )
        self.assertEqual(generated.status_code, 200)
        scene = generated.json()

        validated = self.client.post("/api/validate-workflow", json=scene)
        self.assertEqual(validated.status_code, 200)
        self.assertTrue(validated.json()["valid"])

        simulated = self.client.post(
            "/api/simulate",
            json={"scene": scene, "algorithm": "dag_deadline", "seed": 11},
        )
        self.assertEqual(simulated.status_code, 200)
        payload = simulated.json()
        self.assertTrue(payload["dag"]["valid"])
        self.assertEqual(payload["metrics"]["task_count"], len(scene["tasks"]))

    def test_workload_catalog_covers_three_classes_and_requested_modules(self):
        response = self.client.get("/api/workload-catalog")
        self.assertEqual(response.status_code, 200)
        workloads = response.json()["workloads"]
        self.assertEqual(
            {item["task_class"] for item in workloads},
            {"local_safety", "realtime_offloadable", "edge_heavy"},
        )
        task_types = {item["task_type"] for item in workloads}
        self.assertTrue(
            {
                "obstacle_avoidance",
                "local_control",
                "environment_understanding",
                "semantic_segmentation",
                "local_llm_7b",
                "local_llm_10b",
            }.issubset(task_types)
        )

    def test_local_runtime_bootstrap_submit_retry_and_result_flow(self):
        bootstrapped = self.client.post("/api/runtime/bootstrap")
        self.assertEqual(bootstrapped.status_code, 200)
        runtime = bootstrapped.json()
        self.assertEqual(
            runtime["topology"],
            {"central_schedulers": 1, "orin_agents": 2, "edge_agents": 1},
        )
        self.assertEqual(len(runtime["agents"]), 3)
        self.assertTrue(all(agent["registered"] for agent in runtime["agents"]))

        scene = self.client.post(
            "/api/generate-scene",
            json={"robot_count": 2, "edge_count": 1, "use_llm": False, "seed": 23},
        ).json()
        accepted = self.client.post(
            "/api/runtime/workflows",
            json={
                "scene": scene,
                "algorithm": "dag_deadline",
                "seed": 23,
                "max_attempts": 2,
                "inject_first_failure": True,
                "failure_task_type": "local_llm_7b",
            },
        )
        self.assertEqual(accepted.status_code, 202)
        run_id = accepted.json()["run_id"]

        payload = None
        for _ in range(100):
            response = self.client.get(f"/api/runtime/workflows/{run_id}")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], "succeeded")
        result = payload["result"]
        self.assertEqual(result["metrics"]["retry_count"], 1)
        self.assertEqual(result["metrics"]["retry_success_count"], 1)
        self.assertEqual(len(result["agents"]), 3)
        self.assertTrue(result["data_edges"])

        events = self.client.get(
            f"/api/runtime/workflows/{run_id}/events?after_sequence=0"
        )
        self.assertEqual(events.status_code, 200)
        event_types = {event["event_type"] for event in events.json()["events"]}
        self.assertIn("attempt_dispatched", event_types)
        self.assertIn("retry_scheduled", event_types)


class LlmFallbackTests(unittest.TestCase):
    def test_invalid_llm_dag_falls_back_to_valid_deterministic_scene(self):
        request = GenerateSceneRequest(robot_count=1, edge_count=1, seed=13, use_llm=True)
        invalid = build_deterministic_scene(request)
        invalid.tasks[1].dependencies = [invalid.tasks[2].id]
        invalid.tasks[2].dependencies = [invalid.tasks[1].id]

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=invalid.model_dump_json())
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response

        settings = SimpleNamespace(
            llm_timeout_seconds=1,
            llm_temperature=0.0,
            current_llm=lambda: {
                "provider": "openai",
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "model": "test-model",
            },
        )
        with patch("backend.app.llm_client.OpenAI", return_value=client):
            scene = generate_scene_with_llm(settings, request)

        validate_scene(scene)
        self.assertTrue(any("deterministic fallback used" in item for item in scene.stressors))


if __name__ == "__main__":
    unittest.main()
