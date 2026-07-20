from __future__ import annotations

import unittest
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
