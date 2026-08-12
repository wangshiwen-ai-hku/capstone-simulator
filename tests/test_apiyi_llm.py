from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app import main as api_main
from backend.app.config import Settings
from backend.app.llm_client import generate_scene_with_llm
from backend.app.mars_adapter import validate_scene
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import GenerateSceneRequest, TaskCategory


@pytest.mark.integration
def test_apiyi_live_generate_scene_workflow() -> None:
    """Call APIYI to generate a validated scene (requires backend/.env)."""
    if os.environ.get("RUN_APIYI_LIVE", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_APIYI_LIVE=1 to call the APIYI endpoint")

    settings = Settings()
    settings.llm_timeout_seconds = max(settings.llm_timeout_seconds, 300)
    provider = settings.model_provider.lower().strip()
    if provider != "apiyi":
        pytest.skip(f"MODEL_PROVIDER is {provider!r}, not apiyi")
    if not settings.apiyi_api_key:
        pytest.skip("APIYI_KEY is not set in backend/.env")

    request = GenerateSceneRequest(
        robot_count=1,
        edge_count=1,
        seed=99,
        use_llm=True,
        task_categories=[
            TaskCategory.localization,
            TaskCategory.local_planning,
            TaskCategory.obstacle_avoidance,
        ],
    )
    scene = generate_scene_with_llm(settings, request)
    validate_scene(scene)

    assert scene.generation_source == "llm", (
        f"expected LLM scene, got {scene.generation_source!r}: "
        f"{scene.generation_note}"
    )
    assert scene.workflow_id
    assert len(scene.tasks) >= 1
    assert len(scene.data_edges) >= 0
    print(
        f"workflow_id={scene.workflow_id} tasks={len(scene.tasks)} "
        f"data_edges={len(scene.data_edges)} title={scene.title!r}"
    )


class ApiyiSettingsTests(unittest.TestCase):
    def test_apiyi_provider_selection_from_env_fields(self) -> None:
        settings = Settings(
            _env_file=None,
            MODEL_PROVIDER=" APIYI ",
            APIYI_KEY="test-apiyi-key",
            APIYI_BASE_URL="https://api.apiyi.com/v1",
            APIYI_MODEL="deepseek-v4-flash",
        )

        self.assertEqual(
            settings.current_llm(),
            {
                "provider": "apiyi",
                "api_key": "test-apiyi-key",
                "base_url": "https://api.apiyi.com/v1",
                "model": "deepseek-v4-flash",
            },
        )
        self.assertEqual(
            settings.public_llm(),
            {
                "provider": "apiyi",
                "model": "deepseek-v4-flash",
                "configured": True,
            },
        )


class ApiyiApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_main.app)

    def test_apiyi_provider_status_never_exposes_credentials(self) -> None:
        secret = "sentinel-apiyi-secret-that-must-not-leak"
        with (
            patch.object(api_main.settings, "model_provider", "apiyi"),
            patch.object(api_main.settings, "apiyi_api_key", secret),
            patch.object(
                api_main.settings,
                "apiyi_base_url",
                "https://api.apiyi.com/v1",
            ),
            patch.object(
                api_main.settings,
                "apiyi_model",
                "deepseek-v4-flash",
            ),
        ):
            providers = self.client.get("/api/providers")
            health = self.client.get("/api/health")

        self.assertEqual(providers.status_code, 200)
        self.assertEqual(health.status_code, 200)
        self.assertIn("apiyi", providers.json()["available"])
        self.assertEqual(
            providers.json()["current"],
            {
                "provider": "apiyi",
                "model": "deepseek-v4-flash",
                "configured": True,
            },
        )
        self.assertEqual(health.json()["provider"], "apiyi")
        self.assertEqual(health.json()["model"], "deepseek-v4-flash")
        self.assertTrue(health.json()["llm_configured"])
        self.assertNotIn(secret, providers.text + health.text)

    def test_apiyi_scene_generation_uses_openai_compatible_client(self) -> None:
        request = GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            seed=41,
            use_llm=True,
        )
        payload = build_deterministic_scene(request).model_dump_json()
        midpoint = len(payload) // 2
        response = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=payload[:midpoint]),
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=payload[midpoint:]),
                        )
                    ]
                ),
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        settings = SimpleNamespace(
            llm_timeout_seconds=30,
            llm_temperature=0.35,
            llm_max_retries=1,
            llm_stream_responses=True,
            current_llm=lambda: {
                "provider": "apiyi",
                "api_key": "test-apiyi-key",
                "base_url": "https://api.apiyi.com/v1",
                "model": "deepseek-v4-flash",
            },
        )

        with patch(
            "backend.app.llm_client.OpenAI",
            return_value=client,
        ) as openai_client:
            scene = generate_scene_with_llm(settings, request)

        openai_client.assert_called_once_with(
            api_key="test-apiyi-key",
            base_url="https://api.apiyi.com/v1",
            timeout=30,
            max_retries=1,
        )
        call = client.chat.completions.create.call_args
        self.assertEqual(call.kwargs["model"], "deepseek-v4-flash")
        self.assertTrue(call.kwargs["stream"])
        validate_scene(scene)
        self.assertEqual(scene.generation_source, "llm")


if __name__ == "__main__":
    unittest.main()
