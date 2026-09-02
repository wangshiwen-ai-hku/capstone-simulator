from __future__ import annotations

import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app import main as api_main
from backend.app.config import Settings
from backend.app.llm_client import (
    _normalize_llm_scene_payload,
    generate_scene_with_llm,
)
from backend.app.mars_adapter import validate_scene
from backend.app.scene_generator import (
    TASK_TYPE_TEMPLATES,
    build_deterministic_scene,
)
from backend.app.schemas import GenerateSceneRequest


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_main.app)

    def _wait_for_run(self, run_id: str) -> dict:
        payload = None
        for _ in range(200):
            response = self.client.get(
                f"/api/runtime/workflows/{run_id}"
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["status"] in {"succeeded", "failed"}:
                return payload
            time.sleep(0.01)
        self.fail(f"runtime workflow did not finish: {payload}")

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

    def test_llm_alias_is_compiled_to_canonical_absolute_tops(self):
        payload = _normalize_llm_scene_payload(
            {
                "tasks": [
                    {
                        "id": "segmentation-task",
                        "task_type": "segmentation",
                        "gpu_demand": 0.85,
                    }
                ]
            }
        )

        self.assertEqual(
            payload["resource_contract_version"],
            "mars.resources.absolute.v1",
        )
        self.assertEqual(
            payload["tasks"][0]["task_type"],
            "semantic_segmentation",
        )
        self.assertEqual(payload["tasks"][0]["gpu_demand"], 36.0)

    def test_deepseek_provider_status_never_exposes_credentials(self):
        secret = "sentinel-deepseek-secret-that-must-not-leak"
        private_base_url = "https://private-gateway.example.invalid/v1"
        with (
            patch.object(api_main.settings, "model_provider", "deepseek"),
            patch.object(api_main.settings, "deepseek_api_key", secret),
            patch.object(
                api_main.settings,
                "deepseek_base_url",
                private_base_url,
            ),
            patch.object(
                api_main.settings,
                "deepseek_model",
                "deepseek-v4-flash",
            ),
        ):
            providers = self.client.get("/api/providers")
            health = self.client.get("/api/health")

        self.assertEqual(providers.status_code, 200)
        self.assertEqual(health.status_code, 200)
        self.assertIn("deepseek", providers.json()["available"])
        self.assertEqual(
            providers.json()["current"],
            {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "configured": True,
            },
        )
        self.assertEqual(health.json()["provider"], "deepseek")
        self.assertEqual(health.json()["model"], "deepseek-v4-flash")
        self.assertTrue(health.json()["llm_configured"])
        self.assertNotIn(secret, providers.text + health.text)
        self.assertNotIn(private_base_url, providers.text + health.text)

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
        self.assertEqual(scene["generation_source"], "deterministic")

        validated = self.client.post("/api/validate-workflow", json=scene)
        self.assertEqual(validated.status_code, 200)
        graph = validated.json()
        self.assertTrue(graph["valid"])
        self.assertEqual(len(graph["tasks"]), len(scene["tasks"]))
        self.assertEqual(graph["depth"], len(graph["level_groups"]))
        self.assertEqual(
            len(graph["data_edges"]),
            len(scene["data_edges"]),
        )
        self.assertTrue(
            all(edge["kind"] == "dependency" for edge in graph["edges"])
        )

        simulated = self.client.post(
            "/api/simulate",
            json={"scene": scene, "algorithm": "dag_deadline", "seed": 11},
        )
        self.assertEqual(simulated.status_code, 200)
        payload = simulated.json()
        self.assertTrue(payload["dag"]["valid"])
        self.assertEqual(payload["metrics"]["task_count"], len(scene["tasks"]))

    def test_workload_catalog_reports_cohorts_and_explicit_placement(self):
        response = self.client.get("/api/workload-catalog")
        self.assertEqual(response.status_code, 200)
        workloads = response.json()["workloads"]
        self.assertEqual(
            {item["task_class"] for item in workloads},
            {"local_safety", "realtime_offloadable", "edge_heavy"},
        )
        task_types = {item["task_type"] for item in workloads}
        self.assertEqual(task_types, set(TASK_TYPE_TEMPLATES))
        self.assertTrue(
            all(
                item["task_class_role"] == "reporting_compatibility"
                for item in workloads
            )
        )
        self.assertTrue(
            all("placement_constraints" in item for item in workloads)
        )

    def test_architecture_exposes_one_runtime_boundary(self):
        response = self.client.get("/api/architecture")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["runtime"],
            "central_scheduler_with_async_runtime_port",
        )
        self.assertEqual(payload["runtime_adapters"], ["in_process"])
        self.assertEqual(
            payload["authoring_modes"],
            ["studio", "authoring_assistant", "templates"],
        )
        self.assertEqual(payload["network_adapters"], [])
        self.assertNotIn("transport_interfaces", payload)

    def test_local_runtime_bootstrap_submit_retry_and_result_flow(self):
        bootstrapped = self.client.post("/api/runtime/bootstrap")
        self.assertEqual(bootstrapped.status_code, 200)
        runtime = bootstrapped.json()
        self.assertEqual(runtime["runtime_adapter_id"], "in_process")
        self.assertEqual(
            runtime["runtime_adapter_implementation"],
            "InProcessRuntimeAdapter",
        )
        self.assertEqual(runtime["topology"]["central_schedulers"], 1)
        self.assertEqual(
            runtime["topology"]["total_agents"],
            len(runtime["agents"]),
        )
        self.assertEqual(
            runtime["topology"]["orin_agents"],
            sum(agent["kind"] == "robot" for agent in runtime["agents"]),
        )
        self.assertEqual(
            runtime["topology"]["edge_agents"],
            sum(agent["kind"] == "edge" for agent in runtime["agents"]),
        )
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

        payload = self._wait_for_run(run_id)
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

    def test_process_local_runtime_accepts_scene_declared_agent_counts(self):
        scene = self.client.post(
            "/api/generate-scene",
            json={"robot_count": 3, "edge_count": 2, "use_llm": False, "seed": 31},
        ).json()

        simulated = self.client.post(
            "/api/simulate",
            json={"scene": scene, "algorithm": "dag_deadline", "seed": 31},
        )
        self.assertEqual(simulated.status_code, 200)

        runtime = self.client.post(
            "/api/runtime/workflows",
            json={"scene": scene, "algorithm": "dag_deadline", "seed": 31},
        )
        self.assertEqual(runtime.status_code, 202)
        payload = self._wait_for_run(runtime.json()["run_id"])
        self.assertEqual(payload["status"], "succeeded")
        agents = payload["result"]["agents"]
        self.assertEqual(
            [agent["kind"] for agent in agents].count("robot"),
            3,
        )
        self.assertEqual(
            [agent["kind"] for agent in agents].count("edge"),
            2,
        )

    def test_process_local_runtime_accepts_robot_only_scene(self):
        scene = self.client.post(
            "/api/generate-scene",
            json={
                "robot_count": 2,
                "edge_count": 0,
                "task_categories": [
                    "localization",
                    "local_control",
                ],
                "use_llm": False,
                "seed": 41,
            },
        )
        self.assertEqual(scene.status_code, 200)
        payload = scene.json()
        self.assertEqual(
            [node["kind"] for node in payload["nodes"]],
            ["robot", "robot"],
        )

        accepted = self.client.post(
            "/api/runtime/workflows",
            json={
                "scene": payload,
                "algorithm": "dag_deadline",
                "seed": 41,
            },
        )
        self.assertEqual(accepted.status_code, 202)
        run = self._wait_for_run(accepted.json()["run_id"])
        self.assertEqual(run["status"], "succeeded")

    def test_failure_injection_requires_a_matching_task_type(self):
        scene = self.client.post(
            "/api/generate-scene",
            json={
                "robot_count": 1,
                "edge_count": 1,
                "task_categories": ["localization"],
                "use_llm": False,
                "seed": 43,
            },
        ).json()

        response = self.client.post(
            "/api/runtime/workflows",
            json={
                "scene": scene,
                "inject_first_failure": True,
                "failure_task_type": "local_llm_7b",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn(
            "failure injection target task type is not present",
            response.json()["detail"],
        )

    def test_all_web_only_task_profiles_execute_in_runtime(self):
        scene = self.client.post(
            "/api/generate-scene",
            json={
                "robot_count": 1,
                "edge_count": 1,
                "task_categories": [
                    "data_compression",
                    "result_verification",
                ],
                "use_llm": False,
                "seed": 47,
            },
        )
        self.assertEqual(scene.status_code, 200)

        accepted = self.client.post(
            "/api/runtime/workflows",
            json={
                "scene": scene.json(),
                "algorithm": "dag_deadline",
                "seed": 47,
            },
        )
        self.assertEqual(accepted.status_code, 202)
        run = self._wait_for_run(accepted.json()["run_id"])
        self.assertEqual(run["status"], "succeeded")
        self.assertEqual(
            {
                result["task_type"]
                for result in run["result"]["task_results"]
            },
            {"data_compression", "result_verification"},
        )

    def test_process_local_runtime_rejects_cloud_nodes(self):
        scene = self.client.post(
            "/api/generate-scene",
            json={
                "robot_count": 2,
                "edge_count": 1,
                "use_llm": False,
                "seed": 37,
            },
        ).json()
        edge = next(node for node in scene["nodes"] if node["kind"] == "edge")
        edge["kind"] = "cloud"

        runtime = self.client.post(
            "/api/runtime/workflows",
            json={"scene": scene, "algorithm": "dag_deadline", "seed": 37},
        )
        simulation = self.client.post(
            "/api/simulate",
            json={"scene": scene, "algorithm": "dag_deadline", "seed": 37},
        )
        self.assertEqual(
            runtime.status_code,
            422,
        )
        self.assertEqual(simulation.status_code, 422)
        self.assertEqual(
            runtime.json()["detail"],
            "the process-local runtime supports robot and edge nodes; "
            "cloud nodes are not supported",
        )
        self.assertEqual(
            simulation.json()["detail"],
            runtime.json()["detail"],
        )


class LlmFallbackTests(unittest.TestCase):
    def test_deepseek_uses_openai_compatible_client(self):
        request = GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            seed=11,
            use_llm=True,
        )
        payload = build_deterministic_scene(request).model_dump_json()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=payload)
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        settings = SimpleNamespace(
            llm_timeout_seconds=12,
            llm_temperature=0.2,
            current_llm=lambda: {
                "provider": "deepseek",
                "api_key": "test-deepseek-key",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            },
        )

        with patch(
            "backend.app.llm_client.OpenAI",
            return_value=client,
        ) as openai_client:
            scene = generate_scene_with_llm(settings, request)

        openai_client.assert_called_once_with(
            api_key="test-deepseek-key",
            base_url="https://api.deepseek.com",
            timeout=12,
        )
        call = client.chat.completions.create.call_args
        self.assertEqual(call.kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(call.kwargs["temperature"], 0.2)
        self.assertEqual(
            call.kwargs["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            [message["role"] for message in call.kwargs["messages"]],
            ["system", "user"],
        )
        validate_scene(scene)
        self.assertEqual(scene.generation_source, "llm")
        self.assertEqual(scene.generation_note, "")

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
        self.assertNotIn("LLM", " ".join(scene.stressors))
        self.assertEqual(
            scene.generation_source,
            "deterministic_fallback",
        )

    def test_llm_null_empty_pin_is_normalized(self):
        request = GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            seed=23,
            use_llm=True,
        )
        payload = build_deterministic_scene(request).model_dump(
            mode="json"
        )
        for task in payload["tasks"]:
            task["placement_constraints"]["pinned_node_id"] = None

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(payload)
                    )
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        settings = SimpleNamespace(
            llm_timeout_seconds=120,
            llm_temperature=0.0,
            current_llm=lambda: {
                "provider": "deepseek",
                "api_key": "test-key",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            },
        )

        with patch(
            "backend.app.llm_client.OpenAI",
            return_value=client,
        ):
            scene = generate_scene_with_llm(settings, request)

        self.assertEqual(scene.generation_source, "llm")
        self.assertTrue(
            all(
                task.placement_constraints.pinned_node_id == ""
                for task in scene.tasks
            )
        )

    def test_llm_conflicting_source_and_explicit_pin_uses_fallback(self):
        request = GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            seed=24,
            use_llm=True,
        )
        payload = build_deterministic_scene(request).model_dump(mode="json")
        placement = payload["tasks"][0]["placement_constraints"]
        placement["pin_to_source"] = True
        placement["pinned_node_id"] = "edge_pc"
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        settings = SimpleNamespace(
            llm_timeout_seconds=120,
            llm_temperature=0.0,
            current_llm=lambda: {
                "provider": "deepseek",
                "api_key": "test-key",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            },
        )

        with patch("backend.app.llm_client.OpenAI", return_value=client):
            scene = generate_scene_with_llm(settings, request)

        self.assertEqual(scene.generation_source, "deterministic_fallback")
        validate_scene(scene)

    def test_llm_scene_with_unsupported_cloud_node_uses_fallback(self):
        request = GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            seed=29,
            use_llm=True,
        )
        payload = build_deterministic_scene(request).model_dump(
            mode="json"
        )
        cloud = dict(payload["nodes"][-1])
        cloud.update(
            {
                "id": "cloud_1",
                "kind": "cloud",
                "display_name": "Cloud 1",
            }
        )
        snapshot = dict(payload["initial_resources"][-1])
        snapshot["node_id"] = "cloud_1"
        payload["nodes"].append(cloud)
        payload["initial_resources"].append(snapshot)

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(payload)
                    )
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        settings = SimpleNamespace(
            llm_timeout_seconds=120,
            llm_temperature=0.0,
            current_llm=lambda: {
                "provider": "deepseek",
                "api_key": "test-key",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            },
        )

        with patch(
            "backend.app.llm_client.OpenAI",
            return_value=client,
        ):
            scene = generate_scene_with_llm(settings, request)

        self.assertEqual(
            scene.generation_source,
            "deterministic_fallback",
        )
        self.assertFalse(any(node.kind == "cloud" for node in scene.nodes))

    def test_llm_scene_without_explicit_placement_uses_fallback(self):
        request = GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            seed=17,
            use_llm=True,
        )
        payload = build_deterministic_scene(request).model_dump(
            mode="json"
        )
        for task in payload["tasks"]:
            task["placement_constraints"] = None

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(payload)
                    )
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

        with patch(
            "backend.app.llm_client.OpenAI",
            return_value=client,
        ):
            scene = generate_scene_with_llm(settings, request)

        self.assertEqual(
            scene.generation_source,
            "deterministic_fallback",
        )
        self.assertTrue(
            all(
                task.placement_constraints is not None
                for task in scene.tasks
            )
        )


class SettingsTests(unittest.TestCase):
    def test_deepseek_provider_selection(self):
        settings = Settings(
            _env_file=None,
            MODEL_PROVIDER=" DeepSeek ",
            DEEPSEEK_API_KEY="test-deepseek-key",
            DEEPSEEK_BASE_URL="https://api.deepseek.com",
            DEEPSEEK_MODEL="deepseek-v4-flash",
        )

        self.assertEqual(
            settings.current_llm(),
            {
                "provider": "deepseek",
                "api_key": "test-deepseek-key",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
            },
        )
        self.assertEqual(
            settings.public_llm(),
            {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "configured": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
