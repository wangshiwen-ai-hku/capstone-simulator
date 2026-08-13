from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import certifi

from fastapi.testclient import TestClient

from backend.app import main as api_main
from backend.app.template_store import TemplateStore
from backend.app.agent_service import MarsAgentService
from backend.app.config import Settings


def test_agent_structures_natural_language_as_studio_scene():
    client = TestClient(api_main.app)
    with (
        patch.object(api_main.settings, "agent_web_search", False),
        patch.object(api_main.settings, "apiyi_api_key", None),
    ):
        discovery = client.post(
            "/api/agent/chat",
            json={
                "message": (
                    "一个广场的两个自动售货机器人，间隔一分钟分别收到"
                    "两个取货的任务"
                ),
                "model": "deepseek-v4-flash",
                "enable_web_search": True,
            },
        )
        assert discovery.status_code == 200
        first = discovery.json()
        assert first["phase"] == "discovery"
        assert first["ready_to_import"] is False

        planned = client.post(
            "/api/agent/chat",
            json={
                "thread_id": first["thread_id"],
                "message": "优先最短完工时间，需要视觉识别和路径规划",
                "model": "deepseek-v4-flash",
                "enable_web_search": False,
            },
        )
        assert planned.status_code == 200
        plan = planned.json()
        assert plan["phase"] == "review"
        assert plan["atomic_tasks"]

        response = client.post(
            "/api/agent/chat",
            json={
                "thread_id": first["thread_id"],
                "message": "确认编译",
                "action": "confirm",
                "model": "deepseek-v4-flash",
                "enable_web_search": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready_to_import"] is True
    assert payload["phase"] == "ready"
    scene = payload["scene_draft"]
    assert sum(node["kind"] == "robot" for node in scene["nodes"]) == 2
    assert {task["arrival_time_ms"] for task in scene["tasks"]} == {0.0, 60_000.0}
    assert all(task["placement_constraints"] for task in scene["tasks"])
    assert client.post("/api/validate-workflow", json=scene).status_code == 200


def test_template_save_list_get_and_delete_round_trip():
    client = TestClient(api_main.app)
    scene = client.post(
        "/api/generate-scene",
        json={"robot_count": 2, "edge_count": 1, "use_llm": False},
    ).json()

    with TemporaryDirectory() as directory, patch.object(
        api_main,
        "template_store",
        TemplateStore(directory),
    ):
        created = client.post(
            "/api/templates",
            json={
                "name": "Two robot pickup benchmark",
                "description": "Arrival-time scheduling regression.",
                "tags": ["pickup", "pickup", "scheduling"],
                "scene": scene,
            },
        )
        assert created.status_code == 201
        template = created.json()
        assert template["schema_version"] == "mars.benchmark.template.v1"
        assert template["tags"] == ["pickup", "scheduling"]

        listed = client.get("/api/templates").json()["templates"]
        assert [item["id"] for item in listed] == [template["id"]]
        fetched = client.get(f"/api/templates/{template['id']}")
        assert fetched.json()["scene"] == scene
        assert client.delete(f"/api/templates/{template['id']}").status_code == 204
        assert client.get(f"/api/templates/{template['id']}").status_code == 404


def test_agent_uses_one_bounded_model_turn_then_compiles_confirmed_plan():
    service = MarsAgentService(Settings(_env_file=None))
    service.settings.apiyi_api_key = "test-key"
    service.settings.agent_web_search = False
    discovery_completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='''{
          "summary": "I understand the pickup scenario.",
          "questions": ["What is the optimization goal?"],
          "assumptions": []
        }'''))]
    )
    planning_completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='''{
          "summary": "Planned a compact pickup DAG.",
          "pipelines": {
            "robot_1": ["localization", "local_planning"],
            "robot_2": ["localization", "local_planning"]
          },
          "reasons": {"localization": "pose", "local_planning": "route"},
          "assumptions": [],
          "insights": ["Review before compilation."]
        }'''))]
    )
    client_instance = MagicMock()
    client_instance.chat.completions.create.side_effect = [
        discovery_completion,
        planning_completion,
    ]

    with patch("backend.app.agent_service.OpenAI", return_value=client_instance) as openai:
        first = service.chat(api_main.AgentChatRequest(message="两台机器人执行取货"))
        review = service.chat(api_main.AgentChatRequest(
            thread_id=first.thread_id,
            message="优先最短完工时间",
        ))

    assert review.phase == "review"
    assert [task.task_type for task in review.atomic_tasks] == [
        "localization",
        "local_planning",
        "localization",
        "local_planning",
    ]
    assert openai.call_args.kwargs["timeout"] == service.settings.agent_model_timeout_seconds
    assert openai.call_args.kwargs["max_retries"] == 0

    ready = service.chat(api_main.AgentChatRequest(
        thread_id=first.thread_id,
        message="确认",
        action="confirm",
    ))
    assert ready.phase == "ready"
    assert ready.scene_draft is not None
    assert ready.scene_draft.generation_source == "llm"
    assert [task.id for task in ready.scene_draft.tasks] == [
        "task_001", "task_002", "task_003", "task_004",
    ]


def test_retrieval_uses_certifi_without_ssl_cert_file_export():
    service = MarsAgentService(Settings(_env_file=None))
    response = MagicMock()
    response.read.return_value = b"<feed></feed>"
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with (
        patch("backend.app.agent_service.ssl.create_default_context", wraps=__import__("ssl").create_default_context) as context,
        patch("backend.app.agent_service.urlopen", return_value=response),
    ):
        sources = service._retrieve("robot scheduling")

    context.assert_called_once_with(cafile=certifi.where())
    assert sources[0].kind == "mars"


def test_agent_recovers_semantic_tasks_from_non_json_api_output():
    service = MarsAgentService(Settings(_env_file=None))
    service.settings.apiyi_api_key = "test-key"
    service.settings.agent_web_search = False
    discovery = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='{"summary":"understood","questions":["goal?"],"assumptions":[]}')
    )])
    malformed_plan = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=(
            "For both robots use localization, object_detection, then "
            "local_planning and local_control. This minimizes route delay."
        ))
    )])
    client = MagicMock()
    client.chat.completions.create.side_effect = [discovery, malformed_plan]

    with patch("backend.app.agent_service.OpenAI", return_value=client):
        first = service.chat(api_main.AgentChatRequest(
            message="two robots receive pickup jobs",
            model="gemini-3.1-flash-lite",
        ))
        review = service.chat(api_main.AgentChatRequest(
            thread_id=first.thread_id,
            message="minimize route delay",
            model="gemini-3.1-flash-lite",
            enable_web_search=False,
        ))

    assert review.provenance == "api_recovered"
    assert review.fallback is False
    assert review.effective_model == "gemini-3.1-flash-lite"
    assert {task.task_type for task in review.atomic_tasks} == {
        "localization",
        "object_detection",
        "local_planning",
        "local_control",
    }
    assert "validation failed" in review.diagnostic
