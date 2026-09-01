from unittest.mock import MagicMock, patch

from backend.app.agent_service import ModellingSession, MarsAgentService
from backend.app.config import Settings
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import (
    AgentAtomicTaskPlan,
    AgentChatRequest,
    AgentSource,
    Difficulty,
    GenerateSceneRequest,
    TaskCategory,
)


def _service() -> MarsAgentService:
    settings = Settings(_env_file=None)
    settings.apiyi_api_key = None
    settings.agent_web_search = True
    return MarsAgentService(settings)


def test_agent_preflights_only_the_confirmed_plan() -> None:
    service = _service()
    requirements = {
        "description": "A heterogeneous local-only workflow",
        "robot_count": 2,
        "edge_count": 0,
        "robot_hardware": "orin_nano",
    }
    session = ModellingSession(
        requirements=requirements,
        atomic_tasks=[
            AgentAtomicTaskPlan(
                id="task_001",
                name="Local LLM / Robot 1",
                task_type="local_llm_7b",
                source_robot_id="robot_1",
                deadline_ms=5_000,
            ),
            AgentAtomicTaskPlan(
                id="task_002",
                name="Localization / Robot 2",
                task_type="localization",
                source_robot_id="robot_2",
                deadline_ms=1_500,
            ),
        ],
    )
    base_request = GenerateSceneRequest(
        scenario_type="custom",
        custom_scene=requirements["description"],
        robot_count=2,
        edge_count=0,
        task_categories=[TaskCategory.local_llm_7b, TaskCategory.localization],
        difficulty=Difficulty.medium,
        seed=7,
        use_llm=False,
        robot_hardware="orin_nano",
    )
    raw_resources = build_deterministic_scene(
        base_request,
        preflight=False,
    ).initial_resources

    with (
        patch(
            "backend.app.schedulability.ensure_generated_scene_schedulable"
        ) as placeholder_preflight,
        patch(
            "backend.app.agent_service.ensure_generated_scene_schedulable"
        ) as final_preflight,
    ):
        scene = service._compile_scene(session)

    placeholder_preflight.assert_not_called()
    final_preflight.assert_called_once_with(scene)
    assert scene.initial_resources == raw_resources
    assert [task.id for task in scene.tasks] == ["task_001", "task_002"]


def test_english_intake_and_local_only_constraint_survive_later_turns() -> None:
    service = _service()

    discovery = service.chat(AgentChatRequest(
        message=(
            "3 vending robots receive pickup tasks one minute apart; "
            "this workflow is local-only."
        ),
    ))
    review = service.chat(AgentChatRequest(
        thread_id=discovery.thread_id,
        message="Prioritize the shortest completion time.",
    ))
    ready = service.chat(AgentChatRequest(
        thread_id=discovery.thread_id,
        message="Confirm and compile.",
        action="confirm",
    ))

    assert review.phase == "review"
    assert ready.phase == "ready"
    assert ready.scene_draft is not None
    assert sum(node.kind == "robot" for node in ready.scene_draft.nodes) == 3
    assert sum(node.kind == "edge" for node in ready.scene_draft.nodes) == 0
    assert {task.arrival_time_ms for task in ready.scene_draft.tasks} == {
        0,
        60_000,
        120_000,
    }


def test_retrieval_is_opt_in_redacted_and_supplied_to_the_planner() -> None:
    assert AgentChatRequest(message="private workflow").enable_web_search is False

    service = _service()
    response = MagicMock()
    response.read.return_value = b"<feed></feed>"
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    requirements = {
        "description": "Project Nightingale at Secret Lab 42",
        "robot_count": 1,
        "edge_count": 0,
        "task_types": ["object_detection"],
    }
    with patch("backend.app.agent_service.urlopen", return_value=response) as open_url:
        service._retrieve(requirements)

    requested_url = open_url.call_args.args[0].full_url.lower()
    assert "nightingale" not in requested_url
    assert "secret" not in requested_url
    assert "object+detection" in requested_url
    assert "on-device+computing" in requested_url

    session = ModellingSession(
        requirements={"robot_count": 1},
        messages=[{"role": "user", "content": "Plan the workflow"}],
        sources=[AgentSource(
            title="Evidence title",
            url="https://example.test/paper",
            snippet="Evidence summary",
            kind="web",
        )],
    )
    model_payload = """{
      "summary": "A grounded plan",
      "pipelines": {"robot_1": ["localization", "local_planning"]},
      "reasons": {},
      "assumptions": [],
      "insights": []
    }"""
    with patch.object(
        service,
        "_call_api_model",
        return_value=(model_payload, "gemini-3.1-flash-lite", ""),
    ) as call_model:
        service._plan_with_model("gemini-3.1-flash-lite", session)

    context = call_model.call_args.args[2]
    assert context["retrieved_evidence"] == [{
        "title": "Evidence title",
        "url": "https://example.test/paper",
        "snippet": "Evidence summary",
    }]
