from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import main as api_main
from backend.app.mars_adapter import validate_scene
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import (
    Difficulty,
    GenerateSceneRequest,
    TaskCategory,
)


ROOT = Path(__file__).resolve().parents[1]


def _all_task_types_scene():
    return build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            task_categories=list(TaskCategory),
            difficulty=Difficulty.easy,
            seed=19,
            use_llm=False,
        )
    )


def test_generated_web_tasks_have_explicit_multidimensional_placement() -> None:
    scene = _all_task_types_scene()

    assert all(
        task.placement_constraints is not None
        for task in scene.tasks
    )
    fingerprints = {
        task.placement_constraints.model_dump_json()
        for task in scene.tasks
        if task.placement_constraints is not None
    }
    assert len(fingerprints) >= 5


def test_validate_workflow_returns_canonical_typed_graph_payload() -> None:
    scene = _all_task_types_scene()
    expected = validate_scene(scene)

    response = TestClient(api_main.app).post(
        "/api/validate-workflow",
        json=scene.model_dump(mode="json"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["levels"] == expected.levels
    assert payload["topological_order"] == list(
        expected.topological_order
    )
    assert len(payload["data_edges"]) == len(scene.data_edges)
    assert all(
        {
            "producer_task",
            "producer_port",
            "consumer_task",
            "consumer_port",
            "message_type",
        }.issubset(edge)
        for edge in payload["data_edges"]
    )


def test_web_copy_uses_neutral_english_product_language() -> None:
    paths = [
        ROOT / "frontend" / "index.html",
        *sorted((ROOT / "frontend" / "src").glob("*.ts")),
        *sorted((ROOT / "frontend" / "src").glob("*.tsx")),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert text.isascii()
