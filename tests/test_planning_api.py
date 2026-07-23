from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from backend.app import main as api_main
from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
)
from backend.app.runtime import _runtime_for_scene
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import BenchmarkScene, GenerateSceneRequest
from mars.optimizers import built_in_registry


def _scene():
    return build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=2,
            edge_count=1,
            use_llm=False,
            seed=31,
        )
    )


def test_omitted_links_and_explicit_empty_links_have_distinct_meaning() -> None:
    payload = _scene().model_dump(mode="json")
    payload.pop("links")
    payload.pop("link_snapshots")
    legacy = BenchmarkScene.model_validate(payload)
    explicit_empty = legacy.model_copy(
        update={"links": [], "link_snapshots": []},
        deep=True,
    )
    explicit_null = legacy.model_copy(
        update={"links": None, "link_snapshots": None},
        deep=True,
    )
    node_count = len(legacy.nodes)

    assert len(build_link_specs(legacy)) == node_count * (
        node_count - 1
    )
    assert len(build_link_snapshots(legacy)) == node_count * (
        node_count - 1
    )
    assert len(build_link_specs(explicit_null)) == node_count * (
        node_count - 1
    )
    assert build_link_specs(explicit_empty) == []
    assert build_link_snapshots(explicit_empty) == []


def test_legacy_scene_without_link_fields_still_simulates() -> None:
    payload = _scene().model_dump(mode="json")
    payload.pop("links")
    payload.pop("link_snapshots")

    response = TestClient(api_main.app).post(
        "/api/simulate",
        json={
            "scene": payload,
            "algorithm": "dag_deadline",
            "seed": 31,
            "network_jitter": 0,
            "resource_noise": 0,
        },
    )

    assert response.status_code == 200
    assert response.json()["metrics"]["task_count"] == len(
        payload["tasks"]
    )


def test_complete_legacy_scene_works_for_both_execution_paths() -> None:
    payload = _scene().model_dump(mode="json")
    payload.pop("links")
    payload.pop("link_snapshots")
    for node in payload["nodes"]:
        node.pop("max_concurrency")
    for task in payload["tasks"]:
        task.pop("placement_constraints")

    client = TestClient(api_main.app)
    simulated = client.post(
        "/api/simulate",
        json={
            "scene": payload,
            "algorithm": "dag_deadline",
            "seed": 31,
            "network_jitter": 0,
            "resource_noise": 0,
        },
    )
    accepted = client.post(
        "/api/runtime/workflows",
        json={
            "scene": payload,
            "algorithm": "dag_deadline",
            "seed": 31,
            "max_attempts": 2,
        },
    )

    assert simulated.status_code == 200
    assert accepted.status_code == 202


def test_architecture_reports_the_pluggable_optimizer_pipeline() -> None:
    response = TestClient(api_main.app).get("/api/architecture")

    assert response.status_code == 200
    payload = response.json()
    assert payload["network_model"] == "directed_link_topology"
    assert set(payload["optimizers"]) == set(
        built_in_registry().ids()
    )
    assert payload["planning_pipeline"][2:5] == [
        "scheduling_problem",
        "optimizer",
        "plan_validation_or_repair",
    ]


def test_runtime_uses_scene_declared_max_concurrency() -> None:
    scene = _scene()
    expected = {
        node.id: node.max_concurrency for node in scene.nodes
    }

    async def describe():
        runtime = _runtime_for_scene(scene)
        await runtime.start(0)
        return await runtime.describe(0)

    actual = {
        item["agent_id"]: item["max_concurrency"]
        for item in asyncio.run(describe())
    }

    assert actual == expected
