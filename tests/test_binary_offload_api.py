from __future__ import annotations

import time

from fastapi.testclient import TestClient

from backend.app import main as api_main


def _scene(
    client: TestClient,
    *,
    edge_count: int = 2,
    robot_count: int = 2,
    task_categories: list[str] | None = None,
) -> dict[str, object]:
    request: dict[str, object] = {
        "robot_count": robot_count,
        "edge_count": edge_count,
        "use_llm": False,
        "seed": 73,
    }
    if task_categories is not None:
        request["task_categories"] = task_categories
    response = client.post(
        "/api/generate-scene",
        json=request,
    )
    assert response.status_code == 200
    return response.json()


def _wait_for_run(client: TestClient, run_id: str) -> dict[str, object]:
    for _ in range(300):
        response = client.get(f"/api/runtime/workflows/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"runtime workflow did not finish: {run_id}")


def test_architecture_declares_bounded_multi_node_binary_optimizer() -> None:
    payload = TestClient(api_main.app).get("/api/architecture").json()

    capabilities = payload["scheduling_capabilities"]
    assert capabilities["schema_version"] == (
        "mars.scheduling-capabilities.v1"
    )
    binary = next(
        item
        for item in capabilities["algorithms"]
        if item["id"] == "binary_offload"
    )
    assert binary["compatibility"]["supports_multiple_nodes"] is True
    assert binary["compatibility"]["requires_source_candidate"] is False
    assert binary["parameters"]["communication_weight"]["default"] == 1.0
    assert binary["search"]["strategy"] == "bounded_exhaustive"
    assert binary["default_formulation"] == "one_hot_placement"
    assert binary["supported_formulations"] == ["one_hot_placement"]
    assert payload["formulations"] == [
        "assign_or_defer",
        "one_hot_placement",
    ]
    deferred = next(
        item
        for item in capabilities["algorithms"]
        if item["id"] == "deferred_offload"
    )
    assert deferred["search"]["strategy"] == "cp_sat"
    assert deferred["default_formulation"] == "assign_or_defer"
    assert deferred["supported_formulations"] == ["assign_or_defer"]
    assert payload["planning_pipeline"].index("formulation") < (
        payload["planning_pipeline"].index("optimizer")
    )


def test_binary_simulation_supports_multiple_edges_and_shared_metrics() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/api/simulate",
        json={
            "scene": _scene(client),
            "algorithm": "binary_offload",
            "formulation": "one_hot_placement",
            "optimizer_options": {"communication_weight": 0.5},
            "network_jitter": 0,
            "resource_noise": 0,
            "seed": 73,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    metrics = payload["metrics"]
    assert metrics["scheduling_epoch_count"] > 0
    assert 0 <= metrics["expected_success_ratio"] <= 1
    assert metrics["communication_time_ms"] >= 0
    assert metrics["maximum_resource_utilization"] >= 0
    assert payload["workflow"]["formulation"] == "one_hot_placement"
    assert payload["workflow"]["scheduling"][
        "requested_formulation"
    ] == "one_hot_placement"
    assert payload["workflow"]["scheduling"][
        "effective_formulations"
    ]["one_hot_placement"] > 0


def test_binary_simulation_defaults_to_one_hot_formulation() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/api/simulate",
        json={
            "scene": _scene(client, edge_count=1),
            "algorithm": "binary_offload",
            "network_jitter": 0,
            "resource_noise": 0,
            "seed": 73,
        },
    )

    assert response.status_code == 200, response.text
    workflow = response.json()["workflow"]
    assert workflow["formulation"] == "one_hot_placement"
    assert workflow["scheduling"]["requested_formulation"] == (
        "one_hot_placement"
    )


def test_optimizer_options_are_not_supported_by_policy_aliases() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/api/simulate",
        json={
            "scene": _scene(client, edge_count=1),
            "algorithm": "dag_deadline",
            "optimizer_options": {"communication_weight": 1.0},
        },
    )

    assert response.status_code == 422
    assert "optimizer_options are not supported" in response.text


def test_one_hot_formulation_is_available_to_policy_aliases() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/api/simulate",
        json={
            "scene": _scene(client, edge_count=1),
            "algorithm": "dag_deadline",
            "formulation": "one_hot_placement",
        },
    )

    assert response.status_code == 200, response.text
    workflow = response.json()["workflow"]
    assert workflow["formulation"] == "one_hot_placement"
    assert workflow["scheduling"]["requested_formulation"] == (
        "one_hot_placement"
    )


def test_unknown_formulation_is_rejected_by_scheduling_configuration() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/api/simulate",
        json={
            "scene": _scene(client, edge_count=1),
            "algorithm": "binary_offload",
            "formulation": "unknown",
        },
    )

    assert response.status_code == 422
    assert "unsupported binary_offload formulation" in response.text


def test_deprecated_beta_is_ignored_for_non_binary_clients() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/api/simulate",
        json={
            "scene": _scene(client, edge_count=1),
            "algorithm": "dag_deadline",
            "beta": 0.01,
            "network_jitter": 0,
            "resource_noise": 0,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["workflow"]["optimizer_options"] == {}


def test_legacy_beta_cannot_silently_override_structured_options() -> None:
    client = TestClient(api_main.app)
    response = client.post(
        "/api/simulate",
        json={
            "scene": _scene(client, edge_count=1),
            "algorithm": "binary_offload",
            "beta": 0.25,
            "optimizer_options": {"communication_weight": 1.0},
        },
    )

    assert response.status_code == 422
    assert "must agree" in response.text


def test_binary_runtime_retry_can_move_to_another_candidate() -> None:
    client = TestClient(api_main.app)
    accepted = client.post(
        "/api/runtime/workflows",
        json={
            "scene": _scene(
                client,
                robot_count=1,
                task_categories=["local_llm_7b"],
            ),
            "algorithm": "binary_offload",
            "formulation": "one_hot_placement",
            "optimizer_options": {"communication_weight": 0.5},
            "seed": 73,
            "max_attempts": 2,
            "inject_first_failure": True,
            "failure_task_type": "local_llm_7b",
            "deterministic": True,
        },
    )
    assert accepted.status_code == 202, accepted.text

    payload = _wait_for_run(client, accepted.json()["run_id"])

    assert payload["status"] == "succeeded", payload.get("error")
    result = payload["result"]
    assert result["metrics"]["retry_count"] == 1
    assert result["workflow"]["requested_algorithm"] == "binary_offload"
    assert result["workflow"]["formulation"] == "one_hot_placement"
    assert result["workflow"]["scheduling"][
        "requested_formulation"
    ] == "one_hot_placement"
    assert result["workflow"]["optimizer_options"][
        "communication_weight"
    ] == 0.5
    retried = next(
        row
        for row in result["task_results"]
        if row["attempt_count"] == 2
    )
    assert [attempt["state"] for attempt in retried["attempts"]] == [
        "failed",
        "succeeded",
    ]
    assert len(
        {attempt["target_node_id"] for attempt in retried["attempts"]}
    ) == 2
