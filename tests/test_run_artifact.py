from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app import main as api_main
from backend.app.config import Settings
from backend.app.mars_adapter import (
    build_link_snapshots,
    build_link_specs,
    build_node_snapshots,
    build_node_specs,
    build_workflow,
)
from backend.app.runtime import LocalRuntimeService
from backend.app.scene_generator import build_deterministic_scene
from backend.app.schemas import GenerateSceneRequest, RuntimeWorkflowRequest
from backend.app.trace_archive import begin_session
from mars import RunArtifact, build_run_artifact
from mars.coordinator import CentralCoordinator
from mars.optimizers import OneHotPlacementFormulation
from mars.runtime import InProcessRuntime


_REPORT_KEYS = {
    "workflow",
    "metrics",
    "task_results",
    "agents",
    "data_edges",
    "events",
    "logs",
}


def _wait_for_terminal_run(
    service: LocalRuntimeService,
    run_id: str,
) -> dict[str, object]:
    for _ in range(200):
        payload = service.get_run(run_id)
        assert payload is not None
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"runtime workflow did not finish: {run_id}")


def _wait_for_file(path: Path) -> None:
    for _ in range(200):
        if path.is_file():
            return
        time.sleep(0.01)
    raise AssertionError(f"runtime trace was not archived: {path}")


def _completed_run():
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            use_llm=False,
            seed=43,
        )
    )
    workflow = build_workflow(scene)
    node_specs = tuple(build_node_specs(scene))
    node_snapshots = tuple(build_node_snapshots(scene))
    link_specs = tuple(build_link_specs(scene))
    link_snapshots = tuple(build_link_snapshots(scene))
    runtime = InProcessRuntime(
        node_specs,
        node_snapshots,
        execution_noise=0.0,
        sample_execution_failures=False,
    )
    coordinator = CentralCoordinator(
        runtime,
        link_specs=link_specs,
        link_snapshots=link_snapshots,
    )
    formulation = OneHotPlacementFormulation()
    report = coordinator.run(
        workflow,
        algorithm="binary_offload",
        formulation=formulation,
        seed=43,
        max_attempts=1,
        deterministic=True,
    )
    artifact = build_run_artifact(
        run_id="run_artifact_test",
        workflow=workflow,
        node_specs=node_specs,
        node_snapshots=node_snapshots,
        link_specs=link_specs,
        link_snapshots=link_snapshots,
        profiles=coordinator.profile_catalog.profiles,
        raw_report=report,
        algorithm="binary_offload",
        formulation=formulation,
        seed=43,
        deterministic=True,
        max_attempts=1,
        network_jitter=0.0,
        resource_noise=0.0,
    )
    return artifact, report


def test_run_artifact_is_frozen_data_and_keeps_report_api_stable() -> None:
    artifact, report = _completed_run()

    assert isinstance(artifact, RunArtifact)
    with pytest.raises(FrozenInstanceError):
        artifact.run_id = "changed"  # type: ignore[misc]

    assert set(report.as_dict()) == _REPORT_KEYS
    assert len(report.scheduling_plans) == report.metrics[
        "scheduling_epoch_count"
    ]

    payload = artifact.as_dict()
    assert set(payload["raw_report"]) == _REPORT_KEYS
    assert payload["raw_report"] == report.as_dict()
    assert payload["run_id"] == "run_artifact_test"
    assert payload["algorithm"] == "binary_offload"
    assert payload["formulation"] == "one_hot_placement"
    assert payload["seed"] == 43
    assert payload["deterministic"] is True
    assert payload["max_attempts"] == 1
    assert payload["network_jitter"] == 0.0
    assert payload["resource_noise"] == 0.0
    assert "evaluation" not in payload
    assert "metric_definitions" not in payload
    json.dumps(payload, allow_nan=False)


def test_run_artifact_preserves_attempts_events_outputs_and_every_plan() -> None:
    artifact, report = _completed_run()
    payload = artifact.as_dict()

    raw = payload["raw_report"]
    assert raw["events"] == report.as_dict()["events"]
    assert [item["attempts"] for item in raw["task_results"]] == [
        item["attempts"] for item in report.task_results
    ]
    assert [item["outputs"] for item in raw["task_results"]] == [
        item["outputs"] for item in report.task_results
    ]
    assert any(item["outputs"] for item in raw["task_results"])

    plans = payload["scheduling_plans"]
    assert len(plans) == len(report.scheduling_plans)
    for serialized, original in zip(
        plans,
        report.scheduling_plans,
        strict=True,
    ):
        assert serialized["epoch_id"] == original.epoch_id
        assert serialized["problem_id"] == original.problem_id
        assert serialized["solve_request_id"] == original.solve_request_id
        assert serialized["optimizer_id"] == original.optimizer_id
        assert serialized["optimizer_version"] == original.optimizer_version
        assert serialized["formulation_id"] == original.formulation_id
        assert serialized["formulation_version"] == (
            original.formulation_version
        )
        assert serialized["formulation_digest"] == (
            original.formulation_digest
        )
        assert serialized["assignments"]


def test_run_artifact_keeps_fallback_lineage_in_plan_diagnostics() -> None:
    artifact, report = _completed_run()
    first = report.scheduling_plans[0]
    fallback_plan = replace(
        first,
        diagnostics={
            **first.diagnostics,
            "repaired_from_optimizer": "candidate_solver",
            "fallback_optimizer": first.optimizer_id,
            "repaired_from_formulation_digest": "requested-digest",
            "fallback_formulation_digest": first.formulation_digest,
            "formulation_changed": True,
            "formulation_relaxed": False,
        },
    )
    report_with_lineage = replace(
        report,
        scheduling_plans=(
            fallback_plan,
            *report.scheduling_plans[1:],
        ),
    )
    artifact_with_lineage = replace(
        artifact,
        raw_report=report_with_lineage,
    )

    diagnostics = artifact_with_lineage.as_dict()["scheduling_plans"][0][
        "diagnostics"
    ]
    assert diagnostics["repaired_from_optimizer"] == "candidate_solver"
    assert diagnostics["fallback_optimizer"] == first.optimizer_id
    assert diagnostics["repaired_from_formulation_digest"] == (
        "requested-digest"
    )
    assert diagnostics["fallback_formulation_digest"] == (
        first.formulation_digest
    )
    assert diagnostics["formulation_changed"] is True
    assert diagnostics["formulation_relaxed"] is False


def test_async_runtime_archives_raw_artifact_without_changing_result_schema(
    tmp_path: Path,
) -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            use_llm=False,
            seed=5,
        )
    )
    request = RuntimeWorkflowRequest(
        scene=scene,
        algorithm="binary_offload",
        seed=5,
        max_attempts=1,
    )
    settings = Settings(
        _env_file=None,
        MARS_TRACE_ARCHIVE=True,
        MARS_TRACE_DIR=str(tmp_path / "traces"),
    )
    trace = begin_session(
        "runtime",
        settings,
        algorithm=request.algorithm,
        scene_id=scene.id,
    )
    assert trace is not None
    service = LocalRuntimeService()
    try:
        accepted = service.submit(request, trace_session=trace)
        artifact_path = trace.directory / "run_artifact.json"
        # Completion must archive independently of status endpoint polling.
        _wait_for_file(artifact_path)
        terminal = _wait_for_terminal_run(service, str(accepted["run_id"]))
    finally:
        service._executor.shutdown(wait=True)

    assert terminal["status"] == "succeeded"
    result = terminal["result"]
    assert isinstance(result, dict)
    assert set(result) == _REPORT_KEYS
    assert "workflow_evaluation_objective" in result["metrics"]

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(artifact["raw_report"]) == _REPORT_KEYS
    assert "workflow_evaluation_objective" not in (
        artifact["raw_report"]["metrics"]
    )
    assert len(artifact["scheduling_plans"]) == (
        artifact["raw_report"]["metrics"]["scheduling_epoch_count"]
    )
    archived_response = json.loads(
        (trace.directory / "response.json").read_text(encoding="utf-8")
    )
    assert set(archived_response) == _REPORT_KEYS
    assert archived_response["workflow"] == result["workflow"]
    assert archived_response["metrics"] == result["metrics"]
    assert archived_response["events"] == result["events"]


def test_sync_simulation_trace_archives_the_raw_run_artifact(
    tmp_path: Path,
) -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            use_llm=False,
            seed=29,
        )
    )
    trace_root = tmp_path / "traces"
    with (
        patch.object(api_main.settings, "mars_trace_archive", True),
        patch.object(
            api_main.settings,
            "mars_trace_dir",
            str(trace_root),
        ),
    ):
        response = TestClient(api_main.app).post(
            "/api/simulate",
            json={
                "scene": scene.model_dump(mode="json"),
                "algorithm": "dag_deadline",
                "seed": 29,
                "network_jitter": 0.0,
                "resource_noise": 0.0,
            },
        )

    assert response.status_code == 200
    artifact_paths = list(trace_root.rglob("run_artifact.json"))
    assert len(artifact_paths) == 1
    artifact = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
    assert set(artifact["raw_report"]) == _REPORT_KEYS
    assert artifact["raw_report"]["workflow"]["state"] == "succeeded"
    assert artifact["scheduling_plans"]
    assert "workflow_evaluation_objective" not in (
        artifact["raw_report"]["metrics"]
    )


def test_async_runtime_archives_artifact_when_post_run_evaluation_fails(
    tmp_path: Path,
) -> None:
    scene = build_deterministic_scene(
        GenerateSceneRequest(
            robot_count=1,
            edge_count=1,
            use_llm=False,
            seed=53,
        )
    )
    request = RuntimeWorkflowRequest(
        scene=scene,
        algorithm="dag_deadline",
        seed=53,
        max_attempts=1,
    )
    settings = Settings(
        _env_file=None,
        MARS_TRACE_ARCHIVE=True,
        MARS_TRACE_DIR=str(tmp_path / "traces"),
    )
    trace = begin_session(
        "runtime",
        settings,
        algorithm=request.algorithm,
        scene_id=scene.id,
    )
    assert trace is not None
    service = LocalRuntimeService()
    try:
        with patch(
            "backend.app.runtime.evaluate_run_artifact",
            side_effect=RuntimeError("evaluation unavailable"),
        ):
            accepted = service.submit(request, trace_session=trace)
            terminal = _wait_for_terminal_run(
                service,
                str(accepted["run_id"]),
            )
    finally:
        service._executor.shutdown(wait=True)

    assert terminal["status"] == "failed"
    assert terminal["result"] is None
    assert terminal["error"] == "RuntimeError: evaluation unavailable"
    assert not (trace.directory / "response.json").exists()
    assert (trace.directory / "error.json").is_file()
    artifact = json.loads(
        (trace.directory / "run_artifact.json").read_text(encoding="utf-8")
    )
    assert artifact["raw_report"]["workflow"]["state"] == "succeeded"
    assert artifact["scheduling_plans"]
    assert "workflow_evaluation_objective" not in (
        artifact["raw_report"]["metrics"]
    )
