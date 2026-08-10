from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.app.trace_archive import (
    archive_llm_request,
    archive_llm_result,
    begin_session,
    summarize_llm_response,
    trace_enabled,
    trace_status,
)


def test_trace_disabled_by_default() -> None:
    settings = Settings(_env_file=None, MARS_TRACE_ARCHIVE=False)
    assert not trace_enabled(settings)
    assert begin_session("generate-scene", settings) is None


def test_trace_writes_session_files(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        MARS_TRACE_ARCHIVE=True,
        MARS_TRACE_DIR=str(tmp_path / "traces"),
    )
    scene_session = begin_session("generate-scene", settings)
    assert scene_session is not None
    scene_session.write_json("scene/request.json", {"seed": 7})
    session = begin_session(
        "simulate",
        settings,
        scene_trace_id=scene_session.scene_trace_id,
        algorithm="binary_offload",
        scene_id="scene_demo",
    )
    assert session is not None
    session.write_request({"scene": "demo"})
    session.write_response({"ok": True})
    written = session.directory / "response.json"
    assert written.is_file()
    assert json.loads(written.read_text()) == {"ok": True}
    assert (scene_session.directory / "scene/meta.json").is_file()
    assert (scene_session.directory / "scene/request.json").is_file()
    assert (session.directory / "request.json").is_file()
    assert session.root_directory == scene_session.root_directory
    assert session.scene_trace_id == scene_session.scene_trace_id
    relative = session.directory.relative_to(scene_session.root_directory)
    assert relative.parts[:2] == ("calls", "simulate")
    assert "binary_offload" in relative.parts[-1]


def test_unlinked_call_creates_an_imported_scene_root(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        MARS_TRACE_ARCHIVE=True,
        MARS_TRACE_DIR=str(tmp_path / "traces"),
    )
    session = begin_session(
        "runtime",
        settings,
        algorithm="dag_deadline",
        scene_id="external_scene",
    )
    assert session is not None
    assert "imported-external_scene" in session.scene_trace_id
    meta = json.loads(
        (session.root_directory / "scene/meta.json").read_text()
    )
    assert meta["status"] == "imported"
    assert meta["scene_id"] == "external_scene"


def test_trace_rejects_path_traversal(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        MARS_TRACE_ARCHIVE=True,
        MARS_TRACE_DIR=str(tmp_path / "traces"),
    )
    session = begin_session("generate-scene", settings)
    assert session is not None
    with pytest.raises(ValueError, match="invalid trace path"):
        session.write_json("../outside.json", {"unsafe": True})


def test_llm_trace_is_structured_and_redacts_credentials(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        MARS_TRACE_ARCHIVE=True,
        MARS_TRACE_DIR=str(tmp_path / "traces"),
    )
    session = begin_session("generate-scene", settings)
    assert session is not None
    archive_llm_request(
        session,
        provider="custom",
        model="demo",
        base_url="https://user:secret@example.com/v1?token=secret",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=30,
        max_retries=1,
        stream=True,
    )
    archive_llm_result(
        session,
        provider="custom",
        model="demo",
        response_content='{"workflow_id": "wf"}',
        success=False,
        elapsed_ms=12.5,
        error=RuntimeError("Bearer sk-secretvalue123456 failed"),
    )

    request = json.loads((session.directory / "llm/request.json").read_text())
    result = json.loads((session.directory / "llm/meta.json").read_text())
    assert request["base_url"] == "https://example.com/v1"
    assert "secretvalue" not in json.dumps(result)
    assert (session.directory / "llm/response.json").is_file()


def test_summarize_llm_response_truncates_and_parses() -> None:
    content = json.dumps(
        {
            "workflow_id": "wf_demo",
            "title": "Demo",
            "tasks": [{"id": "t1"}],
            "data_edges": [],
            "nodes": [{"id": "n1"}],
        }
    )
    summary = summarize_llm_response(content, preview_chars=20)
    assert summary["preview_truncated"] is True
    assert summary["parsed_json"]["workflow_id"] == "wf_demo"
    assert summary["parsed_json"]["task_count"] == 1


def test_trace_env_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARS_TRACE_ARCHIVE", "1")
    settings = Settings(_env_file=None)
    assert trace_enabled(settings)
    status = trace_status(settings)
    assert status["enabled"] is True
    assert status["schema_version"] == "mars.trace.v3"
    assert status["layout"] == "scene/calls-by-solver"
