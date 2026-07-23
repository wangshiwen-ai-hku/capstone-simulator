from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = REPO_ROOT / "interfaces/proto/mars/v1"
PROTO_FILES = (
    "common.proto",
    "workflow.proto",
    "topology.proto",
    "optimization.proto",
    "runtime.proto",
    "profiling.proto",
)


def test_proto_boundary_is_versioned_and_data_only() -> None:
    for filename in PROTO_FILES:
        source = (PROTO_ROOT / filename).read_text(encoding="utf-8")
        assert 'syntax = "proto3";' in source
        assert "package mars.v1;" in source
        assert "service " not in source


def test_proto_contract_matches_executable_v1_boundaries() -> None:
    workflow = (PROTO_ROOT / "workflow.proto").read_text(
        encoding="utf-8"
    )
    optimization = (PROTO_ROOT / "optimization.proto").read_text(
        encoding="utf-8"
    )
    runtime = (PROTO_ROOT / "runtime.proto").read_text(
        encoding="utf-8"
    )

    assert "message InputArtifactBinding" in workflow
    assert "input_artifact_bindings" in optimization
    assert "input_artifact_bindings" in runtime
    assert "OBJECTIVE_METRIC_RELIABILITY_RISK" not in optimization
    assert "OBJECTIVE_METRIC_ACCURACY_RATIO" not in optimization
    assert "repeated double objective_key" in optimization
    assert "string problem_id = 11;" in runtime


@pytest.mark.skipif(
    shutil.which("protoc") is None,
    reason="protoc is not installed",
)
def test_proto_contracts_compile_together(tmp_path: Path) -> None:
    descriptor = tmp_path / "mars-v1.pb"
    subprocess.run(
        [
            shutil.which("protoc") or "protoc",
            "-I",
            str(REPO_ROOT),
            "--include_imports",
            f"--descriptor_set_out={descriptor}",
            *(str(PROTO_ROOT / filename) for filename in PROTO_FILES),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert descriptor.stat().st_size > 0
