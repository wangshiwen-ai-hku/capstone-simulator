from pathlib import Path
import re
import shutil
import subprocess

import pytest

from mars.optimizers import ObjectiveMetric


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

OBJECTIVE_METRIC_PROTO_LEGACY_NAMES = {
    "TOTAL_COMMUNICATION_MS": "TOTAL_COMMUNICATION_TIME_MS",
    "DROPPED_TASKS": "DROPPED_TASK_COUNT",
    "NON_SOURCE_ASSIGNMENTS": "NON_SOURCE_ASSIGNMENT_COUNT",
    "NON_EDGE_ASSIGNMENTS": "NON_EDGE_ASSIGNMENT_COUNT",
}

OBJECTIVE_METRIC_PROTO_NUMBERS = {
    "UNSPECIFIED": 0,
    "MAKESPAN_MS": 1,
    "TOTAL_DEADLINE_VIOLATION_MS": 2,
    "TOTAL_ENERGY_J": 3,
    "TOTAL_COMMUNICATION_TIME_MS": 4,
    "LOCALITY_PENALTY": 5,
    "DROPPED_TASK_COUNT": 6,
    "TOTAL_COMPLETION_TIME_MS": 7,
    "CRITICAL_PATH_FINISH_MS": 8,
    "NON_SOURCE_ASSIGNMENT_COUNT": 9,
    "NON_EDGE_ASSIGNMENT_COUNT": 10,
    "PLACEMENT_PREFERENCE_PENALTY": 11,
    "RULE_MISMATCH_COUNT": 12,
    "EXPECTED_WEIGHTED_SUCCESS_RATIO": 13,
    "NORMALIZED_COMMUNICATION_RATIO": 14,
    "MAXIMUM_RESOURCE_UTILIZATION": 15,
}


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


def test_proto_objective_metric_catalog_matches_python_enum() -> None:
    source = (PROTO_ROOT / "optimization.proto").read_text(
        encoding="utf-8"
    )
    enum_match = re.search(
        r"enum ObjectiveMetric \{(?P<body>.*?)^\}",
        source,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert enum_match is not None

    proto_catalog = dict(
        (name, int(number))
        for name, number in re.findall(
            r"^\s*OBJECTIVE_METRIC_([A-Z0-9_]+)\s*=\s*(\d+);",
            enum_match.group("body"),
            flags=re.MULTILINE,
        )
    )
    assert proto_catalog == OBJECTIVE_METRIC_PROTO_NUMBERS
    proto_names = set(proto_catalog)
    proto_names.remove("UNSPECIFIED")
    python_names = {
        OBJECTIVE_METRIC_PROTO_LEGACY_NAMES.get(metric.name, metric.name)
        for metric in ObjectiveMetric
    }

    assert proto_names == python_names


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
