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
SERVICE_PROTO_FILES = ("runtime_service.proto",)

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

FORMULATION_ENUM_NUMBERS = {
    "AssignmentCardinality": {
        "ASSIGNMENT_CARDINALITY_UNSPECIFIED": 0,
        "ASSIGNMENT_CARDINALITY_EXACTLY_ONE": 1,
    },
    "FormulationTaskOrder": {
        "FORMULATION_TASK_ORDER_UNSPECIFIED": 0,
        "FORMULATION_TASK_ORDER_EPOCH": 1,
    },
    "FormulationCandidateOrder": {
        "FORMULATION_CANDIDATE_ORDER_UNSPECIFIED": 0,
        "FORMULATION_CANDIDATE_ORDER_NODE_ID": 1,
    },
}

OPTIMIZATION_IDENTITY_FIELD_NUMBERS = {
    "SchedulingProblem": {
        "schema_version": 1,
        "problem_id": 2,
        "snapshot": 3,
        "policy": 4,
        "solve_limits": 5,
        "metric_contract_id": 6,
    },
    "OneHotPlacementConfig": {
        "assignment_cardinality": 1,
        "allow_drop": 2,
        "allow_defer": 3,
        "allow_split": 4,
        "allow_replication": 5,
        "task_order": 6,
        "candidate_order": 7,
    },
    "FormulationSpec": {
        "schema_version": 1,
        "formulation_id": 2,
        "formulation_version": 3,
        "materializer_id": 4,
        "materializer_version": 5,
        "formulation_digest": 6,
        "one_hot_placement": 10,
    },
    "OptimizerSpec": {
        "optimizer_id": 1,
        "optimizer_version": 2,
        "optimizer_config_digest": 3,
    },
    "SchedulingSolveRequest": {
        "schema_version": 1,
        "solve_request_id": 2,
        "problem": 3,
        "formulation": 4,
        "optimizer": 5,
        "continuation_contract_id": 6,
    },
    "Assignment": {
        "task_id": 1,
        "target_node_id": 2,
        "execution_mode": 3,
        "estimated_start_time_ms": 4,
        "estimated_finish_time_ms": 5,
        "compute_time_ms": 6,
        "communication_time_ms": 7,
        "energy_j": 8,
        "reason": 9,
        "input_node_ids": 10,
        "transfer_link_ids": 11,
        "optimizer_id": 12,
        "epoch_id": 13,
        "output_size_mb": 14,
        "success_probability_ratio": 15,
    },
    "SchedulingPlan": {
        "schema_version": 1,
        "problem_id": 2,
        "snapshot_id": 3,
        "policy_id": 4,
        "policy_version": 5,
        "epoch_id": 6,
        "optimizer_id": 7,
        "optimizer_version": 8,
        "assignments": 9,
        "node_reservations": 10,
        "transfer_reservations": 11,
        "deferred_task_ids": 12,
        "objective_value": 13,
        "objective_evaluations": 14,
        "diagnostics": 15,
        "constraint_evaluations": 16,
        "objective_key": 17,
        "solve_status": 18,
        "solve_elapsed_time_ms": 19,
        "iteration_count": 20,
        "termination_reason": 21,
        "solve_request_id": 22,
        "metric_contract_id": 23,
        "formulation_id": 24,
        "formulation_version": 25,
        "formulation_digest": 26,
    },
}

DISPATCH_FIELD_NUMBERS = {
    "schema_version": 1,
    "attempt_id": 2,
    "attempt_number": 3,
    "task": 4,
    "assignment": 5,
    "resource_reservation": 6,
    "transfer_reservations": 7,
    "input_artifact_bindings": 8,
    "random_seed": 9,
    "inject_failure": 10,
    "problem_id": 11,
    "snapshot_id": 12,
    "policy_id": 13,
    "policy_version": 14,
    "solve_request_id": 15,
}


def _proto_block(source: str, kind: str, name: str) -> str:
    match = re.search(rf"^\s*{kind}\s+{name}\s*\{{", source, re.MULTILINE)
    assert match is not None, f"missing {kind} {name}"
    start = match.end()
    depth = 1
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]
    raise AssertionError(f"unterminated {kind} {name}")


def _message_field_numbers(source: str, name: str) -> dict[str, int]:
    body = _proto_block(source, "message", name)
    return {
        field_name: int(number)
        for field_name, number in re.findall(
            r"^\s*(?:(?:optional|repeated)\s+)?"
            r"(?:map<[^>]+>|[A-Za-z_][A-Za-z0-9_.]*)\s+"
            r"([a-z][a-z0-9_]*)\s*=\s*(\d+);",
            body,
            flags=re.MULTILINE,
        )
    }


def _enum_numbers(source: str, name: str) -> dict[str, int]:
    body = _proto_block(source, "enum", name)
    return {
        value_name: int(number)
        for value_name, number in re.findall(
            r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(\d+);",
            body,
            flags=re.MULTILINE,
        )
    }


def test_proto_boundary_is_versioned_and_data_only() -> None:
    for filename in PROTO_FILES:
        source = (PROTO_ROOT / filename).read_text(encoding="utf-8")
        assert 'syntax = "proto3";' in source
        assert "package mars.v1;" in source
        assert "service " not in source


def test_runtime_service_reuses_the_data_only_runtime_contract() -> None:
    source = (PROTO_ROOT / "runtime_service.proto").read_text(encoding="utf-8")
    assert 'syntax = "proto3";' in source
    assert "package mars.v1;" in source
    assert 'import "interfaces/proto/mars/v1/runtime.proto";' in source
    assert "service AgentRuntime" in source
    for method in (
        "RegisterAgent",
        "GetState",
        "DispatchTask",
        "StreamCompletions",
        "CancelAttempt",
    ):
        assert f"rpc {method}" in source


def test_proto_contract_matches_executable_v1_boundaries() -> None:
    workflow = (PROTO_ROOT / "workflow.proto").read_text(encoding="utf-8")
    optimization = (PROTO_ROOT / "optimization.proto").read_text(encoding="utf-8")
    runtime = (PROTO_ROOT / "runtime.proto").read_text(encoding="utf-8")

    assert "message InputArtifactBinding" in workflow
    assert "input_artifact_bindings" in optimization
    assert "input_artifact_bindings" in runtime
    assert "OBJECTIVE_METRIC_RELIABILITY_RISK" not in optimization
    assert "OBJECTIVE_METRIC_ACCURACY_RATIO" not in optimization
    assert "repeated double objective_key" in optimization
    assert "message SchedulingSolveRequest" in optimization
    assert "message FormulationSpec" in optimization
    assert "string metric_contract_id = 6;" in optimization
    assert "string problem_id = 11;" in runtime
    assert "string solve_request_id = 15;" in runtime


def test_proto_objective_metric_catalog_matches_python_enum() -> None:
    source = (PROTO_ROOT / "optimization.proto").read_text(encoding="utf-8")
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


def test_proto_formulation_enums_and_field_numbers_are_stable() -> None:
    optimization = (PROTO_ROOT / "optimization.proto").read_text(encoding="utf-8")
    runtime = (PROTO_ROOT / "runtime.proto").read_text(encoding="utf-8")

    for enum_name, expected in FORMULATION_ENUM_NUMBERS.items():
        assert _enum_numbers(optimization, enum_name) == expected
    for message_name, expected in OPTIMIZATION_IDENTITY_FIELD_NUMBERS.items():
        assert _message_field_numbers(optimization, message_name) == expected
    assert _message_field_numbers(runtime, "DispatchCommand") == (
        DISPATCH_FIELD_NUMBERS
    )


def test_proto_formulation_is_typed_data_not_an_executable_model() -> None:
    optimization = (PROTO_ROOT / "optimization.proto").read_text(encoding="utf-8")
    formulation = _proto_block(optimization, "message", "FormulationSpec")
    request = _proto_block(
        optimization,
        "message",
        "SchedulingSolveRequest",
    )

    assert "oneof formulation" in formulation
    assert "OneHotPlacementConfig one_hot_placement = 10;" in formulation
    assert "map<" not in formulation
    assert "SchedulingProblem problem = 3;" in request
    assert "FormulationSpec formulation = 4;" in request
    assert "OptimizerSpec optimizer = 5;" in request


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
            *(
                str(PROTO_ROOT / filename)
                for filename in (*PROTO_FILES, *SERVICE_PROTO_FILES)
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert descriptor.stat().st_size > 0

    descriptor_pb2 = pytest.importorskip("google.protobuf.descriptor_pb2")
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.ParseFromString(descriptor.read_bytes())
    files = {item.name: item for item in descriptor_set.file}
    optimization_descriptor = files["interfaces/proto/mars/v1/optimization.proto"]
    runtime_descriptor = files["interfaces/proto/mars/v1/runtime.proto"]
    messages = {item.name: item for item in optimization_descriptor.message_type}
    runtime_messages = {item.name: item for item in runtime_descriptor.message_type}

    assert {
        item.name: item.number for item in messages["SchedulingSolveRequest"].field
    } == OPTIMIZATION_IDENTITY_FIELD_NUMBERS["SchedulingSolveRequest"]
    formulation = messages["FormulationSpec"]
    assert [item.name for item in formulation.oneof_decl] == ["formulation"]
    one_hot = next(
        item for item in formulation.field if item.name == "one_hot_placement"
    )
    assert one_hot.oneof_index == 0
    assert one_hot.type_name == ".mars.v1.OneHotPlacementConfig"
    assert {
        item.name: item.number for item in runtime_messages["DispatchCommand"].field
    } == DISPATCH_FIELD_NUMBERS
