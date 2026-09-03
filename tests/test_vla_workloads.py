"""Artifact contracts on CPU; real GPU execution requires explicit opt-in.

The hand-written evidence fixtures below test rejection/validation logic only.
They are never passed through a model or reported as a hardware execution.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from examples.vla_workloads import PORT_TYPES, WorkloadError, execute
from examples.vla_workloads.bundle import (
    POLICY_ID,
    POLICY_REVISION,
    VLM_ID,
    VLM_REVISION,
)
from examples.vla_workloads.pipeline import (
    MAX_PAYLOAD_BYTES,
    digest,
    validate_observation,
)


def _png(width: int = 2, height: int = 2) -> dict:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    raw = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(
            b"IDAT", zlib.compress((b"\x00" + bytes([30, 70, 120]) * width) * height)
        )
        + chunk(b"IEND", b"")
    )
    return {"encoding": "png", "data_base64": base64.b64encode(raw).decode("ascii")}


@pytest.fixture
def observation() -> dict:
    return {
        "schema": "mars.vla.observation.v1",
        "task": "Unit test artifact only: pick up the block",
        "state": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "images": {
            "observation.images.camera1": _png(),
            "observation.images.camera2": _png(3, 2),
        },
        "provenance": {
            "kind": "contract_test_fixture_not_a_hardware_run",
            "frame_index": 0,
        },
    }


def _measurement() -> dict:
    return {
        "device": "cuda:0",
        "device_name": "synthetic_contract_fixture",
        "torch_version": "test-fixture",
        "cuda_version": "test-fixture",
        "cuda_event_ms": [12.0],
        "synchronized_wall_ms": [13.0],
        "peak_memory_allocated_bytes": 16384,
        "memory_allocated_bytes": 8192,
        "warmup": 1,
        "repeats": 1,
        "input_devices": {
            key: "cuda:0"
            for key in (
                "observation.state",
                "observation.images.camera1",
                "observation.images.camera2",
                "observation.language.tokens",
                "observation.language.attention_mask",
            )
        },
        "output_device": "cuda:0",
        "parameter_devices": ["cuda:0"],
        "timing_scope": "policy_predict_action_chunk_only",
    }


@pytest.fixture
def actions(observation: dict) -> dict:
    chunk = [[float(step + axis) for axis in range(6)] for step in range(3)]
    return {
        "schema": "mars.vla.actions.v1",
        "source_hashes": {"observation": digest(observation)},
        "task": observation["task"],
        "state_dim": 6,
        "image_keys": sorted(observation["images"]),
        "missing_camera_keys": ["observation.images.camera3"],
        "action_chunk": chunk,
        "action_sha256": digest(chunk),
        "shape": [3, 6],
        "model": {
            "policy": {"repo_id": POLICY_ID, "revision": POLICY_REVISION},
            "vlm": {"repo_id": VLM_ID, "revision": VLM_REVISION},
            "manifest_sha256": "c" * 64,
            "state_dim": 6,
            "action_dim": 6,
            "chunk_size": 3,
            "image_keys": [f"observation.images.camera{index}" for index in (1, 2, 3)],
            "weights_verified": True,
            "strict_load": True,
        },
        "inference": {"seed": 19, "num_steps": 10},
        "measurement": _measurement(),
        "robot_control_executed": False,
    }


def _validate(observation: dict, actions: dict) -> dict:
    return execute(
        "hil_vla_validate", {"observation": observation, "actions": actions}, 0
    )["validation"]


def test_observer_preserves_real_input_file_without_seed_substitution(
    tmp_path: Path, observation: dict
) -> None:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(observation), encoding="utf-8")
    for seed in (0, 123, 2**32 - 1):
        result = execute("hil_vla_observe", {}, seed, {"observation_file": str(path)})
        assert result == {"observation": observation}
    assert result["observation"]["provenance"] == observation["provenance"]


def test_contract_validation_records_narrow_scope(
    observation: dict, actions: dict
) -> None:
    result = _validate(observation, actions)
    assert result["valid"] is True
    assert result["scope"] == "gpu_vla_inference_and_artifact_integrity"
    assert result["robot_control_executed"] is False
    assert result["task_success_evaluated"] is False
    assert result["source_hashes"] == {
        "observation": digest(observation),
        "actions": digest(actions),
    }
    assert result["action_shape"] == [3, 6]


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "different task"),
        ("state", [0] * 6),
        ("provenance", {"kind": "changed"}),
    ],
)
def test_validation_rejects_substituted_observation(
    observation: dict, actions: dict, field: str, value: object
) -> None:
    observation[field] = value
    with pytest.raises(WorkloadError, match="not computed from this observation"):
        _validate(observation, actions)


def test_validation_rejects_different_camera_bytes(
    observation: dict, actions: dict
) -> None:
    observation["images"]["observation.images.camera1"] = _png(4, 2)
    with pytest.raises(WorkloadError, match="not computed from this observation"):
        _validate(observation, actions)


def test_validation_rejects_changed_action_even_when_finite(
    observation: dict, actions: dict
) -> None:
    actions["action_chunk"][0][0] += 0.1
    with pytest.raises(WorkloadError, match="checksum"):
        _validate(observation, actions)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shape", [3, 5]),
        ("action_chunk", [[0] * 6]),
        ("state_dim", 7),
        ("image_keys", []),
        ("missing_camera_keys", []),
        ("robot_control_executed", True),
    ],
)
def test_validation_rejects_model_or_input_contract_tampering(
    observation: dict, actions: dict, field: str, value: object
) -> None:
    actions[field] = value
    with pytest.raises(WorkloadError):
        _validate(observation, actions)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.1"])
def test_validation_rejects_non_numeric_or_nonfinite_action(
    observation: dict, actions: dict, value: object
) -> None:
    actions["action_chunk"][0][0] = value
    with pytest.raises(WorkloadError):
        _validate(observation, actions)


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cpu"),
        ("output_device", "cpu"),
        ("input_devices", {"observation.state": "cpu"}),
        ("input_devices", {"observation.state": "cuda:0"}),
        ("parameter_devices", ["cuda:0", "cpu"]),
        ("cuda_event_ms", [0]),
        ("synchronized_wall_ms", []),
        ("peak_memory_allocated_bytes", 0),
        ("memory_allocated_bytes", 20000),
        ("cuda_version", "None"),
        ("timing_scope", "model_loading"),
    ],
)
def test_validation_rejects_incomplete_or_cpu_execution_evidence(
    observation: dict, actions: dict, field: str, value: object
) -> None:
    actions["measurement"][field] = value
    with pytest.raises(WorkloadError):
        _validate(observation, actions)


@pytest.mark.parametrize(
    "field,value",
    [
        ("strict_load", False),
        ("weights_verified", False),
        ("manifest_sha256", "missing"),
    ],
)
def test_validation_rejects_unverified_model(
    observation: dict, actions: dict, field: str, value: object
) -> None:
    actions["model"][field] = value
    with pytest.raises(WorkloadError):
        _validate(observation, actions)


def test_validation_rejects_floating_model_revision(
    observation: dict, actions: dict
) -> None:
    actions["model"]["policy"]["revision"] = "main"
    with pytest.raises(WorkloadError, match="immutable revision"):
        _validate(observation, actions)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", ""),
        ("state", []),
        ("state", [float("nan")]),
        ("state", [True]),
        ("images", {}),
        ("provenance", {}),
    ],
)
def test_observation_requires_measured_inputs_and_provenance(
    observation: dict, field: str, value: object
) -> None:
    observation[field] = value
    with pytest.raises(WorkloadError):
        validate_observation(observation)


@pytest.mark.parametrize("damage", ["base64", "checksum", "truncated", "dimensions"])
def test_observation_rejects_corrupt_or_unbounded_image(
    observation: dict, damage: str
) -> None:
    image = observation["images"]["observation.images.camera1"]
    if damage == "base64":
        image["data_base64"] = "%%%invalid%%%"
    elif damage == "dimensions":
        image.update(_png(2049, 1))
    else:
        raw = bytearray(base64.b64decode(image["data_base64"]))
        if damage == "checksum":
            raw[40] ^= 1
        else:
            raw = raw[:-1]
        image["data_base64"] = base64.b64encode(raw).decode()
    with pytest.raises(WorkloadError):
        validate_observation(observation)


def test_observation_limit_leaves_space_for_artifact_envelope(
    observation: dict, tmp_path: Path
) -> None:
    observation["padding"] = "x" * MAX_PAYLOAD_BYTES
    path = tmp_path / "too-large.json"
    path.write_text(json.dumps(observation))
    with pytest.raises(WorkloadError, match="1.5 MB"):
        execute("hil_vla_observe", {}, 0, {"observation_file": str(path)})


def _cuda_result_fixture() -> dict:
    # Explicit contract fixture, not execution of or emulation of CUDA.
    size, seed = 64, 19
    indexes = [0, 1, size // 2, size - 1]
    samples = [
        {
            "row": row,
            "column": column,
            "value": float(size * ((row + seed) % 17 - 8) * ((column + seed) % 13 - 6)),
        }
        for row in indexes
        for column in indexes
    ]
    measurement = _measurement()
    measurement["input_devices"] = {"first": "cuda:0", "second": "cuda:0"}
    measurement["timing_scope"] = "torch_matmul_only"
    return {
        "schema": "mars.cuda.result.v1",
        "operation": "float32_dense_matrix_multiply",
        "matrix_size": size,
        "seed": seed,
        "dtype": "float32",
        "max_abs_error": 0.0,
        "full_matrix_reference_match": True,
        "samples": samples,
        "sample_sha256": digest(samples),
        "measurement": measurement,
    }


def test_cuda_validator_independently_recomputes_expected_samples() -> None:
    payload = _cuda_result_fixture()
    result = execute("hil_cuda_validate", {"cuda_result": payload}, 999)["validation"]
    assert result["valid"] is True
    assert result["scope"] == "gpu_computation_and_artifact_integrity"
    assert result["source_hashes"] == {"cuda_result": digest(payload)}
    payload["samples"][0]["value"] += 1
    payload["sample_sha256"] = digest(payload["samples"])
    with pytest.raises(WorkloadError, match="independent exact reference"):
        execute("hil_cuda_validate", {"cuda_result": payload}, 999)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_abs_error", 1),
        ("full_matrix_reference_match", False),
        ("seed", 18),
        ("sample_sha256", "0" * 64),
    ],
)
def test_cuda_validator_rejects_tampering(field: str, value: object) -> None:
    payload = _cuda_result_fixture()
    payload[field] = value
    with pytest.raises(WorkloadError):
        execute("hil_cuda_validate", {"cuda_result": payload}, 19)


@pytest.mark.parametrize(
    "task,inputs,seed,options",
    [
        ("not_a_workload", {}, 0, {}),
        ("hil_vla_observe", {"unexpected": {}}, 0, {}),
        ("hil_vla_validate", {}, 0, {}),
        ("hil_vla_observe", {}, -1, {}),
        ("hil_vla_observe", {}, True, {}),
        ("hil_vla_observe", {}, 0, []),
        ("hil_cuda_validate", {"cuda_result": {"schema": "unexpected"}}, 0, {}),
        ("hil_cuda_smoke", {}, 0, {"device": "cpu"}),
        ("probe", {}, 0, {"device": "mps"}),
    ],
)
def test_rejects_invalid_requests_and_cpu_fallback(
    task: str, inputs: dict, seed: int, options: dict
) -> None:
    with pytest.raises(WorkloadError):
        execute(task, inputs, seed, options)


def _worker(request: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "examples.vla_workloads.worker"],
        input=request,
        text=True,
        capture_output=True,
        timeout=15,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_worker_outputs_one_bounded_json_value_without_gpu_imports(
    observation: dict, tmp_path: Path
) -> None:
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(observation))
    result = _worker(
        json.dumps(
            {
                "task_type": "hil_vla_observe",
                "inputs": {},
                "seed": 0,
                "options": {"observation_file": str(path)},
            }
        )
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 1
    assert json.loads(result.stdout) == {"observation": observation}


def test_package_import_does_not_load_torch_lerobot_or_agent_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import examples.vla_workloads; assert not any(k == 'torch' or k == 'lerobot' or k == 'grpc' or k.startswith('agent.') for k in sys.modules)",
        ],
        text=True,
        capture_output=True,
        timeout=15,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "request_text",
    [
        "bad json",
        "{}",
        "[]",
        '{"task_type":"probe","inputs":{},"seed":0}',
        '{"task_type":"probe","inputs":{},"seed":0,"options":{"device":"cpu"}}',
    ],
)
def test_worker_failures_never_emit_success_json(request_text: str) -> None:
    result = _worker(request_text)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "GPU/VLA workload failed:" in result.stderr


def test_declared_ports_use_distinct_versioned_vla_and_cuda_schemas() -> None:
    assert (
        PORT_TYPES["hil_vla_infer"]["inputs"]
        == PORT_TYPES["hil_vla_observe"]["outputs"]
    )
    assert (
        PORT_TYPES["hil_cuda_validate"]["inputs"]
        == PORT_TYPES["hil_cuda_smoke"]["outputs"]
    )
    assert set(PORT_TYPES["hil_vla_validate"]["inputs"]) == {"observation", "actions"}


@pytest.mark.skipif(
    os.environ.get("MARS_TEST_CUDA") != "1",
    reason="set MARS_TEST_CUDA=1 on a real CUDA host",
)
def test_opt_in_actual_cuda_matrix_execution() -> None:
    result = execute(
        "hil_cuda_smoke", {}, 19, {"matrix_size": 256, "warmup": 1, "repeats": 2}
    )
    validation = execute("hil_cuda_validate", result, 19)["validation"]
    assert validation["valid"] is True
    assert result["cuda_result"]["measurement"]["device"].startswith("cuda:")


@pytest.mark.skipif(
    not (
        os.environ.get("MARS_TEST_SMOLVLA_BUNDLE")
        and os.environ.get("MARS_TEST_SMOLVLA_OBSERVATION")
    ),
    reason="set MARS_TEST_SMOLVLA_BUNDLE and MARS_TEST_SMOLVLA_OBSERVATION on a real CUDA host",
)
def test_opt_in_actual_pretrained_smolvla_execution() -> None:
    observed = execute(
        "hil_vla_observe",
        {},
        19,
        {"observation_file": os.environ["MARS_TEST_SMOLVLA_OBSERVATION"]},
    )
    result = execute(
        "hil_vla_infer",
        observed,
        19,
        {
            "model_dir": os.environ["MARS_TEST_SMOLVLA_BUNDLE"],
            "warmup": 0,
            "repeats": 1,
        },
    )
    validation = execute("hil_vla_validate", {**observed, **result}, 19)["validation"]
    assert validation["valid"] is True
    assert result["actions"]["shape"] == [50, 6]
    assert result["actions"]["model"]["strict_load"] is True
