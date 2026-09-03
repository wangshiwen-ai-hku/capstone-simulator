"""Real CUDA computation and offline SmolVLA inference artifact contracts.

Validation checks computation records and data integrity. It does not establish
robot control, action safety, task success, or cryptographic GPU attestation.
Only the GPU stages import PyTorch or LeRobot, so acquisition and validation run
in the ordinary hardware Agent environment on a PC without those packages.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import re
import struct
import time
import zlib
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .bundle import POLICY_ID, POLICY_REVISION, VLM_ID, VLM_REVISION

PORT_TYPES = {
    "hil_vla_observe": {
        "inputs": {},
        "outputs": {"observation": "mars.vla.observation.v1"},
    },
    "hil_vla_infer": {
        "inputs": {"observation": "mars.vla.observation.v1"},
        "outputs": {"actions": "mars.vla.actions.v1"},
    },
    "hil_vla_validate": {
        "inputs": {
            "observation": "mars.vla.observation.v1",
            "actions": "mars.vla.actions.v1",
        },
        "outputs": {"validation": "mars.vla.validation.v1"},
    },
    "hil_cuda_smoke": {
        "inputs": {},
        "outputs": {"cuda_result": "mars.cuda.result.v1"},
    },
    "hil_cuda_validate": {
        "inputs": {"cuda_result": "mars.cuda.result.v1"},
        "outputs": {"validation": "mars.cuda.validation.v1"},
    },
}

# Leave room for the Agent's 2 MiB artifact envelope and transport metadata.
MAX_PAYLOAD_BYTES = 1_500_000
MAX_IMAGE_PIXELS = 1024 * 1024
MAX_IMAGE_BYTES = 750_000
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_CUDA_DEVICE = re.compile(r"cuda(?::([0-9]+))?\Z")


class WorkloadError(ValueError):
    """An invalid input, unavailable hardware, or unverified computation."""


def canonical_json(payload: Any) -> bytes:
    try:
        data = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise WorkloadError("payload must contain finite JSON values") from exc
    if len(data) > MAX_PAYLOAD_BYTES:
        raise WorkloadError("payload exceeds the 1.5 MB workload limit")
    return data


def digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _number(
    value: Any, name: str, minimum: float = -1e12, maximum: float = 1e12
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise WorkloadError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise WorkloadError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise WorkloadError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise WorkloadError(f"{name} must be a SHA256 digest")
    return value


def _png_bytes(payload: Any, name: str) -> tuple[bytes, tuple[int, int]]:
    if not isinstance(payload, dict) or set(payload) != {"encoding", "data_base64"}:
        raise WorkloadError(f"{name} must contain encoding and data_base64")
    if payload["encoding"] != "png" or not isinstance(payload["data_base64"], str):
        raise WorkloadError(f"{name} must be a base64-encoded PNG")
    if len(payload["data_base64"]) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise WorkloadError(f"{name} PNG is too large")
    try:
        raw = base64.b64decode(payload["data_base64"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WorkloadError(f"{name} has invalid base64") from exc
    if len(raw) > MAX_IMAGE_BYTES or raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise WorkloadError(f"{name} has invalid PNG data")
    # Validate framing/CRCs without importing an image library on the CPU Agent.
    # Full pixel decoding is performed by Pillow before model inference.
    offset, dimensions, saw_data, saw_end = 8, None, False, False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise WorkloadError(f"{name} has a truncated PNG chunk")
        size = struct.unpack_from(">I", raw, offset)[0]
        kind = raw[offset + 4 : offset + 8]
        end = offset + 12 + size
        if end > len(raw):
            raise WorkloadError(f"{name} has a truncated PNG chunk")
        chunk = raw[offset + 8 : offset + 8 + size]
        checksum = struct.unpack_from(">I", raw, offset + 8 + size)[0]
        if zlib.crc32(kind + chunk) & 0xFFFFFFFF != checksum:
            raise WorkloadError(f"{name} PNG checksum mismatch")
        if offset == 8:
            if kind != b"IHDR" or size != 13:
                raise WorkloadError(f"{name} PNG must begin with IHDR")
            width, height = struct.unpack_from(">II", chunk)
            if (
                not 1 <= width <= 2048
                or not 1 <= height <= 2048
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise WorkloadError(f"{name} PNG dimensions exceed the limit")
            dimensions = (width, height)
        elif kind == b"IHDR":
            raise WorkloadError(f"{name} has duplicate PNG headers")
        if kind == b"IDAT":
            saw_data = True
        if kind == b"IEND":
            if size != 0 or end != len(raw):
                raise WorkloadError(f"{name} has invalid PNG termination")
            saw_end = True
        offset = end
    if dimensions is None or not saw_data or not saw_end:
        raise WorkloadError(f"{name} PNG is incomplete")
    return raw, dimensions


def validate_observation(payload: Any) -> dict:
    """Check a supplied real observation without constructing substitute inputs."""
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "mars.vla.observation.v1"
    ):
        raise WorkloadError("observation has an unsupported schema")
    canonical_json(payload)
    task = payload.get("task")
    if not isinstance(task, str) or not task.strip() or len(task) > 4096:
        raise WorkloadError(
            "observation task must be nonempty text up to 4096 characters"
        )
    state = payload.get("state")
    if not isinstance(state, list) or not 1 <= len(state) <= 256:
        raise WorkloadError("observation state must contain 1 to 256 measured values")
    for value in state:
        _number(value, "observation state", -1e9, 1e9)
    images = payload.get("images")
    if not isinstance(images, dict) or not 1 <= len(images) <= 8:
        raise WorkloadError("observation requires 1 to 8 real camera images")
    for key, image in images.items():
        if (
            not isinstance(key, str)
            or not key.startswith("observation.images.")
            or len(key) > 160
        ):
            raise WorkloadError("image keys must be observation.images feature names")
        _png_bytes(image, key)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise WorkloadError("observation requires nonempty source provenance")
    return payload


def _observe(options: Mapping) -> dict:
    path = options.get("observation_file")
    if not isinstance(path, str) or not path:
        raise WorkloadError(
            "set observation_file to a prepared real observation JSON file"
        )
    try:
        with Path(path).expanduser().open("rb") as handle:
            raw = handle.read(MAX_PAYLOAD_BYTES + 1)
    except OSError as exc:
        raise WorkloadError(f"cannot read observation_file: {exc}") from exc
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise WorkloadError("observation_file exceeds the 1.5 MB workload limit")
    try:
        observation = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise WorkloadError("observation_file is not valid JSON") from exc
    return {"observation": validate_observation(observation)}


def _torch_cuda(device_name: Any):
    if not isinstance(device_name, str) or not _CUDA_DEVICE.fullmatch(device_name):
        raise WorkloadError(
            "GPU workloads require device cuda or cuda:N; CPU fallback is disabled"
        )
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise WorkloadError(
            "CUDA-enabled PyTorch is unavailable in the worker Python environment"
        ) from exc
    if not torch.cuda.is_available():
        raise WorkloadError(
            "torch.cuda.is_available() is false; install the PyTorch build matching JetPack/CUDA"
        )
    device = torch.device(device_name)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise WorkloadError(f"CUDA device index {index} is unavailable")
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    return torch, device


def probe_cuda(device: str = "cuda:0") -> dict:
    """Execute and synchronize a CUDA operation before advertising GPU support."""
    torch, selected = _torch_cuda(device)
    value = torch.arange(16, dtype=torch.float32, device=selected)
    result = (value * value).sum()
    torch.cuda.synchronize(selected)
    if result.device != selected or result.item() != 1240:
        raise WorkloadError("CUDA readiness computation failed")
    return {
        "available": True,
        "device": str(selected),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(selected),
        "compute_capability": list(torch.cuda.get_device_capability(selected)),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "probe_operation": "sum_of_squares_0_to_15",
        "probe_result": 1240,
    }


def _run_counts(options: Mapping) -> tuple[int, int]:
    warmup = _integer(options.get("warmup", 1), "warmup", 0, 10)
    repeats = _integer(options.get("repeats", 3), "repeats", 1, 30)
    return warmup, repeats


def _measure(torch, device, function, warmup: int, repeats: int):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    cuda_ms, wall_ms, result = [], [], None
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        start.record()
        result = function()
        end.record()
        torch.cuda.synchronize(device)
        wall_ms.append((time.perf_counter() - began) * 1000)
        cuda_ms.append(start.elapsed_time(end))
    return result, {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "cuda_event_ms": cuda_ms,
        "synchronized_wall_ms": wall_ms,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "memory_allocated_bytes": torch.cuda.memory_allocated(device),
        "warmup": warmup,
        "repeats": repeats,
    }


def _measurement(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise WorkloadError("missing CUDA measurement evidence")
    device = payload.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device):
        raise WorkloadError("measurement must name a concrete CUDA device")
    for name in ("device_name", "torch_version", "cuda_version"):
        if (
            not isinstance(payload.get(name), str)
            or not payload[name]
            or payload[name] == "None"
        ):
            raise WorkloadError(f"measurement requires {name}")
    repeats = _integer(payload.get("repeats"), "measurement repeats", 1, 30)
    _integer(payload.get("warmup"), "measurement warmup", 0, 10)
    for name in ("cuda_event_ms", "synchronized_wall_ms"):
        values = payload.get(name)
        if not isinstance(values, list) or len(values) != repeats:
            raise WorkloadError(
                f"measurement {name} must contain every measured iteration"
            )
        for value in values:
            _number(value, name, 1e-9, 3_600_000)
    _integer(payload.get("peak_memory_allocated_bytes"), "CUDA peak memory", 1, 2**60)
    _integer(payload.get("memory_allocated_bytes"), "CUDA allocated memory", 1, 2**60)
    if payload["memory_allocated_bytes"] > payload["peak_memory_allocated_bytes"]:
        raise WorkloadError("CUDA allocated memory exceeds recorded peak")
    if payload.get("output_device") != device:
        raise WorkloadError("computed output was not on the measured CUDA device")
    inputs = payload.get("input_devices")
    if (
        not isinstance(inputs, dict)
        or not inputs
        or any(value != device for value in inputs.values())
    ):
        raise WorkloadError("all tensor inputs must reside on the measured CUDA device")
    return payload


def _cuda_samples(size: int) -> list[tuple[int, int]]:
    indexes = sorted({0, 1, size // 2, size - 1})
    return [(row, column) for row in indexes for column in indexes]


def _cuda_expected(size: int, seed: int, row: int, column: int) -> float:
    return float(size * ((row + seed) % 17 - 8) * ((column + seed) % 13 - 6))


def _cuda_smoke(seed: int, options: Mapping) -> dict:
    torch, device = _torch_cuda(options.get("device", "cuda:0"))
    size = _integer(options.get("matrix_size", 1024), "matrix_size", 64, 4096)
    warmup, repeats = _run_counts(options)
    # Integer-valued FP32 matrices have an exact, independently computable answer.
    # Materialize both full matrices so each measured operation is a real GEMM.
    indexes = torch.arange(size, dtype=torch.int64, device=device)
    rows = ((indexes + seed) % 17 - 8).to(torch.float32)
    columns = ((indexes + seed) % 13 - 6).to(torch.float32)
    first = rows[:, None].expand(size, size).contiguous()
    second = columns[None, :].expand(size, size).contiguous()
    with torch.inference_mode():
        result, measurement = _measure(
            torch, device, lambda: first @ second, warmup, repeats
        )
    if result.device != device or not bool(torch.isfinite(result).all().item()):
        raise WorkloadError(
            "CUDA matrix multiplication did not produce finite CUDA output"
        )
    expected = size * rows[:, None] * columns[None, :]
    max_error = (result - expected).abs().max().item()
    if max_error != 0:
        raise WorkloadError(
            f"CUDA matrix result differs from exact reference: maximum error {max_error}"
        )
    samples = [
        {"row": row, "column": column, "value": result[row, column].item()}
        for row, column in _cuda_samples(size)
    ]
    measurement.update(
        input_devices={"first": str(first.device), "second": str(second.device)},
        output_device=str(result.device),
        timing_scope="torch_matmul_only",
    )
    payload = {
        "schema": "mars.cuda.result.v1",
        "operation": "float32_dense_matrix_multiply",
        "matrix_size": size,
        "seed": seed,
        "dtype": "float32",
        "max_abs_error": max_error,
        "full_matrix_reference_match": True,
        "samples": samples,
        "sample_sha256": digest(samples),
        "measurement": measurement,
    }
    _validate_cuda(payload)
    return {"cuda_result": payload}


def _validate_cuda(payload: dict) -> dict:
    size = _integer(payload.get("matrix_size"), "matrix_size", 64, 4096)
    seed = _integer(payload.get("seed"), "seed", 0, 2**32 - 1)
    if (
        payload.get("operation") != "float32_dense_matrix_multiply"
        or payload.get("dtype") != "float32"
    ):
        raise WorkloadError("unsupported CUDA smoke operation")
    if (
        payload.get("full_matrix_reference_match") is not True
        or _number(payload.get("max_abs_error"), "max_abs_error", 0) != 0
    ):
        raise WorkloadError("CUDA matrix reference check failed")
    expected_samples = [
        {"row": row, "column": column, "value": _cuda_expected(size, seed, row, column)}
        for row, column in _cuda_samples(size)
    ]
    samples = payload.get("samples")
    if samples != expected_samples or _hash(
        payload.get("sample_sha256"), "sample_sha256"
    ) != digest(samples):
        raise WorkloadError(
            "CUDA output samples do not match the independent exact reference"
        )
    measurement = _measurement(payload.get("measurement"))
    if set(measurement["input_devices"]) != {"first", "second"}:
        raise WorkloadError("CUDA matrix input evidence is incomplete")
    if measurement.get("timing_scope") != "torch_matmul_only":
        raise WorkloadError("CUDA matrix timing scope is missing or unsupported")
    return {
        "validation": {
            "schema": "mars.cuda.validation.v1",
            "valid": True,
            "scope": "gpu_computation_and_artifact_integrity",
            "source_hashes": {"cuda_result": digest(payload)},
            "checks": [
                "independent_exact_matrix_samples",
                "full_matrix_reference_record",
                "cuda_tensor_devices",
                "cuda_timing_and_memory",
            ],
            "gpu": measurement["device_name"],
            "robot_control_executed": False,
        }
    }


def _tensor_devices(torch, value: Any, prefix: str = "") -> dict[str, str]:
    if torch.is_tensor(value):
        return {prefix: str(value.device)}
    devices = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            devices.update(
                _tensor_devices(torch, item, f"{prefix}.{key}" if prefix else str(key))
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            devices.update(_tensor_devices(torch, item, f"{prefix}.{index}"))
    return devices


def _infer(observation: dict, seed: int, options: Mapping) -> dict:
    validate_observation(observation)
    torch, device = _torch_cuda(options.get("device", "cuda:0"))
    model_dir = options.get("model_dir")
    if not isinstance(model_dir, str) or not model_dir:
        raise WorkloadError(
            "set model_dir to a downloaded, verified SmolVLA model bundle"
        )
    from .bundle import validate_bundle

    root = Path(model_dir).expanduser().resolve()
    try:
        manifest = validate_bundle(root)
    except (OSError, ValueError, TypeError) as exc:
        raise WorkloadError(f"model bundle verification failed: {exc}") from exc
    try:
        if version("lerobot") != "0.4.4":
            raise WorkloadError("this workload requires the verified LeRobot 0.4.4 API")
        if version("transformers") != "4.57.1":
            raise WorkloadError(
                "this workload requires the verified Transformers 4.57.1 API"
            )
        import numpy as np
        from PIL import Image
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except (ImportError, OSError, PackageNotFoundError) as exc:
        raise WorkloadError(
            "SmolVLA dependencies are unavailable; use the separate LeRobot 0.4.4 worker environment"
        ) from exc

    # Environment is set by worker.py before any HF/transformers import. Explicit
    # local config and tokenizer paths also prevent opportunistic Hub downloads.
    load_began = time.perf_counter()
    cfg = SmolVLAConfig.from_pretrained(str(root / "policy"), local_files_only=True)
    cfg.vlm_model_name = str(root / "vlm")
    cfg.load_vlm_weights = False
    cfg.device = str(device)
    cfg.compile_model = False
    if getattr(cfg, "empty_cameras", 0) != 0:
        raise WorkloadError(
            "this real-input workload requires empty_cameras=0; camera tensors cannot be invented"
        )
    if "inference_steps" in options:
        cfg.num_steps = _integer(options["inference_steps"], "inference_steps", 1, 100)
    state_feature = cfg.input_features.get("observation.state")
    if state_feature is None or list(state_feature.shape) != [
        len(observation["state"])
    ]:
        raise WorkloadError(
            "observation state dimension does not match the pretrained policy"
        )
    image_keys = sorted(observation["images"])
    expected_image_keys = sorted(cfg.image_features)
    if not image_keys or not set(image_keys).issubset(expected_image_keys):
        raise WorkloadError(
            "observation camera keys do not match pretrained policy image features"
        )
    allowed_features = {"observation.state", *expected_image_keys}
    if set(cfg.input_features) - allowed_features:
        raise WorkloadError(
            "pretrained policy requires input features this observation schema cannot provide"
        )
    action_feature = cfg.output_features.get("action")
    if action_feature is None or len(action_feature.shape) != 1:
        raise WorkloadError("pretrained policy must expose a vector action feature")
    action_dim = _integer(
        int(action_feature.shape[0]), "model action dimension", 1, 256
    )
    chunk_size = _integer(int(cfg.chunk_size), "model action chunk size", 1, 1024)
    policy = (
        SmolVLAPolicy.from_pretrained(
            str(root / "policy"), config=cfg, local_files_only=True, strict=True
        )
        .to(device)
        .eval()
    )
    parameter_devices = sorted(
        {str(parameter.device) for parameter in policy.parameters()}
    )
    if parameter_devices != [str(device)]:
        raise WorkloadError(
            "all pretrained model parameters must be on the selected CUDA device"
        )
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        str(root / "policy"),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(root / "vlm")},
        },
    )
    frame = {
        "observation.state": torch.tensor(observation["state"], dtype=torch.float32),
        "task": observation["task"],
    }
    for key in image_keys:
        raw, dimensions = _png_bytes(observation["images"][key], key)
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                opened.load()
                if opened.format != "PNG" or opened.size != dimensions:
                    raise WorkloadError(
                        f"{key} does not decode to the advertised PNG dimensions"
                    )
                rgb = np.array(opened.convert("RGB"), dtype=np.uint8, copy=True)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise WorkloadError(f"{key} PNG cannot be decoded: {exc}") from exc
        # LeRobot's processor adds the batch dimension and handles normalization.
        frame[key] = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
    batch = preprocessor(frame)
    input_devices = _tensor_devices(torch, batch)
    if not input_devices or any(item != str(device) for item in input_devices.values()):
        raise WorkloadError(
            "all preprocessed input tensors must be on the selected CUDA device"
        )
    required_tensor_inputs = {
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
        *image_keys,
    }
    if not required_tensor_inputs.issubset(input_devices):
        raise WorkloadError(
            "preprocessing omitted required state, image, or language tensors"
        )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_began
    warmup, repeats = _run_counts(options)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        chunk, measurement = _measure(
            torch, device, lambda: policy.predict_action_chunk(batch), warmup, repeats
        )
        if not torch.is_tensor(chunk) or chunk.device != device:
            raise WorkloadError("SmolVLA did not return a CUDA action tensor")
        if list(chunk.shape) != [1, chunk_size, action_dim] or not bool(
            torch.isfinite(chunk).all().item()
        ):
            raise WorkloadError(
                "SmolVLA action chunk has unexpected shape or nonfinite values"
            )
        actions = postprocessor(chunk)
        if not torch.is_tensor(actions) or list(actions.shape) != [
            1,
            chunk_size,
            action_dim,
        ]:
            raise WorkloadError(
                "SmolVLA postprocessor did not preserve the action chunk shape"
            )
        if not bool(torch.isfinite(actions).all().item()):
            raise WorkloadError(
                "SmolVLA postprocessed actions contain nonfinite values"
            )
        action_chunk = actions[0].detach().to("cpu", dtype=torch.float32).tolist()
    measurement.update(
        input_devices=input_devices,
        output_device=str(chunk.device),
        parameter_devices=parameter_devices,
        parameter_dtypes=sorted(
            {str(parameter.dtype) for parameter in policy.parameters()}
        ),
        timing_scope="policy_predict_action_chunk_only",
        model_load_and_preprocess_seconds=load_seconds,
    )
    payload = {
        "schema": "mars.vla.actions.v1",
        "source_hashes": {"observation": digest(observation)},
        "task": observation["task"],
        "state_dim": len(observation["state"]),
        "image_keys": image_keys,
        "missing_camera_keys": sorted(set(expected_image_keys) - set(image_keys)),
        "action_chunk": action_chunk,
        "action_sha256": digest(action_chunk),
        "shape": [chunk_size, action_dim],
        "model": {
            "policy": manifest["policy"],
            "vlm": manifest["vlm"],
            "manifest_sha256": digest(manifest),
            "state_dim": len(observation["state"]),
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "image_keys": expected_image_keys,
            "weights_verified": True,
            "strict_load": True,
        },
        "inference": {"seed": seed, "num_steps": int(cfg.num_steps)},
        "measurement": measurement,
        "robot_control_executed": False,
    }
    _validate_vla(observation, payload)
    return {"actions": payload}


def _validate_vla(observation: dict, actions: dict) -> dict:
    validate_observation(observation)
    if actions.get("source_hashes") != {"observation": digest(observation)}:
        raise WorkloadError("actions were not computed from this observation")
    if actions.get("task") != observation["task"] or actions.get("state_dim") != len(
        observation["state"]
    ):
        raise WorkloadError("action task/state provenance does not match observation")
    if actions.get("image_keys") != sorted(observation["images"]):
        raise WorkloadError("action image provenance does not match observation")
    model = actions.get("model")
    if (
        not isinstance(model, dict)
        or model.get("weights_verified") is not True
        or model.get("strict_load") is not True
    ):
        raise WorkloadError(
            "actions require verified, strictly loaded pretrained weights"
        )
    for name in ("policy", "vlm"):
        identity = model.get(name)
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("repo_id"), str)
            or not identity["repo_id"]
        ):
            raise WorkloadError(f"model identity requires {name} repo_id")
        revision = identity.get("revision")
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise WorkloadError(f"model {name} must use an immutable revision")
    if model["policy"] != {"repo_id": POLICY_ID, "revision": POLICY_REVISION} or model[
        "vlm"
    ] != {"repo_id": VLM_ID, "revision": VLM_REVISION}:
        raise WorkloadError(
            "model identity does not match the pinned public SmolVLA bundle"
        )
    _hash(model.get("manifest_sha256"), "model manifest_sha256")
    chunk_size = _integer(model.get("chunk_size"), "model chunk_size", 1, 1024)
    action_dim = _integer(model.get("action_dim"), "model action_dim", 1, 256)
    if model.get("state_dim") != len(observation["state"]):
        raise WorkloadError("model state dimension does not match observation")
    expected_images = model.get("image_keys")
    if (
        not isinstance(expected_images, list)
        or not expected_images
        or any(not isinstance(key, str) for key in expected_images)
    ):
        raise WorkloadError("model image features are missing")
    if len(set(expected_images)) != len(expected_images) or not set(
        observation["images"]
    ).issubset(expected_images):
        raise WorkloadError("observation image keys do not match model features")
    if actions.get("missing_camera_keys") != sorted(
        set(expected_images) - set(observation["images"])
    ):
        raise WorkloadError("missing camera provenance does not match model features")
    if actions.get("shape") != [chunk_size, action_dim]:
        raise WorkloadError("action shape does not match model feature dimensions")
    chunk = actions.get("action_chunk")
    if not isinstance(chunk, list) or len(chunk) != chunk_size:
        raise WorkloadError("action chunk length does not match the model")
    for step in chunk:
        if not isinstance(step, list) or len(step) != action_dim:
            raise WorkloadError("action dimension does not match the model")
        for value in step:
            _number(value, "action value")
    if _hash(actions.get("action_sha256"), "action_sha256") != digest(chunk):
        raise WorkloadError("action values do not match their recorded checksum")
    inference = actions.get("inference")
    if not isinstance(inference, dict):
        raise WorkloadError("missing inference settings")
    _integer(inference.get("seed"), "inference seed", 0, 2**32 - 1)
    _integer(inference.get("num_steps"), "inference num_steps", 1, 100)
    measurement = _measurement(actions.get("measurement"))
    if measurement.get("parameter_devices") != [measurement["device"]]:
        raise WorkloadError(
            "pretrained parameters were not entirely on the measured CUDA device"
        )
    required_tensor_inputs = {
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
        *observation["images"],
    }
    if not required_tensor_inputs.issubset(measurement["input_devices"]):
        raise WorkloadError(
            "CUDA input evidence omits state, image, or language tensors"
        )
    if measurement.get("timing_scope") != "policy_predict_action_chunk_only":
        raise WorkloadError("VLA inference timing scope is missing or unsupported")
    if actions.get("robot_control_executed") is not False:
        raise WorkloadError("this workload cannot claim physical robot execution")
    return {
        "validation": {
            "schema": "mars.vla.validation.v1",
            "valid": True,
            "scope": "gpu_vla_inference_and_artifact_integrity",
            "source_hashes": {
                "observation": digest(observation),
                "actions": digest(actions),
            },
            "action_shape": [chunk_size, action_dim],
            "gpu": measurement["device_name"],
            "model": model,
            "checks": [
                "observation_identity",
                "pretrained_weight_verification_record",
                "finite_action_chunk",
                "model_feature_dimensions",
                "real_image_provenance",
                "cuda_parameter_and_tensor_devices",
                "cuda_timing_and_memory",
            ],
            "robot_control_executed": False,
            "task_success_evaluated": False,
        }
    }


def execute(
    task_type: str,
    inputs: Mapping[str, dict],
    seed: int,
    options: Mapping | None = None,
) -> dict[str, dict]:
    """Run one fixed task; local options are supplied by the Agent configuration."""
    if not isinstance(task_type, str) or task_type not in {*PORT_TYPES, "probe"}:
        raise WorkloadError(f"unsupported task type: {task_type!r}")
    _integer(seed, "seed", 0, 2**32 - 1)
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise WorkloadError("options must be an object")
    canonical_json(dict(options))
    required = {} if task_type == "probe" else PORT_TYPES[task_type]["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != set(required):
        raise WorkloadError(
            f"{task_type} requires exactly these inputs: {sorted(required)}"
        )
    for port, schema in required.items():
        if not isinstance(inputs[port], dict) or inputs[port].get("schema") != schema:
            raise WorkloadError(f"{port} has an unsupported schema")
        canonical_json(inputs[port])
    if task_type == "probe":
        output = {"gpu_info": probe_cuda(options.get("device", "cuda:0"))}
    elif task_type == "hil_vla_observe":
        output = _observe(options)
    elif task_type == "hil_vla_infer":
        output = _infer(inputs["observation"], seed, options)
    elif task_type == "hil_vla_validate":
        output = _validate_vla(inputs["observation"], inputs["actions"])
    elif task_type == "hil_cuda_smoke":
        output = _cuda_smoke(seed, options)
    else:
        output = _validate_cuda(inputs["cuda_result"])
    for payload in output.values():
        canonical_json(payload)
    return output
