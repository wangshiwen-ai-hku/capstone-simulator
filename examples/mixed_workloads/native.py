"""Bounded stdin/JSON adapter for the locally built CUDA Runtime helper.

No torch, CUDA Python bindings, downloads, shell invocation, or CPU fallback.
Only a binary accompanied by a matching local build manifest can execute. The
manifest detects stale/substituted artifacts; it is not a signature against a
hostile local user who can rewrite both the executable and its manifest.

Wire protocol (ASCII whitespace separated): MARS_INFLATE_V1; width, height,
offset count, device ordinal, repeats; flattened cells; dx/dy pairs; border 0/1.
Offsets lie in [-8, 8] on each axis; an empty list means border-only blocking.
Successful stdout is one JSON object, bounded to 128 KiB; diagnostics use stderr.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

SOURCE_PATH = Path(__file__).resolve().with_name("inflate.cu")
SOURCE_RELATIVE = "examples/mixed_workloads/inflate.cu"
MANIFEST_SCHEMA = "mars.native-cuda-inflation.v1"
DEFAULT_BINARY = Path(".mars-hil/bin/inflate_cuda")
DEFAULT_ARCH = "sm_87"
MAX_CELLS = 16384
MAX_OFFSETS = 17 * 17
MAX_PROTOCOL_BYTES = 128 * 1024
MAX_MANIFEST_BYTES = 16 * 1024
PROCESS_TIMEOUT_SECONDS = 30
BUILD_HINT = (
    "Rebuild locally with python -m scripts.build_cuda_smoke --output <binary>."
)


class NativeCudaError(RuntimeError):
    """Native compilation, provenance, execution, or measurement verification failed."""


def file_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(binary: str | os.PathLike) -> Path:
    return Path(str(binary) + ".manifest.json")


def validate_arch(arch: str) -> str:
    if not isinstance(arch, str) or not re.fullmatch(r"sm_[0-9]{2,3}[af]?", arch):
        raise ValueError("arch must be an explicit CUDA architecture such as sm_87")
    return arch


def _device_ordinal(device: str) -> int:
    if not isinstance(device, str) or not re.fullmatch(
        r"cuda:(0|[1-9][0-9]{0,9})", device
    ):
        raise ValueError(
            "device must be a canonical CUDA device such as cuda:0; no CPU fallback"
        )
    ordinal = int(device[5:])
    if ordinal > 2**31 - 1:
        raise ValueError("CUDA device ordinal exceeds int32")
    return ordinal


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _load_json(text: str) -> dict:
    result = json.loads(
        text, object_pairs_hook=_json_object, parse_constant=_invalid_constant
    )
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def validate_binary(binary: str | os.PathLike) -> dict:
    """Verify source and executable hashes without executing manifest-specified code."""
    try:
        path = Path(binary).expanduser().resolve(strict=True)
        if (
            not path.is_file()
            or not os.access(path, os.X_OK)
            or path.stat().st_size == 0
        ):
            raise ValueError("binary must be a nonempty executable file")
        with manifest_path(path).open("rb") as stream:
            encoded = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds 16 KiB")
        manifest = _load_json(encoded.decode("utf-8"))
        arch = validate_arch(manifest.get("arch"))
        if (
            manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("source") != SOURCE_RELATIVE
            or manifest.get("binary_name") != path.name
            or manifest.get("flags") != ["-O2", "-std=c++17", f"-arch={arch}"]
        ):
            raise ValueError("unrecognized build manifest or compile flags")
        compiler = manifest.get("compiler")
        if not isinstance(compiler, dict) or any(
            not isinstance(compiler.get(key), str) or not compiler[key].strip()
            for key in ("path", "version")
        ):
            raise ValueError("manifest lacks compiler metadata")
        for key, actual in (
            ("source_sha256", file_sha256(SOURCE_PATH)),
            ("binary_sha256", file_sha256(path)),
        ):
            if manifest.get(key) != actual:
                raise ValueError(f"{key} mismatch (stale source or changed binary)")
        return {"binary": str(path), "manifest": manifest}
    except (OSError, TypeError, ValueError, RecursionError) as error:
        raise NativeCudaError(
            f"CUDA binary provenance check failed: {error}. {BUILD_HINT}"
        ) from error


def _sequence(value: object, length: int, field: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must be a flat list/tuple of length {length}")


def _request(width, height, cells, offsets, border, device, repeats) -> str:
    if any(
        type(value) is not int or not 1 <= value <= MAX_CELLS
        for value in (width, height)
    ):
        raise ValueError("width and height must be positive integers <= 16384")
    count = width * height
    if count > MAX_CELLS:
        raise ValueError("width * height must be <= 16384")
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("repeats must be an integer in 1..20")
    ordinal = _device_ordinal(device)
    _sequence(cells, count, "cells")
    _sequence(border, count, "border")
    if any(type(value) is not int or value not in (-1, 0, 1) for value in cells):
        raise ValueError("cells must contain only integers -1, 0, or 1")
    if any(type(value) is not bool for value in border):
        raise ValueError("border must contain only boolean flags")
    if not isinstance(offsets, (list, tuple)) or len(offsets) > MAX_OFFSETS:
        raise ValueError("offsets must be a list/tuple with at most 17x17 pairs")
    for pair in offsets:
        _sequence(pair, 2, "offset pair")
        if any(type(value) is not int or not -8 <= value <= 8 for value in pair):
            raise ValueError(
                "offset coordinates must be integers in [-8, 8] (17x17 neighborhood)"
            )
    request = "\n".join(
        (
            "MARS_INFLATE_V1",
            f"{width} {height} {len(offsets)} {ordinal} {repeats}",
            " ".join(map(str, cells)),
            " ".join(str(value) for pair in offsets for value in pair),
            " ".join("1" if value else "0" for value in border),
            "",
        )
    )
    if len(request.encode("ascii")) > MAX_PROTOCOL_BYTES:
        raise ValueError("Native CUDA request exceeds 128 KiB")
    return request


def _validate_result(
    result: dict, count: int, offset_count: int, device: str, repeats: int
) -> None:
    def require(condition: bool, field: str) -> None:
        if not condition:
            raise NativeCudaError(
                f"Invalid native CUDA output/measurement evidence: {field}"
            )

    blocked = result.get("blocked")
    require(isinstance(blocked, list) and len(blocked) == count, "blocked length")
    require(
        all(type(value) is int and value in (0, 1) for value in blocked),
        "blocked domain",
    )
    info, measurement = result.get("gpu_info"), result.get("measurement")
    require(
        isinstance(info, dict) and isinstance(measurement, dict),
        "gpu_info/measurement objects",
    )
    require(info.get("available") is True, "CUDA availability")
    require(info.get("kernel_execution_verified") is True, "kernel_execution_verified")
    require(info.get("backend") == "cuda_runtime", "backend")
    require(info.get("device") == device, "requested device")
    device_count = info.get("device_count")
    require(
        type(device_count) is int
        and _device_ordinal(device) < device_count <= 2**31 - 1,
        "device_count",
    )
    name = info.get("device_name")
    require(
        isinstance(name, str) and bool(name.strip()) and len(name) <= 256, "device_name"
    )
    capability = info.get("compute_capability")
    require(isinstance(capability, list) and len(capability) == 2, "compute_capability")
    require(
        all(type(value) is int for value in capability)
        and 1 <= capability[0] <= 999
        and 0 <= capability[1] <= 99,
        "compute_capability values",
    )
    for key in ("cuda_runtime_version", "cuda_driver_version"):
        require(type(info.get(key)) is int and 1000 <= info[key] <= 1000000, key)
    for key in (
        "backend",
        "device",
        "device_name",
        "compute_capability",
        "cuda_runtime_version",
        "cuda_driver_version",
    ):
        require(
            type(measurement.get(key)) is type(info[key])
            and measurement[key] == info[key],
            f"measurement.{key}",
        )
    require(
        all(type(value) is int for value in measurement["compute_capability"]),
        "measurement.compute_capability values",
    )
    require(
        type(measurement.get("repeats")) is int and measurement["repeats"] == repeats,
        "repeats",
    )
    require(
        type(measurement.get("warmup")) is int and measurement["warmup"] == 1, "warmup"
    )
    require(
        measurement.get("timing_scope") == "occupancy_inflation_kernel_only",
        "timing_scope",
    )
    for key in ("input_device", "output_device"):
        require(measurement.get(key) == device, key)
    allocated = measurement.get("allocated_device_bytes")
    require(
        type(allocated) is int and allocated == count * 12 + max(1, offset_count) * 8,
        "allocated_device_bytes",
    )
    for key in ("cuda_event_ms", "synchronized_wall_ms"):
        values = measurement.get(key)
        require(isinstance(values, list) and len(values) == repeats, key)
        for value in values:
            require(type(value) in (int, float), key)
            try:
                valid = math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            require(valid, f"positive finite {key}")
        measurement[key] = [float(value) for value in values]


def run_inflation(
    width, height, cells, offsets, border, *, binary, device="cuda:0", repeats=3
) -> dict:
    """Run one warmup and 1..20 measured CUDA executions, returning device output.

    Event times bracket the kernel; synchronized wall times include launch/event
    and synchronization overhead. Allocation, transfers and correctness checks
    are excluded. Both metadata objects include verified artifact SHA-256 hashes.
    """
    request = _request(width, height, cells, offsets, border, device, repeats)
    artifact = validate_binary(binary)
    try:
        completed = subprocess.run(
            [artifact["binary"]],
            input=request,
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise NativeCudaError(
            f"Native CUDA helper could not execute: {error}. Check the local NVIDIA CUDA installation; no CPU fallback."
        ) from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip()[:4096] or "no diagnostic from helper"
        raise NativeCudaError(
            f"Native CUDA helper failed (exit {completed.returncode}): {diagnostic}"
        )
    if len(completed.stdout.encode("utf-8")) > MAX_PROTOCOL_BYTES:
        raise NativeCudaError("Native CUDA response exceeds 128 KiB")
    try:
        result = _load_json(completed.stdout)
    except (ValueError, RecursionError) as error:
        raise NativeCudaError(
            f"Native CUDA helper did not return one valid JSON object: {error}"
        ) from error
    _validate_result(result, width * height, len(offsets), device, repeats)
    if validate_binary(binary) != artifact:
        raise NativeCudaError(
            "CUDA build artifacts changed during execution; rebuild and retry"
        )
    for target in (result["measurement"], result["gpu_info"]):
        for key in ("binary_sha256", "source_sha256"):
            target[key] = artifact["manifest"][key]
    return result


def probe_cuda(binary, device="cuda:0") -> dict:
    """Return verified GPU identity with a nested measurement from a real fixture.

    Every fixture output is compared to an independently calculated reference.
    A CPU reference is used only to reject incorrect GPU results, never as output.
    Device/runtime absence, mismatches and invalid timing evidence raise errors.
    """
    width, height = 11, 10
    cells = [0] * (width * height)
    for index, value in ((0, 1), (10, -1), (26, 1), (72, -1), (99, -1), (109, 1)):
        cells[index] = value
    offsets = [(0, 0), (1, 0), (-2, 1), (0, -1), (8, -8), (-8, 8)]
    border = [False] * len(cells)
    border[16] = border[82] = True
    expected = [int(flag) for flag in border]
    # Scatter source occupancy in the reverse offset direction, independent of
    # the GPU's per-target gather loop. Include asymmetric offsets and edges.
    for source, value in enumerate(cells):
        if value == 0:
            continue
        sy, sx = divmod(source, width)
        for dx, dy in offsets:
            tx, ty = sx - dx, sy - dy
            if 0 <= tx < width and 0 <= ty < height:
                expected[ty * width + tx] = 1
    result = run_inflation(
        width, height, cells, offsets, border, binary=binary, device=device, repeats=3
    )
    if result["blocked"] != expected:
        raise NativeCudaError(
            "CUDA probe output mismatches the complete independent reference"
        )
    return {**result["gpu_info"], "measurement": result["measurement"]}
