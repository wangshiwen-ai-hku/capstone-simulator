"""Protocol/build tests use explicitly mocked subprocesses, never simulated GPU claims.

Real CUDA tests are opt-in: set MARS_NATIVE_CUDA_TEST_BINARY to a helper built
locally by scripts.build_cuda_smoke. No CUDA execution is tested on a Mac by
the mock tests, and no executable fixture below is ever run.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from examples.mixed_workloads import native
from scripts import build_cuda_smoke as build


@pytest.fixture
def binary(tmp_path):
    path = tmp_path / "native binary; not a shell command"
    path.write_bytes(b"Inert MOCK executable bytes; this file must never execute.\n")
    path.chmod(0o700)
    manifest = {
        "schema": native.MANIFEST_SCHEMA,
        "source": native.SOURCE_RELATIVE,
        "source_sha256": native.file_sha256(native.SOURCE_PATH),
        "binary_name": path.name,
        "binary_sha256": native.file_sha256(path),
        "arch": "sm_87",
        "flags": ["-O2", "-std=c++17", "-arch=sm_87"],
        "compiler": {
            "path": "/mock/nvcc",
            "version": "MOCK CUDA compiler release 13.2",
        },
    }
    native.manifest_path(path).write_text(json.dumps(manifest))
    return path


def _inputs(**changes):
    values = {
        "width": 3,
        "height": 2,
        "cells": [0, -1, 0, 1, 0, 0],
        "offsets": [(0, 0), (1, 0)],
        "border": [False] * 5 + [True],
        "device": "cuda:0",
        "repeats": 3,
    }
    values.update(changes)
    return values


def _decode(request):
    tokens = request.split()
    assert tokens.pop(0) == "MARS_INFLATE_V1"
    numbers = list(map(int, tokens))
    width, height, offset_count, ordinal, repeats = numbers[:5]
    count = width * height
    cells = numbers[5 : 5 + count]
    flat_offsets = numbers[5 + count : 5 + count + offset_count * 2]
    border = numbers[5 + count + offset_count * 2 :]
    assert len(border) == count
    assert all(value in (0, 1) for value in border)
    return {
        "width": width,
        "height": height,
        "cells": cells,
        "offsets": list(zip(flat_offsets[::2], flat_offsets[1::2])),
        "border": list(map(bool, border)),
        "device": f"cuda:{ordinal}",
        "repeats": repeats,
    }


def _reference(width, height, cells, offsets, border):
    # Test oracle gathers neighbors. Probe's implementation scatters occupancy.
    return [
        int(
            border[y * width + x]
            or any(
                0 <= x + dx < width
                and 0 <= y + dy < height
                and cells[(y + dy) * width + x + dx] != 0
                for dx, dy in offsets
            )
        )
        for y in range(height)
        for x in range(width)
    ]


def _mock_response(request):
    values = _decode(request)
    device, repeats = values.pop("device"), values.pop("repeats")
    common = {
        "backend": "cuda_runtime",
        "device": device,
        "device_name": 'MOCK Orin "device"; no GPU executed',
        "compute_capability": [8, 7],
        "cuda_runtime_version": 13020,
        "cuda_driver_version": 13020,
    }
    return {
        "blocked": _reference(**values),
        "gpu_info": {
            **copy.deepcopy(common),
            "available": True,
            "device_count": 2,
            "kernel_execution_verified": True,
        },
        "measurement": {
            **copy.deepcopy(common),
            "cuda_event_ms": [0.01] * repeats,
            "synchronized_wall_ms": [0.05] * repeats,
            "allocated_device_bytes": len(values["cells"]) * 12
            + max(1, len(values["offsets"])) * 8,
            "repeats": repeats,
            "warmup": 1,
            "timing_scope": "occupancy_inflation_kernel_only",
            "input_device": device,
            "output_device": device,
        },
    }


@pytest.fixture
def mock_native_process(monkeypatch):
    calls = []

    def run(command, **kwargs):
        assert len(command) == 1
        assert kwargs == {
            "input": kwargs["input"],
            "text": True,
            "encoding": "utf-8",
            "errors": "strict",
            "capture_output": True,
            "timeout": 30,
            "check": False,
            "shell": False,
        }
        assert len(kwargs["input"].encode("ascii")) <= native.MAX_PROTOCOL_BYTES
        calls.append((command, _decode(kwargs["input"])))
        return subprocess.CompletedProcess(
            command, 0, json.dumps(_mock_response(kwargs["input"])), ""
        )

    monkeypatch.setattr(native.subprocess, "run", run)
    return calls


def test_mock_protocol_exact_neighbors_border_and_provenance(
    binary, mock_native_process
):
    result = native.run_inflation(**_inputs(), binary=binary)
    assert result["blocked"] == [1, 1, 0, 1, 0, 1]
    assert mock_native_process == [([str(binary)], _inputs())]
    for target in (result["gpu_info"], result["measurement"]):
        assert target["binary_sha256"] == native.file_sha256(binary)
        assert target["source_sha256"] == native.file_sha256(native.SOURCE_PATH)
    assert result["measurement"]["allocated_device_bytes"] == 88
    assert all(type(value) is float for value in result["measurement"]["cuda_event_ms"])


@pytest.mark.parametrize("repeats", [1, 20])
@pytest.mark.parametrize(
    "offsets",
    [[], [(8, -8), (-8, 8)], [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9)]],
)
def test_mock_maximum_grid_offset_and_repeat_boundaries(
    binary, mock_native_process, repeats, offsets
):
    result = native.run_inflation(
        **_inputs(
            width=128,
            height=128,
            cells=[0] * 16384,
            border=[False] * 16383 + [True],
            offsets=offsets,
            repeats=repeats,
            device="cuda:1",
        ),
        binary=binary,
    )
    assert result["blocked"] == [0] * 16383 + [1]
    assert result["measurement"]["repeats"] == repeats
    assert result["gpu_info"]["device"] == "cuda:1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 0),
        ("width", -1),
        ("width", True),
        ("width", 3.0),
        ("width", "3"),
        ("height", False),
        ("height", 16385),
        ("width", 16384),
        ("cells", []),
        ("cells", [0] * 5),
        ("cells", [[0] * 3] * 2),
        ("cells", [0, 0, 0, 0, 0, 2]),
        ("cells", [0, 0, 0, 0, 0, True]),
        ("cells", [0, 0, 0, 0, 0, 1.0]),
        ("cells", "000000"),
        ("border", [0] * 6),
        ("border", [False] * 5),
        ("border", [False] * 5 + [None]),
        ("offsets", None),
        ("offsets", [(0, 0)] * 290),
        ("offsets", [(0, 0, 0)]),
        ("offsets", [(9, 0)]),
        ("offsets", [(0, -9)]),
        ("offsets", [(True, 0)]),
        ("offsets", [(0, 1.0)]),
        ("offsets", ["00"]),
        ("repeats", 0),
        ("repeats", 21),
        ("repeats", True),
        ("repeats", 3.0),
        ("device", "cpu"),
        ("device", "cuda"),
        ("device", "cuda:-1"),
        ("device", "cuda:01"),
        ("device", "cuda:0\n"),
        ("device", "cuda:2147483648"),
        ("device", "cuda:0; echo unsafe"),
        ("device", None),
    ],
)
def test_invalid_input_never_executes(monkeypatch, field, value):
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *a, **k: pytest.fail("invalid input executed a process"),
    )
    with pytest.raises(ValueError):
        native.run_inflation(**_inputs(**{field: value}), binary="missing")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "binary",
        "source",
        "schema",
        "flags",
        "arch",
        "compiler",
        "name",
        "source_path",
        "oversized",
        "duplicates",
        "nonfinite",
        "list",
        "nonexecutable",
    ],
)
def test_manifest_rejects_stale_or_unrecorded_programs_before_execution(
    binary, monkeypatch, mutation
):
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *a, **k: pytest.fail("unverified program executed"),
    )
    path = native.manifest_path(binary)
    manifest = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    elif mutation == "binary":
        binary.write_bytes(b"changed binary")
    elif mutation == "nonexecutable":
        binary.chmod(0o600)
    elif mutation in ("oversized", "duplicates", "nonfinite", "list"):
        path.write_text(
            {
                "oversized": "x" * (native.MAX_MANIFEST_BYTES + 1),
                "duplicates": '{"schema":1,"schema":2}',
                "nonfinite": '{"schema":NaN}',
                "list": "[]",
            }[mutation]
        )
    else:
        key, value = {
            "source": ("source_sha256", "0" * 64),
            "schema": ("schema", "other"),
            "flags": ("flags", ["-O0"]),
            "arch": ("arch", "sm_87;bad"),
            "compiler": ("compiler", {}),
            "name": ("binary_name", "other"),
            "source_path": ("source", "/remote/arbitrary.cu"),
        }[mutation]
        manifest[key] = value
        path.write_text(json.dumps(manifest))
    with pytest.raises(native.NativeCudaError, match="provenance.*Rebuild locally"):
        native.run_inflation(**_inputs(), binary=binary)


def test_missing_binary_fails_actionably(tmp_path):
    with pytest.raises(native.NativeCudaError, match="scripts.build_cuda_smoke"):
        native.run_inflation(**_inputs(), binary=tmp_path / "absent")


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "blocked", [0]),
        (None, "blocked", [True] * 6),
        (None, "blocked", [2] * 6),
        (None, "gpu_info", None),
        (None, "measurement", None),
        ("gpu_info", "available", False),
        ("gpu_info", "available", 1),
        ("gpu_info", "kernel_execution_verified", False),
        ("gpu_info", "kernel_execution_verified", 1),
        ("gpu_info", "backend", "torch"),
        ("gpu_info", "device", "cuda:1"),
        ("gpu_info", "device_count", 0),
        ("gpu_info", "device_count", True),
        ("gpu_info", "device_name", " "),
        ("gpu_info", "compute_capability", [8]),
        ("gpu_info", "compute_capability", [True, 7]),
        ("gpu_info", "compute_capability", [0, 0]),
        ("gpu_info", "cuda_runtime_version", 0),
        ("gpu_info", "cuda_driver_version", "13020"),
        ("measurement", "backend", "cpu"),
        ("measurement", "device_name", "wrong"),
        ("measurement", "compute_capability", [8, 6]),
        ("measurement", "cuda_runtime_version", 13010),
        ("measurement", "cuda_driver_version", 13010),
        ("measurement", "device", "cpu"),
        ("measurement", "input_device", "cpu"),
        ("measurement", "output_device", "cpu"),
        ("measurement", "repeats", 2),
        ("measurement", "repeats", 3.0),
        ("measurement", "warmup", True),
        ("measurement", "warmup", 0),
        ("measurement", "timing_scope", "whole_process"),
        ("measurement", "allocated_device_bytes", 0),
        ("measurement", "allocated_device_bytes", 89),
        ("measurement", "allocated_device_bytes", 88.0),
        ("measurement", "cuda_event_ms", [0.01]),
        ("measurement", "cuda_event_ms", [0, 1, 1]),
        ("measurement", "cuda_event_ms", [True, 1, 1]),
        ("measurement", "cuda_event_ms", ["1", 1, 1]),
        ("measurement", "cuda_event_ms", [10**400, 1, 1]),
        ("measurement", "synchronized_wall_ms", [1, -1, 1]),
        ("measurement", "synchronized_wall_ms", None),
    ],
)
def test_rejects_mock_response_without_measurement_evidence(
    binary, monkeypatch, section, key, value
):
    def run(command, **kwargs):
        result = _mock_response(kwargs["input"])
        (result if section is None else result[section])[key] = value
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(native.NativeCudaError, match="evidence"):
        native.run_inflation(**_inputs(), binary=binary)


@pytest.mark.parametrize(
    "stdout",
    [
        "log line\n{}",
        "{}\n{}",
        "[]",
        "",
        "{",
        '{"blocked":[],"blocked":[]}',
        '{"value":NaN}',
        '{"value":Infinity}',
        "x" * (native.MAX_PROTOCOL_BYTES + 1),
    ],
)
def test_invalid_or_unbounded_stdout_rejected(binary, monkeypatch, stdout):
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout, ""),
    )
    with pytest.raises(native.NativeCudaError):
        native.run_inflation(**_inputs(), binary=binary)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_timing_is_not_measurement_evidence(binary, monkeypatch, value):
    def run(command, **kwargs):
        result = _mock_response(kwargs["input"])
        result["measurement"]["cuda_event_ms"][0] = value
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(native.NativeCudaError, match="Non-finite"):
        native.run_inflation(**_inputs(), binary=binary)


@pytest.mark.parametrize("failure", ["timeout", "os_error", "driver"])
def test_mock_process_failures_have_no_fallback(binary, monkeypatch, failure):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        if failure == "os_error":
            raise OSError("Exec format error")
        return subprocess.CompletedProcess(
            command, 1, "", "cudaGetDeviceCount: no CUDA-capable device"
        )

    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(native.NativeCudaError):
        native.run_inflation(**_inputs(), binary=binary)
    assert calls == [[str(binary)]]


def test_artifacts_changed_during_execution_are_rejected(binary, monkeypatch):
    def run(command, **kwargs):
        binary.write_bytes(b"changed during execution")
        return subprocess.CompletedProcess(
            command, 0, json.dumps(_mock_response(kwargs["input"])), ""
        )

    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(native.NativeCudaError, match="binary_sha256 mismatch"):
        native.run_inflation(**_inputs(), binary=binary)


def test_probe_mock_protocol_uses_full_asymmetric_fixture(binary, mock_native_process):
    info = native.probe_cuda(binary)
    assert info["available"] is True
    assert info["kernel_execution_verified"] is True
    assert info["measurement"]["repeats"] == 3
    assert len(mock_native_process) == 1
    _, values = mock_native_process[0]
    device, repeats = values.pop("device"), values.pop("repeats")
    assert (device, repeats) == ("cuda:0", 3)
    expected = _reference(**values)
    assert set(expected) == {0, 1}
    assert set(values["cells"]) == {-1, 0, 1}
    assert any((-dx, -dy) not in values["offsets"] for dx, dy in values["offsets"])
    without_unknown = _reference(
        **{**values, "cells": [max(0, cell) for cell in values["cells"]]}
    )
    assert expected != without_unknown
    assert expected != _reference(**{**values, "border": [False] * len(expected)})
    assert expected != _reference(
        **{
            **values,
            "offsets": [pair for pair in values["offsets"] if max(map(abs, pair)) < 8],
        }
    )


@pytest.mark.parametrize("index", range(110))
def test_probe_rejects_mismatch_at_every_output_position(binary, monkeypatch, index):
    def run(command, **kwargs):
        result = _mock_response(kwargs["input"])
        result["blocked"][index] ^= 1
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(native.NativeCudaError, match="complete independent reference"):
        native.probe_cuda(binary)


@pytest.fixture
def mock_compiler(tmp_path, monkeypatch):
    compiler = tmp_path / "CUDA Toolkit" / "nvcc"
    compiler.parent.mkdir()
    compiler.write_bytes(b"Inert MOCK nvcc bytes; never execute.\n")
    compiler.chmod(0o700)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert "NVCC_PREPEND_FLAGS" not in kwargs["env"]
        assert "NVCC_APPEND_FLAGS" not in kwargs["env"]
        if command[1:] == ["--version"]:
            return subprocess.CompletedProcess(command, 0, "MOCK nvcc release 13.2", "")
        assert command[1:3] == ["-O2", "-std=c++17"]
        assert command[3].startswith("-arch=sm_")
        assert Path(command[4]).read_bytes() == native.SOURCE_PATH.read_bytes()
        assert Path(command[4]) != native.SOURCE_PATH
        assert command[5] == "-o"
        candidate = Path(command[6])
        candidate.write_bytes(b"Inert MOCK compiled binary; never execute.\n")
        candidate.chmod(0o700)
        return subprocess.CompletedProcess(command, 0, "mock compiler diagnostics", "")

    monkeypatch.setattr(build.subprocess, "run", run)
    return compiler, calls


def test_mock_compile_only_writes_verifiable_manifest_without_gpu_claim(
    tmp_path, monkeypatch, mock_compiler
):
    compiler, calls = mock_compiler
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "--unsafe-unrecorded")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "--other-unrecorded")
    monkeypatch.setattr(
        build, "probe_cuda", lambda *a, **k: pytest.fail("compile-only called probe")
    )
    output = tmp_path / "output dir" / "helper; literal name"
    report = build.build_cuda_smoke(compiler, output, compile_only=True)
    assert report["compiled"] is True
    assert report["runtime_verified"] is False
    assert report["gpu_info"] is None and report["measurement"] is None
    assert len(calls) == 2
    assert calls[1][0][3] == "-arch=sm_87"
    artifact = native.validate_binary(output)
    assert artifact["manifest"] == report["build"]
    assert report["build"]["compiler"]["sha256"] == native.file_sha256(compiler)
    assert not list(output.parent.glob(".inflate-build-*"))


def test_mock_default_build_runs_probe_and_reports_metadata(
    tmp_path, monkeypatch, mock_compiler
):
    compiler, _ = mock_compiler
    probes = []

    def probe(binary, device):
        native.validate_binary(binary)
        probes.append((binary, device))
        info = _mock_response(native._request(**_inputs(device=device)))
        return {**info["gpu_info"], "measurement": info["measurement"]}

    monkeypatch.setattr(build, "probe_cuda", probe)
    output = tmp_path / "inflate_cuda"
    report = build.build_cuda_smoke(compiler, output, device="cuda:1", arch="sm_89")
    assert probes == [(output, "cuda:1")]
    assert report["runtime_verified"] is True
    assert report["gpu_info"]["device"] == "cuda:1"
    assert report["measurement"]["cuda_runtime_version"] == 13020
    assert report["build"]["arch"] == "sm_89"


def test_mock_compiled_binary_does_not_imply_runtime_success(
    tmp_path, monkeypatch, mock_compiler
):
    compiler, _ = mock_compiler

    def probe(*args, **kwargs):
        raise native.NativeCudaError("no CUDA-capable device")

    monkeypatch.setattr(build, "probe_cuda", probe)
    output = tmp_path / "inflate_cuda"
    with pytest.raises(
        native.NativeCudaError, match="Compiled.*runtime probe failed.*compile-only"
    ):
        build.build_cuda_smoke(compiler, output)
    assert native.validate_binary(output)


@pytest.mark.parametrize(
    "failure", ["exit", "missing_binary", "empty_binary", "timeout", "source_change"]
)
def test_mock_compile_failure_preserves_existing_artifact(
    binary, monkeypatch, mock_compiler, failure
):
    compiler, _ = mock_compiler
    previous = binary.read_bytes(), native.manifest_path(binary).read_bytes()
    original_run = build.subprocess.run

    def run(command, **kwargs):
        if command[1:] == ["--version"]:
            return original_run(command, **kwargs)
        if failure == "exit":
            return subprocess.CompletedProcess(
                command, 1, "", "nvcc: unsupported host compiler"
            )
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 180)
        if failure == "missing_binary":
            return subprocess.CompletedProcess(command, 0, "", "")
        result = original_run(command, **kwargs)
        if failure == "empty_binary":
            Path(command[-1]).write_bytes(b"")
        elif failure == "source_change":
            actual_hash = build.file_sha256
            monkeypatch.setattr(
                build,
                "file_sha256",
                lambda path: (
                    "changed" if Path(path) == native.SOURCE_PATH else actual_hash(path)
                ),
            )
        return result

    monkeypatch.setattr(build.subprocess, "run", run)
    with pytest.raises(native.NativeCudaError):
        build.build_cuda_smoke(compiler, binary, compile_only=True)
    assert (binary.read_bytes(), native.manifest_path(binary).read_bytes()) == previous


@pytest.mark.parametrize(
    "arch", ["sm_87;touch bad", "compute_87", "87", "sm_87\n", None]
)
def test_invalid_arch_never_invokes_compiler(monkeypatch, arch):
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda *a, **k: pytest.fail("invalid architecture executed"),
    )
    with pytest.raises(ValueError, match="architecture"):
        build.build_cuda_smoke(arch=arch)


def test_cli_missing_compiler_fails_with_no_success_json(tmp_path):
    # This runs only the Python CLI, using an explicitly nonexistent nvcc path.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_cuda_smoke",
            "--nvcc",
            str(tmp_path / "absent-nvcc"),
            "--output",
            str(tmp_path / "helper"),
            "--compile-only",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "CUDA compiler not found" in result.stderr
    assert "--nvcc /path/to/nvcc" in result.stderr
    assert not (tmp_path / "helper").exists()


@pytest.fixture
def cuda_target_binary():
    binary = os.environ.get("MARS_NATIVE_CUDA_TEST_BINARY")
    if not binary:
        pytest.skip(
            "real CUDA requires a locally built helper; mock tests do not verify GPU runtime"
        )
    return binary


@pytest.mark.integration
def test_real_cuda_probe_when_explicitly_enabled(cuda_target_binary):
    info = native.probe_cuda(
        cuda_target_binary,
        device=os.environ.get("MARS_NATIVE_CUDA_TEST_DEVICE", "cuda:0"),
    )
    assert info["kernel_execution_verified"] is True
    assert info["measurement"]["source_sha256"] == native.file_sha256(
        native.SOURCE_PATH
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "width,height,offsets,repeats",
    [
        (1, 1, [], 1),
        (1, 1, [(0, 0)], 20),
        (37, 23, [(0, 0), (1, -2), (-8, 8), (8, -8)], 3),
        (128, 128, [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9)], 1),
    ],
)
def test_real_cuda_full_output_when_explicitly_enabled(
    cuda_target_binary, width, height, offsets, repeats
):
    cells = [
        -1 if i % 31 == 0 else 1 if i % 43 == 0 else 0 for i in range(width * height)
    ]
    border = [i % 61 == 0 for i in range(width * height)]
    result = native.run_inflation(
        width,
        height,
        cells,
        offsets,
        border,
        binary=cuda_target_binary,
        device=os.environ.get("MARS_NATIVE_CUDA_TEST_DEVICE", "cuda:0"),
        repeats=repeats,
    )
    assert result["blocked"] == _reference(width, height, cells, offsets, border)
    assert len(result["measurement"]["cuda_event_ms"]) == repeats


def test_manifest_is_checked_against_current_source(binary, tmp_path, monkeypatch):
    source = tmp_path / "inflate.cu"
    source.write_bytes(native.SOURCE_PATH.read_bytes() + b"\n// changed source\n")
    monkeypatch.setattr(native, "SOURCE_PATH", source)
    monkeypatch.setattr(
        native.subprocess, "run", lambda *a, **k: pytest.fail("stale build executed")
    )
    with pytest.raises(native.NativeCudaError, match="source_sha256 mismatch"):
        native.run_inflation(**_inputs(), binary=binary)


def test_cli_arch_and_compile_only_emit_only_json(
    tmp_path, monkeypatch, mock_compiler, capsys
):
    compiler, calls = mock_compiler
    monkeypatch.chdir(compiler.parent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_cuda_smoke",
            "--nvcc",
            "./nvcc",
            "--output",
            str(tmp_path / "inflate_cuda"),
            "--arch",
            "sm_87",
            "--compile-only",
        ],
    )
    monkeypatch.setattr(
        build, "probe_cuda", lambda *a, **k: pytest.fail("compile-only probed CUDA")
    )
    build.main()
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report["compiled"] is True
    assert report["runtime_verified"] is False
    assert report["gpu_info"] is None
    assert report["build"]["arch"] == "sm_87"
    assert calls[1][0][1:4] == ["-O2", "-std=c++17", "-arch=sm_87"]


@pytest.mark.integration
def test_real_helper_rejects_bad_protocol_before_cuda_when_enabled(cuda_target_binary):
    # Uses the compiled parser itself; these failures must precede device setup.
    artifact = native.validate_binary(cuda_target_binary)
    requests = [
        ("BAD_PROTOCOL\n", "protocol"),
        ("MARS_INFLATE_V1\n0 1 0 0 1\n", "width"),
        ("MARS_INFLATE_V1\n128 129 0 0 1\n", "exceeds 16384"),
        ("MARS_INFLATE_V1\n1 1 290 0 1\n", "offset count"),
        ("MARS_INFLATE_V1\n1 1 0 -1 1\n", "device ordinal"),
        ("MARS_INFLATE_V1\n1 1 0 0 21\n", "repeats"),
        ("MARS_INFLATE_V1\n1 1 0 0 1\n2 0\n", "cell"),
        ("MARS_INFLATE_V1\n1 1 1 0 1\n0 9 0 0\n", "offset dx"),
        ("MARS_INFLATE_V1\n1 1 0 0 1\n0 2\n", "border flag"),
        ("MARS_INFLATE_V1\n1 1 0 0 1\n0 0 trailing\n", "trailing"),
        ("x" * (native.MAX_PROTOCOL_BYTES + 1), "128 KiB"),
    ]
    for request, diagnostic in requests:
        result = subprocess.run(
            [artifact["binary"]],
            input=request,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=15,
        )
        assert result.returncode != 0
        assert result.stdout == ""
        assert diagnostic in result.stderr
