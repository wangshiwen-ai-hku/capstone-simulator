"""CPU algorithms/provenance tests with explicit CUDA fixtures, no GPU claim."""

from copy import deepcopy
import asyncio
import os
import sys

import pytest
import psutil

from agent.executor import MixedExecutor, NavigationExecutor
from agent.telemetry import detected_node
from examples.hardware_workloads import pipeline as navigation
from examples.mixed_workloads.pipeline import checked_inflation, execute, inflation_spec


def fixture_measurement():
    return {
        "backend": "cuda_runtime",
        "device": "cuda:0",
        "device_name": "FAKE CUDA TEST FIXTURE",
        "compute_capability": [8, 7],
        "cuda_runtime_version": 13020,
        "cuda_driver_version": 13020,
        "cuda_event_ms": [0.1],
        "synchronized_wall_ms": [0.2],
        "allocated_device_bytes": 4096,
        "repeats": 1,
        "warmup": 1,
        "timing_scope": "occupancy_inflation_kernel_only",
        "input_device": "cuda:0",
        "output_device": "cuda:0",
        "test_fixture": True,
        "source_sha256": "a" * 64,
        "binary_sha256": "b" * 64,
    }


def fixture_inflation(payload):
    blocked = navigation._blocked(payload)
    width, height = payload["width_cells"], payload["height_cells"]
    return {
        "schema_version": 1,
        "kind": "hil.inflated_map.v1",
        "scene_id": payload["scene_id"],
        "width_cells": width,
        "height_cells": height,
        "source_hashes": {"map": navigation._digest(payload)},
        "blocked": [
            int((x, y) in blocked) for y in range(height) for x in range(width)
        ],
        "measurement": fixture_measurement(),
    }


@pytest.fixture(scope="module")
def scene():
    source = execute("hil_mixed_sensor", {}, 19)
    payload = execute(
        "hil_mixed_mapping", {"observations": source["observations"]}, 19
    )["map"]
    return source, payload


def test_planning_consumes_supplied_mask_without_cpu_recomputation(scene, monkeypatch):
    _, payload = scene
    inflated = fixture_inflation(payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU inflation must not replace GPU planner input")

    monkeypatch.setattr(navigation, "_blocked", forbidden)
    result = execute("hil_mixed_planning", {"map": payload, "inflated": inflated}, 19)
    assert result["trajectory"]["inflation_sha256"] == navigation._digest(inflated)
    # An additional block at the start must change behavior, not be ignored.
    changed = deepcopy(inflated)
    x, y = (int(v / payload["resolution_m"]) for v in payload["start_m"])
    changed["blocked"][y * payload["width_cells"] + x] = 1
    with pytest.raises(ValueError, match="start or goal"):
        execute("hil_mixed_planning", {"map": payload, "inflated": changed}, 19)


@pytest.mark.parametrize("seed", [19, 20, 21])
def test_full_cpu_pipeline_with_explicit_gpu_fixture(seed):
    source = execute("hil_mixed_sensor", {}, seed)
    payload = execute(
        "hil_mixed_mapping", {"observations": source["observations"]}, seed
    )["map"]
    inflated = fixture_inflation(payload)
    trajectory = execute(
        "hil_mixed_planning", {"map": payload, "inflated": inflated}, seed
    )["trajectory"]
    inputs = {
        "map": payload,
        "inflated": inflated,
        "trajectory": trajectory,
        "truth": source["truth"],
    }
    validation = execute("hil_mixed_validation", inputs, seed)["validation"]
    assert validation["valid"] and validation["gpu_full_reference_match"]
    assert validation["gpu_cells_checked"] == 96 * 64
    assert validation["source_hashes"] == {
        k: navigation._digest(v) for k, v in inputs.items()
    }


def test_off_path_corruption_is_rejected_even_with_updated_hashes(scene):
    source, payload = scene
    inflated = fixture_inflation(payload)
    trajectory = execute(
        "hil_mixed_planning", {"map": payload, "inflated": inflated}, 19
    )["trajectory"]
    # Far corner is not on the path. Source/hash consistency is insufficient.
    inflated["blocked"][-1] ^= 1
    trajectory["inflation_sha256"] = navigation._digest(inflated)
    with pytest.raises(ValueError, match="full CPU reference"):
        execute(
            "hil_mixed_validation",
            {
                "map": payload,
                "inflated": inflated,
                "trajectory": trajectory,
                "truth": source["truth"],
            },
            19,
        )


def test_kernel_input_geometry_matches_cpu_unknown_obstacle_and_border_rules(scene):
    _, payload = scene
    width, height, cells, offsets, border = inflation_spec(payload)
    stencil = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if border[y * width + x]
        or any(
            0 <= x + dx < width
            and 0 <= y + dy < height
            and cells[(y + dy) * width + x + dx] != 0
            for dx, dy in offsets
        )
    }
    assert -1 in cells and 1 in cells and 0 in cells
    assert stencil == navigation._blocked(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "torch"),
        ("input_device", "cpu"),
        ("output_device", "cpu"),
        ("cuda_event_ms", [0]),
        ("cuda_event_ms", [float("nan")]),
        ("synchronized_wall_ms", []),
        ("allocated_device_bytes", 0),
        ("cuda_runtime_version", True),
        ("repeats", True),
    ],
)
def test_gpu_metadata_cannot_replace_measured_execution(scene, field, value):
    _, payload = scene
    inflated = fixture_inflation(payload)
    inflated["measurement"][field] = value
    with pytest.raises(ValueError):
        checked_inflation(payload, inflated)


def test_mixed_task_contracts_and_local_binary_requirement(tmp_path):
    pc = MixedExecutor("pc")
    assert set(pc.ports) == {"hil_mixed_mapping", "hil_mixed_planning"}
    assert pc.gpu_demands == {}
    with pytest.raises(ValueError, match="cuda-binary"):
        MixedExecutor("orin")
    with pytest.raises(ValueError, match="does not exist"):
        MixedExecutor("orin", cuda_binary=tmp_path / "missing")
    binary = tmp_path / "fixture"
    binary.write_text("not an actual CUDA binary")
    orin = MixedExecutor("orin", cuda_binary=binary)
    assert set(orin.ports) == {
        "hil_mixed_sensor",
        "hil_mixed_inflation",
        "hil_mixed_validation",
    }
    assert orin.gpu_demands == {"hil_mixed_inflation": 1.0}


def test_native_gpu_admission_requires_executed_kernel_not_torch_metadata():
    gpu = {
        "available": True,
        "backend": "cuda_runtime",
        "device": "cuda:0",
        "device_count": 1,
        "device_name": "TEST FIXTURE",
        "compute_capability": [8, 7],
        "cuda_runtime_version": 13020,
        "cuda_driver_version": 13020,
        "kernel_execution_verified": True,
    }
    node = detected_node("robot", gpu_info=gpu, capabilities=["hil_mixed_orin_v1"])
    assert node["gpu_capacity"] == 1
    assert "torch_version" not in node["gpu_info"]
    for name, value in [
        ("kernel_execution_verified", False),
        ("backend", "pretend"),
        ("cuda_driver_version", 0),
    ]:
        with pytest.raises(ValueError, match="preflight"):
            detected_node("robot", gpu_info={**gpu, name: value})


@pytest.mark.skipif(os.name != "posix", reason="hardware agents target Linux/POSIX")
def test_cancellation_stops_nested_native_helper(tmp_path, monkeypatch):
    """A Python worker's child must not survive an Agent task cancellation."""

    async def run():
        pid_path = tmp_path / "child.pid"
        code = (
            "import subprocess,sys,time,pathlib; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid)); time.sleep(30)"
        )
        original_spawn = asyncio.create_subprocess_exec
        processes = []

        async def spawn(*args, **kwargs):
            process = await original_spawn(sys.executable, "-c", code, **kwargs)
            processes.append(process)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(NavigationExecutor().execute("hil_sensor", {}, 19))
        child = None
        try:
            for _ in range(100):
                if pid_path.exists() and pid_path.read_text():
                    break
                await asyncio.sleep(0.02)
            else:
                pytest.fail("test helper did not start")
            child = psutil.Process(int(pid_path.read_text()))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert processes[0].returncode is not None
            for _ in range(100):
                if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("native helper survived cancellation of its Python worker")
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if child is not None and child.is_running():
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

    asyncio.run(run())
