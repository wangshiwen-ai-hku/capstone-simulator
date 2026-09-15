"""Measured CPU/CUDA navigation smoke; no model download or physical actuation.

CPU planning consumes the received GPU mask. Only the subsequent independent
validator recomputes inflation on CPU, comparing every cell before checking
the trajectory against continuous synthetic scene geometry.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

from examples.hardware_workloads import pipeline as navigation


GPU_TASK_TYPE = "hil_mixed_inflation"
PORT_TYPES = {
    "hil_mixed_sensor": navigation.PORT_TYPES["hil_sensor"],
    "hil_mixed_mapping": navigation.PORT_TYPES["hil_mapping"],
    GPU_TASK_TYPE: {
        "inputs": {"map": "hil.occupancy_map.v1"},
        "outputs": {"inflated": "hil.inflated_map.v1"},
    },
    "hil_mixed_planning": {
        "inputs": {
            "map": "hil.occupancy_map.v1",
            "inflated": "hil.inflated_map.v1",
        },
        "outputs": {"trajectory": "hil.trajectory.v1"},
    },
    "hil_mixed_validation": {
        "inputs": {
            "map": "hil.occupancy_map.v1",
            "inflated": "hil.inflated_map.v1",
            "trajectory": "hil.trajectory.v1",
            "truth": "hil.truth.v1",
        },
        "outputs": {"validation": "hil.mixed_validation.v1"},
    },
}


def inflation_spec(payload: dict) -> tuple[int, int, list, list, list]:
    """Prepare bounded kernel inputs; do not calculate the obstacle mask here."""
    width, height, resolution, cells = navigation._validated_map(payload)
    radius = navigation._limits(payload["limits"])["robot_radius_m"]
    noise = navigation._number(
        payload.get("range_noise_bound_m"),
        "range_noise_bound_m",
        minimum=0,
        maximum=0.05,
    )
    inflation = radius + noise + resolution * math.sqrt(2) / 2
    reach = math.ceil(inflation / resolution)
    if reach > 8:
        raise ValueError("robot footprint exceeds the eight-cell inflation bound")
    offsets = [
        [dx, dy]
        for dx in range(-reach, reach + 1)
        for dy in range(-reach, reach + 1)
        if math.hypot(dx, dy) * resolution <= inflation
    ]
    border = [
        min(
            (x + 0.5) * resolution,
            (y + 0.5) * resolution,
            (width - x - 0.5) * resolution,
            (height - y - 0.5) * resolution,
        )
        < radius
        for y in range(height)
        for x in range(width)
    ]
    return width, height, [value for row in cells for value in row], offsets, border


def verify_measurement(measurement: dict) -> dict:
    """Check runtime evidence separately from node capability advertisements."""
    if not isinstance(measurement, dict):
        raise ValueError("missing native CUDA measurement")
    device = measurement.get("device")
    if (
        measurement.get("backend") != "cuda_runtime"
        or not isinstance(device, str)
        or not device.startswith("cuda:")
        or not device[5:].isdigit()
        or measurement.get("input_device") != device
        or measurement.get("output_device") != device
        or measurement.get("timing_scope") != "occupancy_inflation_kernel_only"
    ):
        raise ValueError("inflation did not report the native CUDA execution contract")
    repeats = measurement.get("repeats")
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("invalid CUDA repeat count")
    for field in ("cuda_event_ms", "synchronized_wall_ms"):
        values = measurement.get(field)
        if (
            not isinstance(values, list)
            or len(values) != repeats
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in values
            )
        ):
            raise ValueError(f"missing positive measured {field}")
    for field in (
        "cuda_runtime_version",
        "cuda_driver_version",
        "allocated_device_bytes",
    ):
        if type(measurement.get(field)) is not int or measurement[field] <= 0:
            raise ValueError(f"missing measured {field}")
    if (
        not isinstance(measurement.get("device_name"), str)
        or not measurement["device_name"].strip()
    ):
        raise ValueError("missing CUDA device name")
    return measurement


def checked_inflation(payload: dict, inflated: dict, *, reference: bool = False) -> set:
    width, height, _, _ = navigation._validated_map(payload)
    if (
        inflated.get("schema_version") != 1
        or inflated.get("kind") != "hil.inflated_map.v1"
        or inflated.get("source_hashes") != {"map": navigation._digest(payload)}
        or inflated.get("width_cells") != width
        or inflated.get("height_cells") != height
        or inflated.get("scene_id") != payload["scene_id"]
    ):
        raise ValueError("inflation does not identify the transported map")
    values = inflated.get("blocked")
    if (
        not isinstance(values, list)
        or len(values) != width * height
        or any(type(value) is not int or value not in (0, 1) for value in values)
    ):
        raise ValueError("GPU mask has invalid size or values")
    verify_measurement(inflated.get("measurement"))
    blocked = {(i % width, i // width) for i, value in enumerate(values) if value}
    if reference and blocked != navigation._blocked(payload):
        raise ValueError(
            "GPU inflation differs from the independent full CPU reference"
        )
    return blocked


def execute(
    task_type: str, inputs: Mapping[str, dict], seed: int, *, options=None
) -> dict:
    if task_type not in PORT_TYPES:
        raise ValueError(f"unsupported mixed task: {task_type}")
    navigation._integer(seed, "seed", 0, 2**32 - 1)
    if not isinstance(inputs, Mapping) or set(inputs) != set(
        PORT_TYPES[task_type]["inputs"]
    ):
        raise ValueError("mixed task input ports do not match its contract")
    for name, kind in PORT_TYPES[task_type]["inputs"].items():
        value = inputs[name]
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("kind") != kind
        ):
            raise ValueError(f"invalid {name} input schema")
        navigation._canonical(value)
    if task_type in {"hil_mixed_sensor", "hil_mixed_mapping"}:
        return navigation.execute(task_type.replace("hil_mixed_", "hil_"), inputs, seed)
    payload = inputs["map"]
    if task_type == GPU_TASK_TYPE:
        from .native import run_inflation

        options = options or {}
        binary = options.get("cuda_binary")
        if not binary:
            raise ValueError("native CUDA binary is required; CPU fallback is disabled")
        result = run_inflation(
            *inflation_spec(payload),
            binary=binary,
            device=options.get("device", "cuda:0"),
            repeats=options.get("repeats", 3),
        )
        inflated = {
            "schema_version": 1,
            "kind": "hil.inflated_map.v1",
            "scene_id": payload["scene_id"],
            "width_cells": payload["width_cells"],
            "height_cells": payload["height_cells"],
            "source_hashes": {"map": navigation._digest(payload)},
            "blocked": result["blocked"],
            "measurement": result["measurement"],
        }
        checked_inflation(payload, inflated)
        return {"inflated": inflated}
    inflated = inputs["inflated"]
    blocked = checked_inflation(
        payload, inflated, reference=task_type == "hil_mixed_validation"
    )
    if task_type == "hil_mixed_planning":
        outputs = navigation._planning(payload, blocked_cells=blocked)
        outputs["trajectory"]["inflation_sha256"] = navigation._digest(inflated)
        return outputs
    trajectory = inputs["trajectory"]
    if trajectory.get("inflation_sha256") != navigation._digest(inflated):
        raise ValueError("planner did not consume this GPU inflation output")
    outputs = navigation._validation(payload, trajectory, inputs["truth"])
    validation = outputs["validation"]
    validation.update(
        kind="hil.mixed_validation.v1",
        source_hashes={
            name: navigation._digest(value) for name, value in inputs.items()
        },
        gpu_full_reference_match=True,
        gpu_cells_checked=len(inflated["blocked"]),
    )
    validation["checks"].extend(
        ["gpu_full_grid_cpu_reference", "gpu_output_used_by_planner"]
    )
    return outputs
