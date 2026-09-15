from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from examples.hardware_workloads import PORT_TYPES, WorkloadError, execute
from examples.hardware_workloads.geometry import (
    grid_line,
    ray_rectangle_distance,
    segment_rectangle_distance,
)
from examples.hardware_workloads.pipeline import MAX_PAYLOAD_BYTES


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _run(seed: int = 19) -> dict[str, dict]:
    artifacts = execute("hil_sensor", {}, seed)
    artifacts.update(
        execute("hil_mapping", {"observations": artifacts["observations"]}, seed)
    )
    artifacts.update(execute("hil_planning", {"map": artifacts["map"]}, seed))
    artifacts.update(
        execute(
            "hil_validation",
            {port: artifacts[port] for port in ("map", "trajectory", "truth")},
            seed,
        )
    )
    return artifacts


@pytest.fixture(scope="module")
def pipeline() -> dict[str, dict]:
    return _run()


def _validate(artifacts: dict[str, dict]) -> dict:
    return execute(
        "hil_validation",
        {port: artifacts[port] for port in ("map", "trajectory", "truth")},
        19,
    )


def _replace_scene_identity(artifacts: dict[str, dict]) -> None:
    """Re-sign modified truth: prove the validator checks geometry, not only hashes."""
    truth = artifacts["truth"]
    scene_id = _digest({key: truth[key] for key in ("seed", "bounds_m", "obstacles_m")})
    for port in ("truth", "map", "trajectory"):
        artifacts[port]["scene_id"] = scene_id
    artifacts["trajectory"]["source_hashes"]["map"] = _digest(artifacts["map"])


@pytest.mark.parametrize("seed", [0, 1, 2, 19, 42, 123, 999, 2**32 - 1])
def test_real_computation_pipeline_succeeds_across_seeds(seed: int) -> None:
    artifacts = _run(seed)
    result = artifacts["validation"]
    assert result["valid"] is True
    assert result["minimum_obstacle_clearance_m"] > result["robot_radius_m"]
    assert result["path_length_m"] > math.dist(
        artifacts["map"]["start_m"], artifacts["map"]["goal_m"]
    )
    assert artifacts["map"]["statistics"]["rays_integrated"] == 5120
    assert artifacts["trajectory"]["expanded_cells"] > 100
    assert result["maximum_speed_m_s"] <= artifacts["truth"]["limits"]["max_speed_m_s"]
    assert (
        result["maximum_acceleration_m_s2"]
        <= artifacts["truth"]["limits"]["max_acceleration_m_s2"]
    )
    for payload in artifacts.values():
        encoded = json.dumps(payload, allow_nan=False).encode()
        assert len(encoded) < MAX_PAYLOAD_BYTES
        assert payload["schema_version"] == 1
        assert json.loads(encoded) == payload


def test_seed_is_deterministic_and_changes_real_observations_and_plan(
    pipeline: dict,
) -> None:
    assert _run() == pipeline
    other = _run(42)
    assert other["observations"]["scans"] != pipeline["observations"]["scans"]
    assert other["map"]["cells"] != pipeline["map"]["cells"]
    assert other["trajectory"]["waypoints_m"] != pipeline["trajectory"]["waypoints_m"]


def test_downstream_seed_cannot_replace_actual_inputs(pipeline: dict) -> None:
    assert (
        execute("hil_mapping", {"observations": pipeline["observations"]}, 123)["map"]
        == pipeline["map"]
    )
    assert (
        execute("hil_planning", {"map": pipeline["map"]}, 123)["trajectory"]
        == pipeline["trajectory"]
    )


def test_output_contract_and_provenance(pipeline: dict) -> None:
    for ports in PORT_TYPES.values():
        for name, kind in ports["outputs"].items():
            assert pipeline[name]["kind"] == kind
    assert pipeline["map"]["source_hashes"] == {
        "observations": _digest(pipeline["observations"])
    }
    assert pipeline["trajectory"]["source_hashes"] == {"map": _digest(pipeline["map"])}
    assert pipeline["validation"]["source_hashes"] == {
        port: _digest(pipeline[port]) for port in ("map", "trajectory", "truth")
    }


def test_mapping_only_clears_measured_rays_and_keeps_unknown_blocked(
    pipeline: dict,
) -> None:
    observations = copy.deepcopy(pipeline["observations"])
    observations["scans"] = [
        {
            "origin_m": [1.5, 0.75],
            "angle_min_rad": 0.0,
            "angle_increment_rad": 1.0,
            "ranges_m": [1.0],
            "hits": [False],
        }
    ]
    mapped = execute("hil_mapping", {"observations": observations}, 0)["map"]
    assert mapped["cells"][6][12] == 0
    assert mapped["cells"][6][20] == 0
    assert mapped["cells"][6][21] == -1
    assert mapped["statistics"]["unknown_cells"] > 6000
    with pytest.raises(WorkloadError, match="unknown"):
        execute("hil_planning", {"map": mapped}, 0)
    observations["scans"][0]["hits"][0] = True
    with_hit = execute("hil_mapping", {"observations": observations}, 0)["map"]
    assert with_hit["cells"][6][20] == 1


@pytest.mark.parametrize("value", [-1, 1])
def test_planning_rejects_unknown_or_occupied_barrier(
    pipeline: dict, value: int
) -> None:
    payload = copy.deepcopy(pipeline["map"])
    for row in payload["cells"]:
        row[40] = value
    with pytest.raises(WorkloadError, match="no collision-free path"):
        execute("hil_planning", {"map": payload}, 19)


def test_planning_rejects_blocked_start(pipeline: dict) -> None:
    payload = copy.deepcopy(pipeline["map"])
    payload["cells"][6][6] = 1
    with pytest.raises(WorkloadError, match="start or goal"):
        execute("hil_planning", {"map": payload}, 19)


def test_real_planning_consumes_changed_goal(pipeline: dict) -> None:
    payload = copy.deepcopy(pipeline["map"])
    payload["goal_m"] = [1.8125, 1.8125]
    planned = execute("hil_planning", {"map": payload}, 19)["trajectory"]
    assert planned["waypoints_m"][-1] == payload["goal_m"]
    assert planned["path_length_m"] < pipeline["trajectory"]["path_length_m"]


def test_validation_catches_new_obstacle_between_valid_waypoints(
    pipeline: dict,
) -> None:
    artifacts = copy.deepcopy(pipeline)
    segment = next(
        item
        for item in artifacts["trajectory"]["segments"]
        if item["kind"] == "translate"
        and math.dist(item["start"][:2], item["end"][:2]) > 1
    )
    x = (segment["start"][0] + segment["end"][0]) / 2
    y = (segment["start"][1] + segment["end"][1]) / 2
    obstacle = [x - 0.01, y - 0.01, x + 0.01, y + 0.01]
    artifacts["truth"]["obstacles_m"].append(obstacle)
    _replace_scene_identity(artifacts)
    assert all(
        segment_rectangle_distance(point, point, obstacle)
        > artifacts["truth"]["limits"]["robot_radius_m"]
        for point in artifacts["trajectory"]["waypoints_m"]
    )
    with pytest.raises(WorkloadError, match="independent scene truth"):
        _validate(artifacts)


def test_validation_checks_truth_when_map_falsely_claims_empty_world(
    pipeline: dict,
) -> None:
    artifacts = copy.deepcopy(pipeline)
    payload = artifacts["map"]
    payload["cells"] = [
        [0] * payload["width_cells"] for _ in range(payload["height_cells"])
    ]
    artifacts["trajectory"] = execute("hil_planning", {"map": payload}, 19)[
        "trajectory"
    ]
    with pytest.raises(WorkloadError, match="independent scene truth"):
        _validate(artifacts)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("peak_speed_m_s", 99.0, "speed or acceleration"),
        ("acceleration_m_s2", 99.0, "speed or acceleration"),
        ("duration_s", 0.1, "trapezoidal velocity timing"),
        ("start_time_s", 1.0, "discontinuous"),
    ],
)
def test_validation_catches_corrupted_motion(
    pipeline: dict, field: str, value: float, match: str
) -> None:
    artifacts = copy.deepcopy(pipeline)
    segment = next(
        item
        for item in artifacts["trajectory"]["segments"]
        if item["kind"] == "translate"
    )
    segment[field] = value
    with pytest.raises(WorkloadError, match=match):
        _validate(artifacts)


def test_validation_catches_corrupted_rotation(pipeline: dict) -> None:
    artifacts = copy.deepcopy(pipeline)
    segment = next(
        item for item in artifacts["trajectory"]["segments"] if item["kind"] == "rotate"
    )
    segment["yaw_rate_rad_s"] = 99
    with pytest.raises(WorkloadError, match="yaw-rate"):
        _validate(artifacts)


def test_validation_rejects_mismatched_artifact_identity(pipeline: dict) -> None:
    artifacts = copy.deepcopy(pipeline)
    artifacts["map"]["statistics"]["rays_integrated"] += 1
    with pytest.raises(WorkloadError, match="not computed from this map"):
        _validate(artifacts)


def test_validation_rejects_mutated_truth_identity(pipeline: dict) -> None:
    artifacts = copy.deepcopy(pipeline)
    artifacts["truth"]["obstacles_m"][0][3] += 0.01
    with pytest.raises(WorkloadError, match="truth content"):
        _validate(artifacts)


@pytest.mark.parametrize(
    "task,inputs,seed",
    [
        ("yolo", {}, 19),
        ("hil_sensor", {"unexpected": {}}, 19),
        ("hil_mapping", {}, 19),
        ("hil_mapping", {"observations": {}}, 19),
        ("hil_sensor", {}, -1),
        ("hil_sensor", {}, True),
        ("hil_sensor", {}, 2**32),
        ("hil_sensor", [], 19),
    ],
)
def test_rejects_invalid_requests(task: str, inputs: dict, seed: int) -> None:
    with pytest.raises(WorkloadError):
        execute(task, inputs, seed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("resolution_m", 0),
        ("resolution_m", float("nan")),
        ("width_cells", 256),
        ("origin_m", [1, 1]),
        ("start_m", [-1, 0]),
        ("cells", []),
        ("limits", {}),
        ("schema_version", True),
    ],
)
def test_rejects_malformed_maps(pipeline: dict, field: str, value: object) -> None:
    payload = copy.deepcopy(pipeline["map"])
    payload[field] = value
    with pytest.raises(WorkloadError):
        execute("hil_planning", {"map": payload}, 19)


def test_rejects_excessive_robot_inflation(pipeline: dict) -> None:
    payload = copy.deepcopy(pipeline["map"])
    payload["limits"]["robot_radius_m"] = 10.0
    with pytest.raises(WorkloadError, match="eight-cell inflation"):
        execute("hil_planning", {"map": payload}, 19)


def test_rejects_excessive_rays_and_invalid_scan_hits(pipeline: dict) -> None:
    observations = copy.deepcopy(pipeline["observations"])
    observations["scans"] = observations["scans"] * 4
    with pytest.raises(WorkloadError, match="ray count"):
        execute("hil_mapping", {"observations": observations}, 19)
    observations = copy.deepcopy(pipeline["observations"])
    observations["scans"][0]["hits"][0] = 1
    with pytest.raises(WorkloadError, match="boolean"):
        execute("hil_mapping", {"observations": observations}, 19)


def test_rejects_oversized_payload(pipeline: dict) -> None:
    payload = copy.deepcopy(pipeline["map"])
    payload["padding"] = "x" * MAX_PAYLOAD_BYTES
    with pytest.raises(WorkloadError, match="2 MiB"):
        execute("hil_planning", {"map": payload}, 19)


@pytest.mark.parametrize(
    "start,end,rectangle,expected",
    [
        ([0, 0], [10, 0], [4.9, -0.1, 5.1, 0.1], 0.0),
        ([0, 0], [10, 0], [4.9, 1.0, 5.1, 2.0], 1.0),
        ([0, 0], [0, 0], [3, 4, 4, 5], 5.0),
        ([0, 0], [1, 1], [0.5, 0.5, 2, 2], 0.0),
    ],
)
def test_continuous_segment_collision_geometry(
    start: list, end: list, rectangle: list, expected: float
) -> None:
    assert segment_rectangle_distance(start, end, rectangle) == pytest.approx(expected)


def test_ray_geometry_and_grid_traversal() -> None:
    assert ray_rectangle_distance([0, 0], [1, 0], [2, -1, 3, 1]) == 2
    assert math.isinf(ray_rectangle_distance([0, 0], [0, 1], [2, -1, 3, 1]))
    assert list(grid_line((0, 0), (2, 2))) == [(0, 0), (1, 1), (2, 2)]


def _worker(request: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "examples.hardware_workloads.worker"],
        input=request,
        text=True,
        capture_output=True,
        timeout=10,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_worker_stdout_is_one_json_result() -> None:
    result = _worker(json.dumps({"task_type": "hil_sensor", "inputs": {}, "seed": 19}))
    assert result.returncode == 0
    assert result.stderr == ""
    assert len(result.stdout.splitlines()) == 1
    assert set(json.loads(result.stdout)) == {"observations", "truth"}


@pytest.mark.parametrize(
    "request_text",
    [
        "bad json",
        "{}",
        "[]",
        '{"task_type":"hil_sensor","inputs":{},"seed":NaN}',
        '{"task_type":"bad","inputs":{},"seed":19}',
    ],
)
def test_worker_invalid_request_fails_without_success_output(request_text: str) -> None:
    result = _worker(request_text)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "hardware workload failed:" in result.stderr
