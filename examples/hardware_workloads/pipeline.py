"""Synthetic acquisition, real mapping/planning, and independent truth validation.

The fixture is a known-pose survey of a static 2D world, not YOLO, VLA, SLAM,
real LiDAR capture, or a robot controller. Unknown map cells are not traversable.
All downstream computations consume the actual preceding task's JSON outputs.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from collections.abc import Mapping
from typing import Any

from .geometry import grid_line, ray_rectangle_distance, segment_rectangle_distance

PORT_TYPES = {
    "hil_sensor": {
        "inputs": {},
        "outputs": {"observations": "hil.observations.v1", "truth": "hil.truth.v1"},
    },
    "hil_mapping": {
        "inputs": {"observations": "hil.observations.v1"},
        "outputs": {"map": "hil.occupancy_map.v1"},
    },
    "hil_planning": {
        "inputs": {"map": "hil.occupancy_map.v1"},
        "outputs": {"trajectory": "hil.trajectory.v1"},
    },
    "hil_validation": {
        "inputs": {
            "map": "hil.occupancy_map.v1",
            "trajectory": "hil.trajectory.v1",
            "truth": "hil.truth.v1",
        },
        "outputs": {"validation": "hil.validation.v1"},
    },
}

MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
MAX_GRID_CELLS = 16_384
MAX_RAYS = 16_384
MAX_PATH_POINTS = 16_384
MAX_SEGMENTS = 2_048
_LIMITS = {
    "robot_radius_m": 0.18,
    "max_speed_m_s": 0.6,
    "max_acceleration_m_s2": 0.4,
    "max_yaw_rate_rad_s": 0.8,
}


class WorkloadError(ValueError):
    """A malformed input or an unsafe/unachievable workload result."""


def _canonical(payload: Any) -> bytes:
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise WorkloadError("payload must contain finite JSON values") from exc
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise WorkloadError("payload exceeds the 2 MiB limit")
    return encoded


def _digest(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _number(
    value: Any, name: str, *, minimum: float = -math.inf, maximum: float = math.inf
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise WorkloadError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise WorkloadError(f"{name} must be between {minimum} and {maximum}")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise WorkloadError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _vector(value: Any, name: str, length: int = 2) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise WorkloadError(f"{name} must have {length} numeric components")
    return [
        _number(component, name, minimum=-1000, maximum=1000) for component in value
    ]


def _limits(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(_LIMITS):
        raise WorkloadError(
            "limits must contain robot radius, speed, acceleration, and yaw-rate limits"
        )
    return {
        key: _number(item, key, minimum=0.001, maximum=10)
        for key, item in value.items()
    }


def _grid_spec(payload: dict) -> tuple[int, int, float]:
    width = _integer(payload.get("width_cells"), "width_cells", 4, 256)
    height = _integer(payload.get("height_cells"), "height_cells", 4, 256)
    if width * height > MAX_GRID_CELLS:
        raise WorkloadError("grid exceeds maximum cell count")
    resolution = _number(
        payload.get("resolution_m"), "resolution_m", minimum=0.025, maximum=1
    )
    if payload.get("origin_m") != [0.0, 0.0]:
        raise WorkloadError("the MVP supports only origin_m=[0,0]")
    _limits(payload.get("limits"))
    for name in ("start_m", "goal_m"):
        x, y = _vector(payload.get(name), name)
        if not 0 <= x < width * resolution or not 0 <= y < height * resolution:
            raise WorkloadError(f"{name} is outside the map")
    if not isinstance(payload.get("scene_id"), str) or len(payload["scene_id"]) != 64:
        raise WorkloadError("scene_id must be a SHA256 identity")
    return width, height, resolution


def _metadata(payload: dict, kind: str, **extra: Any) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "scene_id": payload["scene_id"],
        "width_cells": payload["width_cells"],
        "height_cells": payload["height_cells"],
        "resolution_m": payload["resolution_m"],
        "origin_m": list(payload["origin_m"]),
        "start_m": list(payload["start_m"]),
        "goal_m": list(payload["goal_m"]),
        "limits": dict(payload["limits"]),
        **extra,
    }


def _sensor(seed: int) -> dict[str, dict]:
    rng = random.Random(seed)
    obstacles = [
        [3.0, 0.0, 3.5, round(5.25 + rng.uniform(-0.15, 0.15), 6)],
        [6.0, round(2.75 + rng.uniform(-0.15, 0.15), 6), 6.5, 8.0],
        [9.0, 0.0, 9.5, round(5.25 + rng.uniform(-0.15, 0.15), 6)],
    ]
    scene = {"seed": seed, "bounds_m": [0.0, 0.0, 12.0, 8.0], "obstacles_m": obstacles}
    base = {
        "scene_id": _digest(scene),
        "width_cells": 96,
        "height_cells": 64,
        "resolution_m": 0.125,
        "origin_m": [0.0, 0.0],
        "start_m": [0.8125, 0.8125],
        "goal_m": [11.1875, 7.1875],
        "limits": dict(_LIMITS),
    }
    truth = _metadata(base, "hil.truth.v1", **scene)
    scans = []
    for x in (1.5, 4.75, 7.75, 10.75):
        for y in (0.75, 2.25, 4.0, 5.75, 7.25):
            ranges, hits = [], []
            for beam in range(256):
                angle = beam * math.tau / 256
                direction = [math.cos(angle), math.sin(angle)]
                bounds = []
                for axis, extent in ((0, 12.0), (1, 8.0)):
                    component = direction[axis]
                    if abs(component) > 1e-12:
                        bounds.append(
                            ((extent if component > 0 else 0.0) - (x, y)[axis])
                            / component
                        )
                distance = min(
                    bounds
                    + [
                        ray_rectangle_distance((x, y), direction, rect)
                        for rect in obstacles
                    ]
                )
                hit = distance <= 16.0
                # A bounded synthetic range perturbation; the map clips at its bounds.
                observed = min(16.0, max(0.001, distance + rng.uniform(-0.004, 0.004)))
                ranges.append(round(observed, 6))
                hits.append(hit)
            scans.append(
                {
                    "origin_m": [x, y],
                    "angle_min_rad": 0.0,
                    "angle_increment_rad": math.tau / 256,
                    "ranges_m": ranges,
                    "hits": hits,
                }
            )
    observations = _metadata(
        base,
        "hil.observations.v1",
        seed=seed,
        acquisition="synthetic_known_pose_2d_range_survey",
        range_noise_bound_m=0.004,
        max_range_m=16.0,
        scans=scans,
    )
    return {"observations": observations, "truth": truth}


def _mapping(observations: dict) -> dict[str, dict]:
    width, height, resolution = _grid_spec(observations)
    noise = _number(
        observations.get("range_noise_bound_m"),
        "range_noise_bound_m",
        minimum=0,
        maximum=0.05,
    )
    maximum_range = _number(
        observations.get("max_range_m"), "max_range_m", minimum=0.1, maximum=100
    )
    scans = observations.get("scans")
    if not isinstance(scans, list) or not 1 <= len(scans) <= 128:
        raise WorkloadError("scans must contain 1 to 128 known-pose scans")
    free: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    ray_count = 0
    for scan in scans:
        if not isinstance(scan, dict):
            raise WorkloadError("scan must be an object")
        origin = _vector(scan.get("origin_m"), "scan origin_m")
        if (
            not 0 <= origin[0] < width * resolution
            or not 0 <= origin[1] < height * resolution
        ):
            raise WorkloadError("scan origin is outside the map")
        angle_min = _number(
            scan.get("angle_min_rad"),
            "angle_min_rad",
            minimum=-math.tau,
            maximum=math.tau,
        )
        increment = _number(
            scan.get("angle_increment_rad"),
            "angle_increment_rad",
            minimum=1e-6,
            maximum=math.tau,
        )
        ranges, hits = scan.get("ranges_m"), scan.get("hits")
        if (
            not isinstance(ranges, list)
            or not isinstance(hits, list)
            or len(ranges) != len(hits)
            or not ranges
        ):
            raise WorkloadError(
                "scan ranges_m and hits must be matching nonempty lists"
            )
        ray_count += len(ranges)
        if ray_count > MAX_RAYS:
            raise WorkloadError("scan ray count exceeds maximum")
        source = (int(origin[0] / resolution), int(origin[1] / resolution))
        for index, (raw_range, hit) in enumerate(zip(ranges, hits)):
            distance = _number(
                raw_range, "range_m", minimum=0.001, maximum=maximum_range + noise
            )
            if not isinstance(hit, bool):
                raise WorkloadError("scan hit flag must be a boolean")
            angle = angle_min + index * increment
            endpoint = (
                origin[0] + distance * math.cos(angle),
                origin[1] + distance * math.sin(angle),
            )
            target = (
                math.floor(endpoint[0] / resolution),
                math.floor(endpoint[1] / resolution),
            )
            # Do not clear unseen cells or anything beyond the measured endpoint.
            for cell in grid_line(source, target):
                if not 0 <= cell[0] < width or not 0 <= cell[1] < height:
                    break
                if not hit or cell != target:
                    free.add(cell)
            if hit and 0 <= target[0] < width and 0 <= target[1] < height:
                occupied.add(target)
    cells = [
        [1 if (x, y) in occupied else 0 if (x, y) in free else -1 for x in range(width)]
        for y in range(height)
    ]
    result = _metadata(
        observations,
        "hil.occupancy_map.v1",
        source_hashes={"observations": _digest(observations)},
        cells=cells,
        range_noise_bound_m=noise,
        statistics={
            "rays_integrated": ray_count,
            "occupied_cells": len(occupied),
            "known_free_cells": len(free - occupied),
            "unknown_cells": width * height - len(free | occupied),
        },
    )
    return {"map": result}


def _validated_map(payload: dict) -> tuple[int, int, float, list[list[int]]]:
    width, height, resolution = _grid_spec(payload)
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != height:
        raise WorkloadError("map cells do not match height_cells")
    for row in cells:
        if not isinstance(row, list) or len(row) != width:
            raise WorkloadError("map cells do not match width_cells")
        if any(type(value) is not int or value not in (-1, 0, 1) for value in row):
            raise WorkloadError(
                "map cells must be -1 (unknown), 0 (free), or 1 (occupied)"
            )
    return width, height, resolution, cells


def _blocked(payload: dict) -> set[tuple[int, int]]:
    width, height, resolution, cells = _validated_map(payload)
    radius = _limits(payload["limits"])["robot_radius_m"]
    noise = _number(
        payload.get("range_noise_bound_m"),
        "range_noise_bound_m",
        minimum=0,
        maximum=0.05,
    )
    # Include half a cell diagonal: occupancy cells are squares, not point obstacles.
    inflation = radius + noise + resolution * math.sqrt(2) / 2
    reach = math.ceil(inflation / resolution)
    if reach > 8:
        raise WorkloadError(
            "robot footprint exceeds the MVP's eight-cell inflation bound"
        )
    offsets = [
        (dx, dy)
        for dx in range(-reach, reach + 1)
        for dy in range(-reach, reach + 1)
        if math.hypot(dx, dy) * resolution <= inflation
    ]
    blocked: set[tuple[int, int]] = set()
    for y, row in enumerate(cells):
        for x, value in enumerate(row):
            if value != 0:
                for dx, dy in offsets:
                    if 0 <= x + dx < width and 0 <= y + dy < height:
                        blocked.add((x + dx, y + dy))
            if (
                min(
                    (x + 0.5) * resolution,
                    (y + 0.5) * resolution,
                    (width - x - 0.5) * resolution,
                    (height - y - 0.5) * resolution,
                )
                < radius
            ):
                blocked.add((x, y))
    return blocked


def _planning(payload: dict) -> dict[str, dict]:
    width, height, resolution, _ = _validated_map(payload)
    blocked = _blocked(payload)
    start = tuple(int(value / resolution) for value in payload["start_m"])
    goal = tuple(int(value / resolution) for value in payload["goal_m"])
    if start in blocked or goal in blocked:
        raise WorkloadError(
            "start or goal is occupied, unknown, or lacks robot clearance"
        )
    if start == goal:
        raise WorkloadError("start and goal must occupy different cells")
    frontier = [(abs(goal[0] - start[0]) + abs(goal[1] - start[1]), 0, start)]
    cost, parent = {start: 0}, {}
    expanded = 0
    while frontier:
        _, distance, current = heapq.heappop(frontier)
        if distance != cost[current]:
            continue
        expanded += 1
        if current == goal:
            break
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                not 0 <= neighbor[0] < width
                or not 0 <= neighbor[1] < height
                or neighbor in blocked
            ):
                continue
            next_cost = distance + 1
            if next_cost < cost.get(neighbor, math.inf):
                cost[neighbor], parent[neighbor] = next_cost, current
                heapq.heappush(
                    frontier,
                    (
                        next_cost
                        + abs(goal[0] - neighbor[0])
                        + abs(goal[1] - neighbor[1]),
                        next_cost,
                        neighbor,
                    ),
                )
    else:
        raise WorkloadError("no collision-free path exists in observed free space")
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    # Remove only collinear points: no shortcuts across unknown space or corners.
    reduced = [path[0]]
    for index in range(1, len(path) - 1):
        before = (
            path[index][0] - path[index - 1][0],
            path[index][1] - path[index - 1][1],
        )
        after = (
            path[index + 1][0] - path[index][0],
            path[index + 1][1] - path[index][1],
        )
        if before != after:
            reduced.append(path[index])
    reduced.append(path[-1])
    points = [[(x + 0.5) * resolution, (y + 0.5) * resolution] for x, y in reduced]
    # Connect the exact requested start/goal, not silently shifted cell centers.
    if points[0] != payload["start_m"]:
        points.insert(0, list(payload["start_m"]))
    if points[-1] != payload["goal_m"]:
        points.append(list(payload["goal_m"]))
    limits = _limits(payload["limits"])
    speed, acceleration = limits["max_speed_m_s"], limits["max_acceleration_m_s2"]
    segments, clock, yaw, length = [], 0.0, 0.0, 0.0
    for first, second in zip(points, points[1:]):
        heading = math.atan2(second[1] - first[1], second[0] - first[0])
        turn = (heading - yaw + math.pi) % math.tau - math.pi
        if abs(turn) > 1e-10:
            duration = abs(turn) / limits["max_yaw_rate_rad_s"]
            segments.append(
                {
                    "kind": "rotate",
                    "start": [*first, yaw],
                    "end": [*first, heading],
                    "start_time_s": clock,
                    "duration_s": duration,
                    "yaw_rate_rad_s": turn / duration,
                }
            )
            clock += duration
        distance = math.dist(first, second)
        peak = min(speed, math.sqrt(distance * acceleration))
        acceleration_time = peak / acceleration
        cruise_time = max(0.0, (distance - peak * acceleration_time) / peak)
        duration = 2 * acceleration_time + cruise_time
        segments.append(
            {
                "kind": "translate",
                "start": [*first, heading],
                "end": [*second, heading],
                "start_time_s": clock,
                "duration_s": duration,
                "peak_speed_m_s": peak,
                "acceleration_m_s2": acceleration,
                "acceleration_time_s": acceleration_time,
                "cruise_time_s": cruise_time,
            }
        )
        clock += duration
        length += distance
        yaw = heading
    if len(segments) > MAX_SEGMENTS:
        raise WorkloadError("computed trajectory exceeds maximum segment count")
    return {
        "trajectory": _metadata(
            payload,
            "hil.trajectory.v1",
            source_hashes={"map": _digest(payload)},
            waypoints_m=points,
            segments=segments,
            duration_s=clock,
            path_length_m=length,
            expanded_cells=expanded,
            motion_model="2d_circular_robot_stop_turn_trapezoidal_translate",
        )
    }


def _truth(truth: dict, payload: dict) -> list[list[float]]:
    _grid_spec(truth)
    for key in (
        "scene_id",
        "width_cells",
        "height_cells",
        "resolution_m",
        "origin_m",
        "start_m",
        "goal_m",
        "limits",
    ):
        if truth.get(key) != payload.get(key):
            raise WorkloadError(f"truth/map mismatch in {key}")
    seed = _integer(truth.get("seed"), "truth seed", 0, 2**32 - 1)
    bounds = _vector(truth.get("bounds_m"), "bounds_m", 4)
    if bounds != [
        0,
        0,
        payload["width_cells"] * payload["resolution_m"],
        payload["height_cells"] * payload["resolution_m"],
    ]:
        raise WorkloadError("truth bounds do not match map dimensions")
    rectangles = truth.get("obstacles_m")
    if not isinstance(rectangles, list) or len(rectangles) > 128:
        raise WorkloadError("truth obstacles must be a list of at most 128 rectangles")
    result = []
    for item in rectangles:
        rectangle = _vector(item, "obstacle rectangle", 4)
        if (
            not 0 <= rectangle[0] < rectangle[2] <= bounds[2]
            or not 0 <= rectangle[1] < rectangle[3] <= bounds[3]
        ):
            raise WorkloadError(
                "obstacle rectangle is degenerate or outside truth bounds"
            )
        result.append(rectangle)
    if (
        _digest(
            {
                "seed": seed,
                "bounds_m": truth["bounds_m"],
                "obstacles_m": truth["obstacles_m"],
            }
        )
        != truth["scene_id"]
    ):
        raise WorkloadError("truth content does not match scene_id")
    return result


def _validation(payload: dict, trajectory: dict, truth: dict) -> dict[str, dict]:
    width, height, resolution, _ = _validated_map(payload)
    rectangles = _truth(truth, payload)
    limits = _limits(truth["limits"])
    for key in (
        "scene_id",
        "width_cells",
        "height_cells",
        "resolution_m",
        "origin_m",
        "start_m",
        "goal_m",
        "limits",
    ):
        if trajectory.get(key) != payload.get(key):
            raise WorkloadError(f"trajectory/map mismatch in {key}")
    if trajectory.get("source_hashes") != {"map": _digest(payload)}:
        raise WorkloadError("trajectory was not computed from this map")
    points = trajectory.get("waypoints_m")
    if not isinstance(points, list) or not 2 <= len(points) <= MAX_PATH_POINTS:
        raise WorkloadError("trajectory must contain a bounded list of waypoints")
    points = [_vector(point, "waypoint") for point in points]
    if points[0] != truth["start_m"] or points[-1] != truth["goal_m"]:
        raise WorkloadError("trajectory does not connect the requested start and goal")
    segments = trajectory.get("segments")
    if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_SEGMENTS:
        raise WorkloadError("trajectory must contain a bounded list of motion segments")
    blocked = _blocked(payload)
    radius = limits["robot_radius_m"]
    min_clearance, elapsed, length, max_speed, max_acceleration, max_yaw_rate = (
        math.inf,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    previous = [*truth["start_m"], 0.0]
    translated_points = [list(truth["start_m"])]
    for segment in segments:
        if not isinstance(segment, dict):
            raise WorkloadError("motion segment must be an object")
        first, second = (
            _vector(segment.get("start"), "segment start", 3),
            _vector(segment.get("end"), "segment end", 3),
        )
        duration = _number(
            segment.get("duration_s"),
            "segment duration_s",
            minimum=1e-9,
            maximum=100000,
        )
        started = _number(
            segment.get("start_time_s"), "segment start_time_s", minimum=0, maximum=1e8
        )
        if any(
            not math.isclose(a, b, abs_tol=1e-8) for a, b in zip(first, previous)
        ) or not math.isclose(started, elapsed, abs_tol=1e-8):
            raise WorkloadError("trajectory has discontinuous poses or timestamps")
        displacement = math.dist(first[:2], second[:2])
        for point in (first, second):
            wall_clearance = min(
                point[0],
                point[1],
                width * resolution - point[0],
                height * resolution - point[1],
            )
            min_clearance = min(min_clearance, wall_clearance)
            if wall_clearance < radius - 1e-9:
                raise WorkloadError("trajectory collides with world boundary")
        for rectangle in rectangles:
            clearance = segment_rectangle_distance(first[:2], second[:2], rectangle)
            min_clearance = min(min_clearance, clearance)
            if clearance < radius - 1e-9:
                raise WorkloadError("trajectory collides with independent scene truth")
        steps = max(1, math.ceil(displacement / (resolution / 2)))
        for step in range(steps + 1):
            fraction = step / steps
            cell = (
                math.floor((first[0] + fraction * (second[0] - first[0])) / resolution),
                math.floor((first[1] + fraction * (second[1] - first[1])) / resolution),
            )
            if cell in blocked:
                raise WorkloadError(
                    "trajectory traverses occupied/unknown map or lacks clearance"
                )
        kind = segment.get("kind")
        if kind == "rotate":
            if displacement > 1e-9:
                raise WorkloadError("rotation changes position")
            turn = (second[2] - first[2] + math.pi) % math.tau - math.pi
            rate = _number(segment.get("yaw_rate_rad_s"), "yaw_rate_rad_s")
            if abs(rate) > limits["max_yaw_rate_rad_s"] + 1e-9 or not math.isclose(
                rate * duration, turn, abs_tol=1e-8
            ):
                raise WorkloadError("rotation violates yaw-rate limit or timing")
            max_yaw_rate = max(max_yaw_rate, abs(rate))
        elif kind == "translate":
            speed = _number(
                segment.get("peak_speed_m_s"), "peak_speed_m_s", minimum=1e-9
            )
            acceleration = _number(
                segment.get("acceleration_m_s2"), "acceleration_m_s2", minimum=1e-9
            )
            ramp = _number(
                segment.get("acceleration_time_s"), "acceleration_time_s", minimum=1e-9
            )
            cruise = _number(segment.get("cruise_time_s"), "cruise_time_s", minimum=0)
            if (
                speed > limits["max_speed_m_s"] + 1e-9
                or acceleration > limits["max_acceleration_m_s2"] + 1e-9
            ):
                raise WorkloadError("translation exceeds speed or acceleration limit")
            heading = math.atan2(second[1] - first[1], second[0] - first[0])
            if (
                displacement <= 0
                or abs((heading - first[2] + math.pi) % math.tau - math.pi) > 1e-8
                or first[2] != second[2]
            ):
                raise WorkloadError("translation violates nonholonomic heading")
            if (
                not math.isclose(acceleration * ramp, speed, abs_tol=1e-8)
                or not math.isclose(2 * ramp + cruise, duration, abs_tol=1e-8)
                or not math.isclose(speed * (ramp + cruise), displacement, abs_tol=1e-8)
            ):
                raise WorkloadError("translation violates trapezoidal velocity timing")
            max_speed, max_acceleration = (
                max(max_speed, speed),
                max(max_acceleration, acceleration),
            )
            length += displacement
            translated_points.append(second[:2])
        else:
            raise WorkloadError("unsupported motion segment kind")
        elapsed += duration
        previous = second
    if translated_points != points or previous[:2] != truth["goal_m"]:
        raise WorkloadError("trajectory segments do not match waypoints/goal")
    if not math.isclose(
        _number(trajectory.get("duration_s"), "duration_s"), elapsed, abs_tol=1e-8
    ) or not math.isclose(
        _number(trajectory.get("path_length_m"), "path_length_m"), length, abs_tol=1e-8
    ):
        raise WorkloadError("trajectory summary does not match its segments")
    return {
        "validation": {
            "schema_version": 1,
            "kind": "hil.validation.v1",
            "scene_id": truth["scene_id"],
            "valid": True,
            "source_hashes": {
                "map": _digest(payload),
                "trajectory": _digest(trajectory),
                "truth": _digest(truth),
            },
            "checked_segments": len(segments),
            "path_length_m": length,
            "planned_motion_duration_s": elapsed,
            "minimum_obstacle_clearance_m": min_clearance,
            "robot_radius_m": radius,
            "maximum_speed_m_s": max_speed,
            "maximum_acceleration_m_s2": max_acceleration,
            "maximum_yaw_rate_rad_s": max_yaw_rate,
            "checks": [
                "source_identity",
                "continuous_truth_collision",
                "observed_map_clearance",
                "start_goal",
                "pose_time_continuity",
                "linear_speed_acceleration",
                "yaw_rate",
                "nonholonomic_heading",
            ],
        }
    }


def execute(task_type: str, inputs: Mapping[str, dict], seed: int) -> dict[str, dict]:
    """Execute exactly one bounded task, returning finite JSON output port values.

    Invalid or unsafe results raise WorkloadError rather than returning a fake
    successful validation artifact. ``seed`` affects acquisition only; later
    stages are functions of their actual input artifacts.
    """
    if not isinstance(task_type, str) or task_type not in PORT_TYPES:
        raise WorkloadError(f"unsupported task type: {task_type!r}")
    _integer(seed, "seed", 0, 2**32 - 1)
    if not isinstance(inputs, Mapping) or set(inputs) != set(
        PORT_TYPES[task_type]["inputs"]
    ):
        raise WorkloadError(
            f"{task_type} requires exactly these inputs: {sorted(PORT_TYPES[task_type]['inputs'])}"
        )
    for port, kind in PORT_TYPES[task_type]["inputs"].items():
        payload = inputs[port]
        if (
            not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
            or payload.get("kind") != kind
        ):
            raise WorkloadError(f"{port} has an unsupported schema or kind")
        _canonical(payload)
    if task_type == "hil_sensor":
        output = _sensor(seed)
    elif task_type == "hil_mapping":
        output = _mapping(inputs["observations"])
    elif task_type == "hil_planning":
        output = _planning(inputs["map"])
    else:
        output = _validation(inputs["map"], inputs["trajectory"], inputs["truth"])
    for payload in output.values():
        _canonical(payload)
    return output
