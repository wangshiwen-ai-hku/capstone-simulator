"""Small, dependency-free geometry routines; no robot or middleware dependencies."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence


def ray_rectangle_distance(
    origin: Sequence[float], direction: Sequence[float], rectangle: Sequence[float]
) -> float:
    """Return the first forward ray/closed-AABB intersection, or infinity."""
    near, far = -math.inf, math.inf
    for axis in (0, 1):
        lower, upper = rectangle[axis], rectangle[axis + 2]
        if abs(direction[axis]) < 1e-12:
            if not lower <= origin[axis] <= upper:
                return math.inf
            continue
        first = (lower - origin[axis]) / direction[axis]
        second = (upper - origin[axis]) / direction[axis]
        near = max(near, min(first, second))
        far = min(far, max(first, second))
    if far < max(near, 0.0):
        return math.inf
    return max(near, 0.0)


def grid_line(
    start: tuple[int, int], end: tuple[int, int]
) -> Iterator[tuple[int, int]]:
    """Integer Bresenham traversal of a measured sensor ray."""
    x, y = start
    dx, dy = abs(end[0] - x), -abs(end[1] - y)
    sx, sy = (1 if x < end[0] else -1), (1 if y < end[1] else -1)
    error = dx + dy
    while True:
        yield x, y
        if (x, y) == end:
            return
        twice = 2 * error
        if twice >= dy:
            error += dy
            x += sx
        if twice <= dx:
            error += dx
            y += sy


def _point_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    squared = dx * dx + dy * dy
    if squared == 0:
        return math.dist(point, start)
    fraction = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / squared),
    )
    return math.hypot(
        point[0] - start[0] - fraction * dx, point[1] - start[1] - fraction * dy
    )


def segment_rectangle_distance(
    start: Sequence[float], end: Sequence[float], rectangle: Sequence[float]
) -> float:
    """Exact segment/AABB distance, including collisions between sample points."""
    direction = (end[0] - start[0], end[1] - start[1])
    if ray_rectangle_distance(start, direction, rectangle) <= 1.0:
        return 0.0
    xmin, ymin, xmax, ymax = rectangle
    corners = ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    distances = [_point_segment_distance(corner, start, end) for corner in corners]
    for point in (start, end):
        distances.append(
            math.hypot(
                max(xmin - point[0], 0.0, point[0] - xmax),
                max(ymin - point[1], 0.0, point[1] - ymax),
            )
        )
    return min(distances)
