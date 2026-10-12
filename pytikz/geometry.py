"""Plane geometry helpers computed in Python.

Useful for constructions (circumcircles, intersections, ...) and used
internally for clipping plot data to the axes box.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .coords import Point, as_point

__all__ = [
    "line_intersection",
    "circle_line_intersections",
    "circle_circle_intersections",
    "circumcenter",
    "circumcircle",
    "incircle",
    "centroid",
    "orthocenter",
    "foot",
    "reflect",
    "polygon_area",
    "regular_polygon",
    "clip_polyline",
    "clip_polygon",
]


def line_intersection(a: Any, b: Any, c: Any, d: Any) -> Point:
    """Intersection of the (infinite) lines ``ab`` and ``cd``."""
    a, b, c, d = map(as_point, (a, b, c, d))
    r, s = b - a, d - c
    denom = r.cross(s)
    if abs(denom) < 1e-12:
        raise ValueError("the lines are parallel")
    t = (c - a).cross(s) / denom
    return a + r * t


def circle_line_intersections(center: Any, radius: float, a: Any, b: Any) -> list[Point]:
    """Points where the line ``ab`` meets the circle (0, 1 or 2 points)."""
    c, a, b = as_point(center), as_point(a), as_point(b)
    d = b - a
    f = a - c
    A, B, C = d.dot(d), 2 * f.dot(d), f.dot(f) - radius * radius
    disc = B * B - 4 * A * C
    if disc < 0:
        return []
    if disc == 0:
        return [a + d * (-B / (2 * A))]
    s = math.sqrt(disc)
    return [a + d * ((-B - s) / (2 * A)), a + d * ((-B + s) / (2 * A))]


def circle_circle_intersections(c1: Any, r1: float, c2: Any, r2: float) -> list[Point]:
    """Intersection points of two circles."""
    p1, p2 = as_point(c1), as_point(c2)
    d = p1.distance(p2)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(max(0.0, r1 * r1 - a * a))
    mid = p1 + (p2 - p1) * (a / d)
    off = (p2 - p1).perp() * (h / d)
    return [mid + off, mid - off] if h > 0 else [mid]


def circumcenter(a: Any, b: Any, c: Any) -> Point:
    a, b, c = map(as_point, (a, b, c))
    d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
    if abs(d) < 1e-12:
        raise ValueError("the points are collinear")
    ux = ((a.dot(a)) * (b.y - c.y) + (b.dot(b)) * (c.y - a.y) + (c.dot(c)) * (a.y - b.y)) / d
    uy = ((a.dot(a)) * (c.x - b.x) + (b.dot(b)) * (a.x - c.x) + (c.dot(c)) * (b.x - a.x)) / d
    return Point(ux, uy)


def circumcircle(a: Any, b: Any, c: Any) -> tuple[Point, float]:
    """``(center, radius)`` of the circle through three points."""
    o = circumcenter(a, b, c)
    return o, o.distance(as_point(a))


def incircle(a: Any, b: Any, c: Any) -> tuple[Point, float]:
    """``(center, radius)`` of the inscribed circle of triangle ``abc``."""
    a, b, c = map(as_point, (a, b, c))
    la, lb, lc = b.distance(c), a.distance(c), a.distance(b)
    p = la + lb + lc
    center = (a * la + b * lb + c * lc) / p
    s = p / 2
    area = abs((b - a).cross(c - a)) / 2
    return center, area / s


def centroid(*points: Any) -> Point:
    pts = [as_point(p) for p in points]
    return Point(sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))


def orthocenter(a: Any, b: Any, c: Any) -> Point:
    a, b, c = map(as_point, (a, b, c))
    return line_intersection(a, foot(a, b, c), b, foot(b, a, c))


def foot(p: Any, a: Any, b: Any) -> Point:
    """Foot of the perpendicular from ``p`` to line ``ab``."""
    p, a, b = map(as_point, (p, a, b))
    d = b - a
    return a + d * ((p - a).dot(d) / d.dot(d))


def reflect(p: Any, a: Any, b: Any) -> Point:
    """Mirror image of ``p`` in the line ``ab``."""
    p = as_point(p)
    f = foot(p, a, b)
    return f * 2 - p


def polygon_area(points: Iterable[Any]) -> float:
    """Signed area (positive for counter-clockwise polygons)."""
    pts = [as_point(p) for p in points]
    return 0.5 * sum(pts[i].cross(pts[(i + 1) % len(pts)]) for i in range(len(pts)))


def regular_polygon(n: int, radius: float = 1.0, center: Any = (0, 0), rotation: float = 90.0) -> list[Point]:
    """Vertices of a regular ``n``-gon (first vertex at angle ``rotation``)."""
    c = as_point(center)
    return [c + Point.polar(rotation + 360.0 * k / n, radius) for k in range(n)]


# ---------------------------------------------------------------------------
# clipping (used by plot axes)
# ---------------------------------------------------------------------------

def _clip_segment(
    x0: float, y0: float, x1: float, y1: float, box: tuple[float, float, float, float]
) -> tuple[float, float, float, float, float, float] | None:
    """Liang-Barsky. Returns (t0, t1, cx0, cy0, cx1, cy1) or None if outside."""
    xmin, ymin, xmax, ymax = box
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return None
                t0 = max(t0, t)
            else:
                if t < t0:
                    return None
                t1 = min(t1, t)
    return t0, t1, x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy


def clip_polyline(
    xs: Sequence[float], ys: Sequence[float], box: tuple[float, float, float, float]
) -> list[list[tuple[float, float]]]:
    """Clip a polyline to ``box = (xmin, ymin, xmax, ymax)``.

    Non-finite values break the line. Returns the visible pieces.
    """
    pieces: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 2:
            pieces.append(current)
        current = []

    prev: tuple[float, float] | None = None
    for x, y in zip(xs, ys):
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            flush()
            prev = None
            continue
        if prev is None:
            prev = (x, y)
            continue
        res = _clip_segment(prev[0], prev[1], x, y, box)
        if res is None:
            flush()
        else:
            t0, t1, cx0, cy0, cx1, cy1 = res
            if not current or t0 > 0:
                flush()
                current = [(cx0, cy0)]
            current.append((cx1, cy1))
            if t1 < 1:
                flush()
        prev = (x, y)
    flush()
    return pieces


def clip_polygon(
    points: Sequence[tuple[float, float]], box: tuple[float, float, float, float]
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clipping of a polygon to an axis-aligned box."""
    xmin, ymin, xmax, ymax = box
    pts = [(float(x), float(y)) for x, y in points if math.isfinite(x) and math.isfinite(y)]

    def clip(pts: list[tuple[float, float]], inside: Any, intersect: Any) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        if not pts:
            return out
        prev = pts[-1]
        for cur in pts:
            if inside(cur):
                if not inside(prev):
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(intersect(prev, cur))
            prev = cur
        return out

    def at_x(x: float) -> Any:
        return lambda p, q: (x, p[1] + (q[1] - p[1]) * (x - p[0]) / (q[0] - p[0]))

    def at_y(y: float) -> Any:
        return lambda p, q: (p[0] + (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1]), y)

    pts = clip(pts, lambda p: p[0] >= xmin, at_x(xmin))
    pts = clip(pts, lambda p: p[0] <= xmax, at_x(xmax))
    pts = clip(pts, lambda p: p[1] >= ymin, at_y(ymin))
    pts = clip(pts, lambda p: p[1] <= ymax, at_y(ymax))
    return pts
