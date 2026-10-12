"""Marching squares: contour lines and filled contour regions from a grid."""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["contour_lines", "superlevel_loops", "auto_levels"]

# Segment table: case index -> list of (edge, edge). Edges: 0 bottom, 1 right,
# 2 top, 3 left. Saddles (5, 10) are resolved with the cell centre value.
_SEGMENTS = {
    1: [(3, 0)],
    2: [(0, 1)],
    3: [(3, 1)],
    4: [(1, 2)],
    6: [(0, 2)],
    7: [(3, 2)],
    8: [(3, 2)],
    9: [(0, 2)],
    11: [(1, 2)],
    12: [(3, 1)],
    13: [(0, 1)],
    14: [(3, 0)],
}


def _edge_key(i: int, j: int, edge: int) -> tuple[str, int, int]:
    if edge == 0:
        return ("h", i, j)
    if edge == 2:
        return ("h", i + 1, j)
    if edge == 3:
        return ("v", i, j)
    return ("v", i, j + 1)


def contour_lines(
    x: Sequence[float], y: Sequence[float], z: np.ndarray, level: float
) -> list[list[tuple[float, float]]]:
    """Polylines where ``z == level``.

    ``z`` has shape ``(len(y), len(x))`` (rows follow y). Closed loops repeat
    their first point at the end.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    ny, nx = z.shape
    if (ny, nx) != (len(y), len(x)):
        raise ValueError("z must have shape (len(y), len(x))")
    above = np.nan_to_num(z, nan=-np.inf) >= level
    idx = (
        above[:-1, :-1].astype(int)
        + 2 * above[:-1, 1:]
        + 4 * above[1:, 1:]
        + 8 * above[1:, :-1]
    )
    points: dict[tuple[str, int, int], tuple[float, float]] = {}

    def point(key: tuple[str, int, int]) -> tuple[float, float]:
        if key not in points:
            kind, i, j = key
            if kind == "h":  # between (i, j) and (i, j + 1)
                a, b = z[i, j], z[i, j + 1]
                t = 0.5 if b == a else (level - a) / (b - a)
                points[key] = (x[j] + (x[j + 1] - x[j]) * t, y[i])
            else:  # between (i, j) and (i + 1, j)
                a, b = z[i, j], z[i + 1, j]
                t = 0.5 if b == a else (level - a) / (b - a)
                points[key] = (x[j], y[i] + (y[i + 1] - y[i]) * t)
        return points[key]

    neighbours: dict[tuple[str, int, int], list[tuple[str, int, int]]] = {}
    cells = np.argwhere((idx != 0) & (idx != 15))
    for i, j in cells:
        case = int(idx[i, j])
        if case in (5, 10):
            centre = np.nanmean([z[i, j], z[i, j + 1], z[i + 1, j + 1], z[i + 1, j]])
            joined = centre >= level
            if case == 5:
                segs = [(0, 1), (3, 2)] if joined else [(3, 0), (1, 2)]
            else:
                segs = [(3, 0), (1, 2)] if joined else [(0, 1), (3, 2)]
        else:
            segs = _SEGMENTS[case]
        for e1, e2 in segs:
            k1, k2 = _edge_key(int(i), int(j), e1), _edge_key(int(i), int(j), e2)
            neighbours.setdefault(k1, []).append(k2)
            neighbours.setdefault(k2, []).append(k1)

    lines: list[list[tuple[float, float]]] = []
    visited: set[tuple[str, int, int]] = set()

    def walk(start: tuple[str, int, int]) -> list[tuple[str, int, int]]:
        chain = [start]
        visited.add(start)
        prev, cur = None, start
        while True:
            nxt = None
            for cand in neighbours[cur]:
                if cand != prev and cand not in visited:
                    nxt = cand
                    break
            if nxt is None:
                # closed loop: link back to the start
                if len(chain) > 2 and start in neighbours[cur]:
                    chain.append(start)
                return chain
            visited.add(nxt)
            chain.append(nxt)
            prev, cur = cur, nxt

    # open chains start at the domain boundary (degree 1), then the loops
    for key, nbrs in neighbours.items():
        if len(nbrs) == 1 and key not in visited:
            lines.append([point(k) for k in walk(key)])
    for key in neighbours:
        if key not in visited:
            lines.append([point(k) for k in walk(key)])
    return [ln for ln in lines if len(ln) >= 2]


def superlevel_loops(
    x: Sequence[float], y: Sequence[float], z: np.ndarray, level: float
) -> list[list[tuple[float, float]]]:
    """Closed loops bounding ``{z >= level}``; fill them with the even-odd rule.

    The grid is padded with a low border so every contour closes; the loops
    then extend slightly outside the grid and must be clipped to it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    low = np.nanmin(z) - 1.0 - abs(level)
    zp = np.full((z.shape[0] + 2, z.shape[1] + 2), low)
    zp[1:-1, 1:-1] = np.nan_to_num(z, nan=low)
    dx0, dx1 = (x[1] - x[0]), (x[-1] - x[-2])
    dy0, dy1 = (y[1] - y[0]), (y[-1] - y[-2])
    xp = np.concatenate([[x[0] - dx0 * 1e-3], x, [x[-1] + dx1 * 1e-3]])
    yp = np.concatenate([[y[0] - dy0 * 1e-3], y, [y[-1] + dy1 * 1e-3]])
    return contour_lines(xp, yp, zp, level)


def auto_levels(z: np.ndarray, n: int = 10) -> list[float]:
    """About ``n`` nicely rounded levels strictly inside the range of ``z``."""
    from .ticks import linear_ticks

    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    levels = [v for v in linear_ticks(lo, hi, n) if lo < v < hi]
    return levels or [0.5 * (lo + hi)]
