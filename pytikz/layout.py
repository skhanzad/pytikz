"""Graph and tree layout algorithms (positions in arbitrary units).

All functions return ``{node: Point}``. They are deterministic: the spring
layout is seeded.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Callable, Hashable, Iterable, Sequence

import numpy as np

from .coords import Point

__all__ = [
    "circular_layout",
    "shell_layout",
    "grid_layout",
    "spring_layout",
    "layered_layout",
    "bipartite_layout",
    "tidy_tree_layout",
]

Node = Hashable


def circular_layout(nodes: Sequence[Node], radius: float = 1.0, start: float = 90.0,
                    clockwise: bool = False) -> dict[Node, Point]:
    """Nodes evenly spaced on a circle, the first one at angle ``start``."""
    n = max(1, len(nodes))
    sign = -1.0 if clockwise else 1.0
    return {v: Point.polar(start + sign * 360.0 * k / n, radius) for k, v in enumerate(nodes)}


def shell_layout(shells: Sequence[Sequence[Node]], radii: Sequence[float] | None = None,
                 start: float = 90.0) -> dict[Node, Point]:
    """Concentric circles; a shell with a single node sits at the centre."""
    pos: dict[Node, Point] = {}
    for k, shell in enumerate(shells):
        r = radii[k] if radii is not None else float(k if len(shells[0]) == 1 else k + 1)
        if len(shell) == 1 and r == 0:
            pos[shell[0]] = Point(0, 0)
            continue
        pos.update(circular_layout(shell, r, start))
    return pos


def grid_layout(nodes: Sequence[Node], columns: int | None = None, spacing: float = 1.0) -> dict[Node, Point]:
    """Row-major grid, top row first."""
    columns = columns or max(1, math.ceil(math.sqrt(len(nodes))))
    return {v: Point((k % columns) * spacing, -(k // columns) * spacing) for k, v in enumerate(nodes)}


def bipartite_layout(left: Sequence[Node], right: Sequence[Node], gap: float = 2.0,
                     spacing: float = 1.0) -> dict[Node, Point]:
    """Two vertical columns."""
    pos = {}
    for k, v in enumerate(left):
        pos[v] = Point(0, ((len(left) - 1) / 2 - k) * spacing)
    for k, v in enumerate(right):
        pos[v] = Point(gap, ((len(right) - 1) / 2 - k) * spacing)
    return pos


def spring_layout(nodes: Sequence[Node], edges: Iterable[tuple[Node, Node]], *, seed: int = 0,
                  iterations: int = 400, align: bool = True) -> dict[Node, Point]:
    """Fruchterman-Reingold force-directed layout, scaled to fit [-1, 1]."""
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: Point(0, 0)}
    index = {v: i for i, v in enumerate(nodes)}
    pairs = np.array([(index[u], index[v]) for u, v in edges if u != v], dtype=int).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * 2 - 1
    k = math.sqrt(4.0 / n)
    t = 0.2
    cooling = t / (iterations + 1)
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, 1.0)
        dist = np.maximum(dist, 1e-3)
        force = (k * k / dist**2)[:, :, None] * delta  # repulsion along delta/dist * k^2/dist
        np.einsum("iik->ik", force)[:] = 0
        disp = force.sum(axis=1)
        if len(pairs):
            d = pos[pairs[:, 0]] - pos[pairs[:, 1]]
            dl = np.maximum(np.linalg.norm(d, axis=1), 1e-3)
            pull = (dl / k)[:, None] * d  # attraction dl^2/k along d/dl
            np.add.at(disp, pairs[:, 0], -pull)
            np.add.at(disp, pairs[:, 1], pull)
        length = np.maximum(np.linalg.norm(disp, axis=1), 1e-9)
        pos += disp / length[:, None] * np.minimum(length, t)[:, None]
        t -= cooling
    pos -= pos.mean(axis=0)
    if align:
        # rotate so the principal axis is horizontal
        cov = pos.T @ pos
        w, v = np.linalg.eigh(cov)
        main = v[:, np.argmax(w)]
        ang = math.atan2(main[1], main[0])
        c, s = math.cos(-ang), math.sin(-ang)
        pos = pos @ np.array([[c, s], [-s, c]])
    scale = np.abs(pos).max() or 1.0
    pos /= scale
    return {v: Point(*pos[i]) for i, v in enumerate(nodes)}


def layered_layout(nodes: Sequence[Node], edges: Iterable[tuple[Node, Node]], *, layer_gap: float = 1.0,
                   node_gap: float = 1.0, sweeps: int = 12) -> dict[Node, Point]:
    """Sugiyama-style layout of a directed graph: sources at the top.

    Layers come from the longest path; nodes within a layer are ordered by
    the barycentre heuristic to reduce crossings.
    """
    nodes = list(nodes)
    edges = [(u, v) for u, v in edges if u != v]
    succ: dict[Node, list[Node]] = defaultdict(list)
    pred: dict[Node, list[Node]] = defaultdict(list)
    # break cycles: drop edges that close a cycle in DFS order
    state: dict[Node, int] = {}
    kept: list[tuple[Node, Node]] = []
    adj: dict[Node, list[Node]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    back: set[tuple[Node, Node]] = set()

    def dfs(start: Node) -> None:
        stack = [(start, iter(adj[start]))]
        state[start] = 1
        while stack:
            u, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[u] = 2
                stack.pop()
                continue
            if state.get(nxt, 0) == 1:
                back.add((u, nxt))
            elif state.get(nxt, 0) == 0:
                state[nxt] = 1
                stack.append((nxt, iter(adj[nxt])))

    for v in nodes:
        if state.get(v, 0) == 0:
            dfs(v)
    for u, v in edges:
        if (u, v) in back:
            u, v = v, u
        kept.append((u, v))
        succ[u].append(v)
        pred[v].append(u)
    # longest-path layering
    layer: dict[Node, int] = {}
    indeg = {v: len(pred[v]) for v in nodes}
    queue = deque(v for v in nodes if indeg[v] == 0)
    for v in queue:
        layer[v] = 0
    while queue:
        u = queue.popleft()
        for v in succ[u]:
            layer[v] = max(layer.get(v, 0), layer[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    for v in nodes:
        layer.setdefault(v, 0)
    # sources (inputs, parameters) sit right above their first consumer
    for v in nodes:
        if not pred[v] and succ[v]:
            layer[v] = max(0, min(layer[w] for w in succ[v]) - 1)
    n_layers = max(layer.values()) + 1 if layer else 0
    layers: list[list[Node]] = [[] for _ in range(n_layers)]
    for v in nodes:
        layers[layer[v]].append(v)
    order = {v: i for lay in layers for i, v in enumerate(lay)}

    def barycentre(v: Node, neighbours: dict[Node, list[Node]]) -> float:
        ns = [order[u] for u in neighbours[v] if u in order]
        return sum(ns) / len(ns) if ns else order[v]

    for sweep in range(sweeps):
        downward = sweep % 2 == 0
        rng = range(1, n_layers) if downward else range(n_layers - 2, -1, -1)
        neighbours = pred if downward else succ
        for li in rng:
            keys = {v: (barycentre(v, neighbours), order[v]) for v in layers[li]}
            layers[li].sort(key=keys.__getitem__)
            for i, v in enumerate(layers[li]):
                order[v] = i
    pos: dict[Node, Point] = {}
    for li, lay in enumerate(layers):
        for i, v in enumerate(lay):
            pos[v] = Point((i - (len(lay) - 1) / 2) * node_gap, -li * layer_gap)
    # nudge single-node layers towards the mean of their neighbours
    for lay in layers:
        if len(lay) == 1:
            v = lay[0]
            ns = [pos[u].x for u in pred[v] + succ[v]]
            if ns:
                pos[v] = Point(sum(ns) / len(ns), pos[v].y)
    return pos


def tidy_tree_layout(
    root: Any,
    children: Callable[[Any], Sequence[Any]],
    *,
    width: Callable[[Any], float] = lambda _: 1.0,
    gap: float = 0.4,
    level_distance: float = 1.0,
) -> dict[int, Point]:
    """Tidy tree drawing (Reingold-Tilford style), keyed by ``id(node)``.

    Subtrees are packed as closely as their contours allow, keeping
    ``gap`` between neighbouring nodes of estimated ``width``; parents are
    centred above their children.
    """
    rel: dict[int, float] = {}  # x offset of a node relative to its parent
    depth_of: dict[int, int] = {}

    def build(node: Any, depth: int) -> dict[int, tuple[float, float]]:
        """Return the contour {depth: (left, right)} relative to ``node``."""
        depth_of[id(node)] = depth
        kids = list(children(node))
        half = width(node) / 2
        if not kids:
            return {depth: (-half, half)}
        contours = []
        offsets = []
        acc: dict[int, tuple[float, float]] = {}
        for k, kid in enumerate(kids):
            c = build(kid, depth + 1)
            if k == 0:
                shift = 0.0
            else:
                shift = max(acc[d][1] - c[d][0] + gap for d in c if d in acc) if any(d in acc for d in c) else 0.0
            offsets.append(shift)
            contours.append(c)
            for d, (lo, hi) in c.items():
                lo, hi = lo + shift, hi + shift
                if d in acc:
                    acc[d] = (min(acc[d][0], lo), max(acc[d][1], hi))
                else:
                    acc[d] = (lo, hi)
        centre = (offsets[0] + offsets[-1]) / 2
        for kid, off in zip(kids, offsets):
            rel[id(kid)] = off - centre
        merged = {d: (lo - centre, hi - centre) for d, (lo, hi) in acc.items()}
        merged[depth] = (-half, half)
        return merged

    build(root, 0)
    pos: dict[int, Point] = {}

    def place(node: Any, x: float) -> None:
        pos[id(node)] = Point(x, -depth_of[id(node)] * level_distance)
        for kid in children(node):
            place(kid, x + rel[id(kid)])

    place(root, 0.0)
    return pos
