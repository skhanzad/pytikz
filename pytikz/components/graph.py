"""Graphs (networks) and finite automata with automatic layouts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Mapping, Sequence

from .._fmt import fmt, label_text
from ..color import color_text
from ..coords import NodeRef, Point, as_point
from ..elements import Element
from ..render import LazyName
from ..theme import ARG
from ..layout import (
    bipartite_layout,
    circular_layout,
    grid_layout,
    layered_layout,
    shell_layout,
    spring_layout,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..render import RenderContext

__all__ = ["Graph", "Automaton"]


def _slug(value: Any) -> str:
    text = str(value)
    out = "".join(ch if ch.isalnum() else "-" for ch in text)
    return out or "v"


class _Edge:
    __slots__ = ("u", "v", "label", "color", "width", "bend", "style", "label_side", "dashed")

    def __init__(self, u: Any, v: Any, label: Any = None, color: Any = None, width: float | None = None,
                 bend: float | None = None, style: str | None = None, label_side: str | None = None,
                 dashed: bool = False):
        self.u, self.v = u, v
        self.label = label
        self.color = color
        self.width = width
        self.bend = bend
        self.style = style
        self.label_side = label_side
        self.dashed = dashed


class Graph(Element):
    """A graph drawn with circles and lines.

    Parameters
    ----------
    nodes:
        Node ids, or a mapping ``{id: label}``. Nodes that only appear in
        ``edges`` are added automatically.
    edges:
        ``(u, v)``, ``(u, v, label)`` or ``(u, v, {"label": ..., "color": ...,
        "bend": 20, "dashed": True})``.
    directed:
        Draw arrow tips.
    layout:
        ``"spring"``, ``"circular"``, ``"shell"``, ``"layered"`` (top-down
        DAG), ``"grid"``, ``"bipartite"`` or a mapping of positions.
    scale:
        Size of the layout in cm (radius for circular, half-width for spring,
        spacing for grid/layered).
    """

    def __init__(
        self,
        nodes: Iterable[Hashable] | Mapping[Hashable, Any] = (),
        edges: Iterable[Any] = (),
        *,
        directed: bool = False,
        layout: str | Mapping[Hashable, Any] = "spring",
        at: Any = (0, 0),
        scale: float = 2.5,
        color: str = "Blue",
        node_size: float = 0.62,
        labels: bool | Mapping[Hashable, Any] = True,
        math_labels: bool = False,
        edge_color: Any = None,
        edge_width: float = 0.8,
        edge_labels: str = "on",
        shells: Sequence[Sequence[Hashable]] | None = None,
        bipartite: tuple[Sequence[Hashable], Sequence[Hashable]] | None = None,
        columns: int | None = None,
        seed: int = 0,
        font: str = r"\small",
        node_style: str | None = None,
        name: str | None = None,
    ):
        self._lname = LazyName(self, self._PREFIX, name)
        self.nodes: list[Hashable] = []
        self._slugs: dict[Hashable, str] = {}
        if isinstance(nodes, Mapping):
            self._labels = dict(nodes)
        else:
            self._labels = {}
        for v in nodes:
            self._register(v)
        if isinstance(labels, Mapping):
            self._labels.update(labels)
        self.show_labels = labels is not False
        self.math_labels = math_labels
        self.edges: list[_Edge] = []
        for e in edges:
            self.add_edge(*e) if not isinstance(e, _Edge) else self.edges.append(e)
        self.directed = directed
        self.layout = layout
        self.origin = as_point(at)
        self.scale = scale
        self.color = color
        self.node_size = node_size
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.edge_labels = edge_labels
        self.shells = shells
        self.bipartite = bipartite
        self.columns = columns
        self.seed = seed
        self.font = font
        self.node_style = node_style
        self._node_colors: dict[Hashable, str] = {}
        self._node_extra: dict[Hashable, str] = {}
        self._positions: dict[Hashable, Point] | None = None

    _PREFIX = "g"

    @property
    def name(self) -> str:
        """Prefix of the node names (``g``, ``g2``, ... unless given)."""
        return self._lname.resolve()

    # -- building ---------------------------------------------------------------------
    def _register(self, v: Hashable) -> None:
        """Add a node id and give it a TikZ-safe name unique within the graph."""
        if v in self._slugs:
            return
        if v not in self.nodes:
            self.nodes.append(v)
        base = _slug(v)
        slug, k = base, 1
        taken = set(self._slugs.values())
        while slug in taken:
            k += 1
            slug = f"{base}-{k}"
        self._slugs[v] = slug

    def add_node(self, v: Hashable, label: Any = None) -> "Graph":
        self._register(v)
        if label is not None:
            self._labels[v] = label
        self._positions = None
        return self

    def add_edge(self, u: Hashable, v: Hashable, label: Any = None, **style: Any) -> "Graph":
        """Add an edge; ``label`` may also be a dict of edge properties."""
        if isinstance(label, Mapping):
            style = {**label, **style}
            label = style.pop("label", None)
        for w in (u, v):
            self._register(w)
        self.edges.append(_Edge(u, v, label, **style))
        self._positions = None
        return self

    # -- references --------------------------------------------------------------------
    def _slug_of(self, v: Hashable) -> str:
        if v not in self._slugs:
            self._register(v)  # nodes appended to ``graph.nodes`` by hand
        return self._slugs[v]

    def node_name(self, v: Hashable) -> str:
        return f"{self.name}-{self._slug_of(v)}"

    def node(self, v: Hashable) -> NodeRef:
        """Reference to the node of ``v``."""
        if v not in self.nodes:
            raise KeyError(v)
        return NodeRef(self._lname.sub("-" + self._slug_of(v)))

    def position(self, v: Hashable) -> Point:
        """Centre of ``v`` in picture coordinates."""
        return self.positions()[v]

    # -- styling -------------------------------------------------------------------------
    def highlight_node(self, v: Hashable, color: str = "Orange", style: str | None = None) -> "Graph":
        self._node_colors[v] = color
        if style:
            self._node_extra[v] = style
        return self

    def highlight_edge(self, u: Hashable, v: Hashable, color: str = "Orange", width: float = 2.0) -> "Graph":
        for e in self.edges:
            if (e.u, e.v) == (u, v) or (not self.directed and (e.v, e.u) == (u, v)):
                e.color = color
                e.width = width
        return self

    def highlight_path(self, path: Sequence[Hashable], color: str = "Orange", width: float = 2.0,
                       nodes: bool = True) -> "Graph":
        """Emphasise the edges (and nodes) along ``path``."""
        for a, b in zip(path, path[1:]):
            self.highlight_edge(a, b, color, width)
        if nodes:
            for v in path:
                self.highlight_node(v, color)
        return self

    # -- layout -----------------------------------------------------------------------------
    def positions(self) -> dict[Hashable, Point]:
        if self._positions is not None:
            return self._positions
        lay = self.layout
        pairs = [(e.u, e.v) for e in self.edges]
        if isinstance(lay, Mapping):
            raw = {v: as_point(p) for v, p in lay.items()}
            missing = [v for v in self.nodes if v not in raw]
            if missing:
                raise ValueError(f"no position for nodes {missing}")
            pos = {v: self.origin + raw[v] for v in self.nodes}
        else:
            if lay == "circular":
                raw = circular_layout(self.nodes, 1.0)
            elif lay == "shell":
                raw = shell_layout(self.shells or [self.nodes])
            elif lay == "spring":
                raw = spring_layout(self.nodes, pairs, seed=self.seed)
            elif lay == "layered":
                raw = layered_layout(self.nodes, pairs)
            elif lay == "grid":
                raw = grid_layout(self.nodes, self.columns)
            elif lay == "bipartite":
                if self.bipartite is None:
                    raise ValueError("bipartite layout needs bipartite=(left, right)")
                raw = bipartite_layout(*self.bipartite)
            else:
                raise ValueError(f"unknown layout {lay!r}")
            pos = {v: self.origin + raw[v] * self.scale for v in self.nodes}
        self._positions = pos
        return pos

    # -- rendering -----------------------------------------------------------------------------
    def _label(self, v: Hashable) -> str:
        if not self.show_labels:
            return ""
        text = str(self._labels.get(v, v))
        if self.math_labels and not text.startswith("$"):
            return f"${text}$"
        return label_text(text)

    def _define_styles(self, ctx: "RenderContext") -> dict[str, str]:
        theme = ctx.theme
        size = fmt(self.node_size * 10, 2) + "mm"
        vertex = ctx.define_style(
            "vertex",
            f"{theme.arg_color()}, circle, draw={theme.stroke(ARG)}, fill={theme.fill(ARG, 20)}, minimum size={size}, "
            f"inner sep=1pt, line width=0.7pt, font={self.font}",
            default="Blue",
            prefix=self.name,
        )
        ec = self.edge_color or f"{theme.ink}!65"
        tip = ", ->, shorten >=1pt" if self.directed else ""
        edge = ctx.define_style("graph edge", f"{ec}, line width={self.edge_width:g}pt{tip}", prefix=self.name)
        label = ctx.define_style(
            "graph label",
            f"font=\\footnotesize, fill={theme.paper or 'white'}, inner sep=1.5pt"
            if self.edge_labels == "on"
            else "font=\\footnotesize, auto, inner sep=2pt",
            prefix=self.name,
        )
        return {"vertex": vertex, "edge": edge, "label": label}

    def _node_options(self, v: Hashable, styles: Mapping[str, str]) -> str:
        style = self.node_style or styles["vertex"]
        color = color_text(self._node_colors.get(v, self.color))
        opts = f"{style}={color}"
        if v in self._node_extra:
            opts += ", " + self._node_extra[v]
        return opts

    def render(self, ctx: "RenderContext") -> list[str]:
        styles = self._define_styles(ctx)
        pos = self.positions()
        out = [f"% graph {self.name}: {len(self.nodes)} nodes, {len(self.edges)} edges"]
        for v in self.nodes:
            p = pos[v]
            out.append(
                f"\\node[{self._node_options(v, styles)}] ({self.node_name(v)}) "
                f"at ({fmt(p.x)},{fmt(p.y)}) {{{self._label(v)}}};"
            )
        out += self._render_edges(ctx, pos, styles)
        return out

    # -- edge geometry ----------------------------------------------------------------------
    def _occupied_angles(self, v: Hashable, pos: Mapping[Hashable, Point]) -> list[float]:
        """Directions (degrees) already used at ``v`` by edges."""
        angles = []
        for e in self.edges:
            if e.u == e.v:
                continue
            if e.u == v:
                angles.append((pos[e.v] - pos[v]).angle())
            elif e.v == v:
                angles.append((pos[e.u] - pos[v]).angle())
        return angles

    def _loop_angle(self, v: Hashable, pos: Mapping[Hashable, Point]) -> float:
        occupied = self._occupied_angles(v, pos)
        centre = sum((p for p in pos.values()), Point(0, 0)) / max(1, len(pos))
        away = (pos[v] - centre).angle() if (pos[v] - centre).norm() > 1e-6 else 90.0

        def gap(a: float, b: float) -> float:
            d = abs(a - b) % 360.0
            return min(d, 360.0 - d)

        def score(a: float) -> float:
            clearance = min((gap(a, o) for o in occupied), default=180.0)
            return min(clearance, 100.0) + 8.0 * math.cos(math.radians(gap(a, away))) + 3.0 * math.cos(math.radians(gap(a, 90.0)))

        return max(range(0, 360, 15), key=score)

    def _blocked(self, e: _Edge, pos: Mapping[Hashable, Point]) -> bool:
        """Whether the straight edge would pass through another node."""
        a, b = pos[e.u], pos[e.v]
        d = b - a
        length2 = d.dot(d)
        if length2 == 0:
            return False
        radius = 0.55 * self.node_size + 0.05
        for w, p in pos.items():
            if w in (e.u, e.v):
                continue
            t = max(0.0, min(1.0, (p - a).dot(d) / length2))
            if (a + d * t).distance(p) < radius:
                return True
        return False

    def _edge_path(self, e: _Edge, pos: Mapping[Hashable, Point], reciprocal: bool) -> str:
        if e.u == e.v:
            ang = self._loop_angle(e.u, pos)
            return f"to[out={fmt(ang + 28)}, in={fmt(ang - 28)}, looseness=8, min distance=5mm]"
        bend = e.bend
        if bend is None and reciprocal:
            bend = 18
        if bend is None and self._blocked(e, pos):
            bend = 30
        if bend:
            side = "left" if bend > 0 else "right"
            return f"to[bend {side}={fmt(abs(bend))}]"
        return "--"

    def _render_edges(self, ctx: "RenderContext", pos: Mapping[Hashable, Point], styles: Mapping[str, str]) -> list[str]:
        out = []
        keys = {(e.u, e.v) for e in self.edges}
        for e in self.edges:
            reciprocal = self.directed and e.u != e.v and (e.v, e.u) in keys
            conn = self._edge_path(e, pos, reciprocal)
            opts = [styles["edge"]]
            if e.color is not None:
                opts.append(color_text(e.color))
            if e.width is not None:
                opts.append(f"line width={e.width:g}pt")
            if e.dashed:
                opts.append("dashed")
            if e.style:
                opts.append(e.style)
            lab = ""
            if e.label is not None:
                bent = conn != "--"
                side = e.label_side or ("auto" if bent and self.edge_labels == "on" else None)
                extra = f", {side}" if side else ""
                text_color = f", text={color_text(e.color)}" if e.color is not None else ""
                lab = f" node[{styles['label']}{extra}{text_color}] {{{label_text(e.label)}}}"
            out.append(f"\\draw[{', '.join(opts)}] ({self.node_name(e.u)}) {conn}{lab} ({self.node_name(e.v)});")
        return out


class Automaton(Graph):
    """A finite state machine (uses TikZ's ``automata`` library).

    >>> dfa = Automaton(["q0", "q1", "q2"],
    ...                 [("q0", "q1", "a"), ("q1", "q1", "b"), ("q1", "q2", "a")],
    ...                 initial="q0", accepting=["q2"], layout="layered")
    """

    def __init__(
        self,
        states: Iterable[Hashable] | Mapping[Hashable, Any],
        transitions: Iterable[Any],
        *,
        initial: Hashable | None = None,
        accepting: Iterable[Hashable] = (),
        layout: str | Mapping[Hashable, Any] = "circular",
        math_labels: bool = True,
        color: str = "Blue",
        node_size: float = 0.8,
        scale: float = 2.2,
        initial_where: str = "left",
        **kwargs: Any,
    ):
        super().__init__(states, transitions, directed=True, layout=layout, math_labels=math_labels,
                         color=color, node_size=node_size, scale=scale, edge_labels="auto", **kwargs)
        self.initial = initial
        self.accepting = set(accepting)
        self.initial_where = initial_where

    def positions(self) -> dict[Hashable, Point]:
        if self._positions is None and self.layout == "layered":
            pairs = [(e.u, e.v) for e in self.edges]
            raw = layered_layout(self.nodes, pairs)
            # automata read better left to right
            self._positions = {v: self.origin + Point(-p.y, -p.x) * self.scale for v, p in raw.items()}
            return self._positions
        return super().positions()

    _PREFIX = "fsm"
    _INITIAL_ANGLE = {"left": 180.0, "right": 0.0, "above": 90.0, "below": 270.0}

    def _node_options(self, v: Hashable, styles: Mapping[str, str]) -> str:
        opts = super()._node_options(v, styles)
        if v == self.initial:
            opts += f", initial, initial where={self.initial_where}, initial text="
        if v in self.accepting:
            # (``double distance`` implies ``double``, so it is set here only)
            opts += ", accepting, double distance=1.4pt"
        return opts

    def _occupied_angles(self, v: Hashable, pos: Mapping[Hashable, Point]) -> list[float]:
        angles = super()._occupied_angles(v, pos)
        if v == self.initial:
            angles.append(self._INITIAL_ANGLE.get(self.initial_where, 180.0))
        return angles

    def _define_styles(self, ctx: "RenderContext") -> dict[str, str]:
        ctx.require("automata")
        theme = ctx.theme
        size = fmt(self.node_size * 10, 2) + "mm"
        vertex = ctx.define_style(
            "state node",
            f"{theme.arg_color()}, state, draw={theme.stroke(ARG)}, fill={theme.fill(ARG, 20)}, minimum size={size}, "
            f"inner sep=1pt, line width=0.7pt, font={self.font}",
            default="Blue",
            prefix=self.name,
        )
        ec = self.edge_color or f"{theme.ink}!80"
        edge = ctx.define_style("transition", f"{ec}, line width={self.edge_width:g}pt, ->, shorten >=1pt",
                                prefix=self.name)
        label = ctx.define_style("transition label", "font=\\small, auto, inner sep=2pt", prefix=self.name)
        return {"vertex": vertex, "edge": edge, "label": label}

    def _render_edges(self, ctx: "RenderContext", pos: Mapping[Hashable, Point], styles: Mapping[str, str]) -> list[str]:
        # merge parallel transitions into one label "a, b"
        merged: dict[tuple[Hashable, Hashable], _Edge] = {}
        for e in self.edges:
            key = (e.u, e.v)
            if key in merged and e.label is not None:
                prev = merged[key]
                prev.label = f"{prev.label}, {e.label}" if prev.label is not None else e.label
            else:
                merged[key] = _Edge(e.u, e.v, e.label, e.color, e.width, e.bend, e.style, e.label_side, e.dashed)
        original = self.edges
        self.edges = list(merged.values())
        for e in self.edges:
            if e.label is not None and self.math_labels and not str(e.label).startswith("$"):
                e.label = f"${e.label}$"
        try:
            return super()._render_edges(ctx, pos, styles)
        finally:
            self.edges = original
