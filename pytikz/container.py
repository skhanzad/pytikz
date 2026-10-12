"""The drawing API shared by :class:`~pytikz.Picture` and :class:`Scope`."""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from .coords import NodeRef, Point, Polar, as_coord, as_point, is_numeric_coord
from .elements import (
    Arc,
    CircleOp,
    Comment,
    Coordinate,
    Element,
    EllipseOp,
    GridOp,
    Move,
    Node,
    Path,
    PathNode,
    Placement,
    PlotOp,
    Raw,
    Rel,
    To,
)
from .options import Options, _Raw
from ._fmt import label_text
from .units import Length, pt

if TYPE_CHECKING:  # pragma: no cover
    from .render import RenderContext

__all__ = ["Container", "Scope", "LazyValue"]

_PLACEMENTS = (
    "above",
    "below",
    "left",
    "right",
    "above_left",
    "above_right",
    "below_left",
    "below_right",
)


class LazyValue:
    """An option value computed at render time (e.g. names of fitted nodes)."""

    def __init__(self, func: Callable[[], str]):
        self.func = func

    def __str__(self) -> str:
        return self.func()


def _indent(lines: Iterable[str], prefix: str = "  ") -> list[str]:
    return [prefix + line if line else line for line in lines]


class Container:
    """Methods for adding nodes, paths and shapes. Every method returns the
    created element so it can be referenced later::

        a = pic.node("A", at=(0, 0), style="block")
        b = pic.node("B", right_of=a, distance=1.5, style="block=Orange")
        pic.arrow(a, b, label="$f$")
    """

    children: list[Element]

    def __init__(self) -> None:
        self.children = []

    # -- generic -----------------------------------------------------------------
    def add(self, *elements: Any) -> Any:
        """Add elements (or components); returns the last one."""
        for element in elements:
            if not isinstance(element, Element):
                raise TypeError(f"{element!r} is not a drawable element")
            self.children.append(element)
        return elements[-1] if elements else None

    def render_children(self, ctx: "RenderContext") -> list[str]:
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(ctx))
        return lines

    def raw(self, tex: str) -> Raw:
        """Insert verbatim TikZ code."""
        return self.add(Raw(tex))

    def comment(self, text: str) -> Comment:
        """Insert a ``%`` comment (handy to structure the generated code)."""
        return self.add(Comment(text))

    def scope(self, *args: Any, **options: Any) -> "Scope":
        """A ``scope`` whose options (shift, rotate, colors...) apply to its content."""
        return self.add(Scope(*args, **options))

    def background(self, *args: Any, **options: Any) -> "Scope":
        """A scope drawn behind everything else (``on background layer``)."""
        return self.add(Scope("on background layer", *args, **options))

    # -- nodes ---------------------------------------------------------------------
    def node(
        self,
        text: Any = "",
        at: Any = None,
        *,
        name: str | None = None,
        style: Any = None,
        distance: Any = None,
        **options: Any,
    ) -> Node:
        """Add a node.

        Place it ``at`` a coordinate, or relative to another node with
        ``right_of=other`` (also ``left_of``, ``above_of``, ``below_of``,
        ``above_right_of``...) and an optional ``distance`` between their
        borders (cm, or a ``(vertical, horizontal)`` pair for diagonals).
        """
        placement = None
        for direction in _PLACEMENTS:
            target = options.pop(f"{direction}_of", None)
            if target is not None:
                if placement is not None:
                    raise ValueError("give only one relative placement")
                placement = Placement(direction, target, distance)
        node = Node(text, at, name=name, options=Options(style, **options), placement=placement)
        return self.add(node)

    def coordinate(self, at: Any, name: str | None = None) -> Coordinate:
        """A named point (``\\coordinate``) to reference later."""
        return self.add(Coordinate(at, name))

    def dot(
        self,
        at: Any,
        label: Any = None,
        *,
        position: Any = "above",
        name: str | None = None,
        color: Any = None,
        size: float | Length | None = None,
        label_color: Any = None,
        **options: Any,
    ) -> Node:
        """A filled point with an optional label (``position`` may be an angle)."""
        opts = Options("dot")
        if color is not None:
            opts.set("fill", color)
        if size is not None:
            opts.set("minimum size", pt(float(size)) if isinstance(size, numbers.Real) else size)
        if label is not None:
            pos = position if isinstance(position, str) else f"{float(position):g}"
            label_opts = Options(text=label_color) if label_color is not None else Options()
            prefix = f"[{label_opts.inner()}]" if label_opts else ""
            opts.set("label", _Raw("{" + f"{prefix}{pos}:{{{label_text(label)}}}" + "}"))
        opts.update(**options)
        return self.add(Node("", at, name=name, options=opts))

    # -- paths -------------------------------------------------------------------------
    def draw(self, *items: Any, **options: Any) -> Path:
        """``\\draw`` a path through ``items``.

        Items are coordinates (consecutive ones are joined with ``--``),
        connector strings (``"--"``, ``"-|"``, ``"|-"``, ``"rectangle"``,
        ``"cycle"``) and path operations (:func:`~pytikz.to`,
        :func:`~pytikz.arc`, :func:`~pytikz.label`, ...)::

            pic.draw((0, 0), (1, 0), (1, 1), "cycle", fill="Blue!15")
        """
        return self.add(Path(items, "draw", Options(**options)))

    def fill(self, *items: Any, **options: Any) -> Path:
        """``\\fill`` a path (no outline)."""
        return self.add(Path(items, "fill", Options(**options)))

    def filldraw(self, *items: Any, **options: Any) -> Path:
        """``\\filldraw``: fill and outline."""
        return self.add(Path(items, "filldraw", Options(**options)))

    def path(self, *items: Any, **options: Any) -> Path:
        """``\\path``: an invisible path (useful for nodes and bounding boxes)."""
        return self.add(Path(items, "path", Options(**options)))

    def clip(self, *items: Any, **options: Any) -> Path:
        """``\\clip`` everything drawn after it in this scope."""
        return self.add(Path(items, "clip", Options(**options)))

    def shade(self, *items: Any, **options: Any) -> Path:
        """``\\shade`` a path, e.g. ``left_color="Blue", right_color="white"``."""
        return self.add(Path(items, "shade", Options(**options)))

    def line(self, *points: Any, **options: Any) -> Path:
        """A polyline through ``points``."""
        if len(points) == 1 and not is_numeric_coord(points[0]) and (
            isinstance(points[0], (list, tuple)) or hasattr(points[0], "shape")
        ):
            points = tuple(points[0])  # a list or an (n, 2) array of points
        return self.draw(*points, **options)

    def arrow(
        self,
        start: Any,
        end: Any,
        label: Any = None,
        *,
        tip: str | None = "->",
        bend: float | None = None,
        route: str | None = None,
        via: Sequence[Any] = (),
        label_side: str | None = "auto",
        label_pos: float = 0.5,
        label_style: Any = None,
        **options: Any,
    ) -> Path:
        """An arrow from ``start`` to ``end``.

        ``bend`` curves it (degrees, positive bends left), ``route="-|"`` or
        ``"|-"`` makes an elbow, ``via`` adds intermediate points. The label
        goes on the ``label_side`` (``"auto"`` = left of the direction of
        travel, ``"swap"``, ``"above"``, ``"below"``..., ``"sloped"`` or
        ``None`` to sit on the line).
        """
        opts = Options()
        if tip:
            opts.set(tip)
        opts.update(**options)
        items: list[Any] = [start, *via]
        if bend:
            items.append(To(bend_left=bend) if bend > 0 else To(bend_right=-bend))
        elif route in ("-|", "|-"):
            items.append(route)
        elif route not in (None, "--"):
            raise ValueError(f"unknown route {route!r}; use '--', '-|' or '|-'")
        else:
            items.append("--")
        if label is not None:
            items.append(_edge_label(label, label_side, label_pos, label_style))
        items.append(end)
        return self.add(Path(items, "draw", opts))

    def polygon(self, points: Iterable[Any], **options: Any) -> Path:
        """A closed polygon through ``points``."""
        return self.draw(*list(points), "cycle", **options)

    def rectangle(self, corner1: Any, corner2: Any, **options: Any) -> Path:
        """An axis-aligned rectangle between two opposite corners."""
        return self.draw(corner1, "rectangle", corner2, **options)

    def circle(self, center: Any, radius: float | Length | str = 1.0, **options: Any) -> Path:
        """A circle (radius in cm unless given as a :class:`Length`)."""
        return self.draw(center, CircleOp(radius), **options)

    def ellipse(self, center: Any, x_radius: Any, y_radius: Any, **options: Any) -> Path:
        """An axis-aligned ellipse."""
        return self.draw(center, EllipseOp(x_radius, y_radius), **options)

    def arc(
        self,
        center: Any,
        radius: float,
        start: float,
        end: float,
        **options: Any,
    ) -> Path:
        """A circular arc around ``center`` from angle ``start`` to ``end`` (degrees)."""
        begin = _center(center) + Polar(start, radius)
        return self.draw(begin, Arc(start, end, radius), **options)

    def sector(self, center: Any, radius: float, start: float, end: float, **options: Any) -> Path:
        """A pie slice (use ``fill=...`` to fill it)."""
        c = _center(center)
        begin = c + Polar(start, radius)
        return self.draw(c, begin, Arc(start, end, radius), "cycle", **options)

    def grid(self, corner1: Any, corner2: Any, step: Any = 1.0, **options: Any) -> Path:
        """Grid lines between two corners (thin gray lines by default)."""
        opts = Options(**options)
        if not opts:
            opts.add("help lines")
        return self.add(Path([corner1, GridOp(step), corner2], "draw", opts))

    def plot(
        self,
        points: Iterable[Any],
        *,
        smooth: bool = False,
        closed: bool = False,
        tension: float | None = None,
        **options: Any,
    ) -> Path:
        """A line through many points; ``smooth=True`` draws a smooth curve."""
        pts = list(points)
        if not smooth:
            items: list[Any] = list(pts)
            if closed:
                items.append("cycle")
            return self.draw(*items, **options)
        plot_opts = Options("smooth cycle" if closed else "smooth")
        if tension is not None:
            plot_opts.set("tension", tension)
        return self.add(Path([PlotOp(pts, plot_opts)], "draw", Options(**options)))

    def function(
        self,
        f: Callable[[float], float],
        domain: tuple[float, float],
        *,
        samples: int = 100,
        **options: Any,
    ) -> Path:
        """Plot ``y = f(x)`` in picture coordinates, sampled in Python.

        Points where ``f`` is undefined (errors, ``nan``, ``inf``) break the curve.
        """
        if samples < 2:
            raise ValueError("samples must be at least 2")
        a, b = domain
        items: list[Any] = []
        joined = False
        for i in range(samples):
            x = a + (b - a) * i / (samples - 1)
            try:
                y = float(f(x))
            except (ArithmeticError, ValueError):
                y = math.nan
            if math.isfinite(y):
                items.append((x, y))
                joined = True
            elif joined:
                items.append(Move())
                joined = False
        return self.draw(*items, **options)

    def parametric(
        self,
        f: Callable[[float], Any],
        domain: tuple[float, float],
        *,
        samples: int = 200,
        closed: bool = False,
        **options: Any,
    ) -> Path:
        """Plot the curve ``t -> f(t) = (x, y)`` in picture coordinates."""
        a, b = domain
        ts = [a + (b - a) * i / (samples - 1) for i in range(samples)]
        items: list[Any] = [f(t) for t in ts]
        if closed:
            items.append("cycle")
        return self.draw(*items, **options)

    # -- block diagrams -------------------------------------------------------------------
    def chain(
        self,
        items: Sequence[Any],
        *,
        at: Any = (0, 0),
        direction: str = "right",
        distance: Any = 0.8,
        style: Any = "block",
        arrows: bool = True,
        arrow_options: dict[str, Any] | None = None,
    ) -> list[Node]:
        """Place nodes one after another and connect them with arrows.

        ``items`` are texts or ``(text, {"style": ..., **options})`` pairs;
        ``direction`` is ``"right"``, ``"left"``, ``"up"`` or ``"down"``.
        Returns the nodes::

            enc, z, dec = pic.chain(["Encoder", "$z$", "Decoder"])
        """
        key = {"right": "right_of", "left": "left_of", "up": "above_of", "down": "below_of"}.get(direction)
        if key is None:
            raise ValueError("direction must be 'right', 'left', 'up' or 'down'")
        nodes: list[Node] = []
        for k, item in enumerate(items):
            if isinstance(item, tuple):
                text, extra = item[0], dict(item[1]) if len(item) > 1 else {}
            else:
                text, extra = item, {}
            st = extra.pop("style", style)
            if k == 0:
                node = self.node(text, at=at, style=st, **extra)
            else:
                extra.setdefault("distance", distance)
                node = self.node(text, style=st, **{key: nodes[-1]}, **extra)
            nodes.append(node)
        if arrows:
            for a, b in zip(nodes, nodes[1:]):
                self.arrow(a, b, **(arrow_options or {}))
        return nodes

    def bypass(
        self,
        start: Any,
        end: Any,
        *,
        side: str = "right",
        offset: float = 0.6,
        tip: str | None = "->",
        label: Any = None,
        rounded: float = 4.0,
        **options: Any,
    ) -> Path:
        """A connection that leaves ``start`` sideways, runs parallel to the
        main flow and turns back into ``end`` (residual / skip connections)::

            pic.bypass(x.east, add.east, side="right", offset=0.8)
        """
        shift = {"right": (offset, 0), "left": (-offset, 0), "above": (0, offset), "below": (0, -offset)}
        if side not in shift:
            raise ValueError("side must be 'right', 'left', 'above' or 'below'")
        elbow = "|-" if side in ("right", "left") else "-|"
        opts = Options()
        if tip:
            opts.set(tip)
        opts.set("rounded corners", rounded)
        opts.update(**options)
        items: list[Any] = [start, Rel(Point(*shift[side])), elbow]
        if label is not None:
            place = {"right": "right", "left": "left", "above": "above", "below": "below"}[side]
            items.append(PathNode(label, Options(place, font=r"\small"), pos=0.25))
        items.append(end)
        return self.add(Path(items, "draw", opts))

    # -- annotations ----------------------------------------------------------------------
    def brace(
        self,
        start: Any,
        end: Any,
        text: Any = None,
        *,
        mirror: bool = False,
        amplitude: float = 5.0,
        raise_: float = 2.0,
        label_style: Any = None,
        **options: Any,
    ) -> Path:
        """A curly brace from ``start`` to ``end``; it opens to the left of
        the direction ``start -> end`` (to the right with ``mirror=True``)."""
        deco = f"brace, amplitude={amplitude:g}pt, raise={raise_:g}pt"
        if mirror:
            deco += ", mirror"
        opts = Options("decorate", decoration=_Raw("{" + deco + "}"))
        opts.update(**options)
        items: list[Any] = [start, "--", end]
        if text is not None:
            side = "auto, swap" if mirror else "auto"
            label_opts = Options("midway", side, outer_sep=pt(amplitude + raise_ + 1.0))
            label_opts.add(label_style)
            items.append(PathNode(text, label_opts))
        return self.add(Path(items, "draw", opts))

    def fit(
        self,
        *items: Any,
        text: Any = "",
        name: str | None = None,
        style: Any = "group",
        padding: float | Length | None = None,
        behind: bool = True,
        title: Any = None,
        title_position: str = "north west",
        **options: Any,
    ) -> Node:
        """A box around nodes/points (``fit`` library), drawn behind them.

        ``title`` writes a small heading just outside the box at
        ``title_position`` (above its top-left corner by default) -- handy
        for "Encoder", "N x" and similar.
        """
        coords = [as_coord(item) for item in items]
        opts = Options(style)
        opts.set("fit", LazyValue(lambda: " ".join(c.tikz() for c in coords)))
        if padding is not None:
            opts.set("inner sep", pt(float(padding)) if isinstance(padding, numbers.Real) else padding)
        opts.update(**options)
        node = Node(text, name=name, options=opts)
        if behind:
            self.add(Scope("on background layer")).add(node)
        else:
            self.add(node)
        if title is not None:
            # a separate node on the main layer, so the fitted content cannot cover it
            self.add(Node(title, node.anchor(title_position),
                          options=Options(anchor=_opposite(title_position), inner_sep="2pt", font=r"\small")))
        return node

    def angle_mark(
        self,
        a: Any,
        vertex: Any,
        b: Any,
        label: Any = None,
        *,
        radius: float = 0.5,
        fill: Any = None,
        label_distance: float | None = None,
        interior: bool = True,
        **options: Any,
    ) -> Path:
        """Mark the angle ``a-vertex-b`` with an arc (numeric points only).

        The arc runs counter-clockwise from ``a`` to ``b``; with
        ``interior=True`` the smaller of the two possible angles is marked.
        """
        v, pa, pb = as_point(vertex), as_point(a), as_point(b)
        start, end = (pa - v).angle(), (pb - v).angle()
        while end <= start:
            end += 360.0
        if interior and end - start > 180.0:
            start, end = end, start + 360.0
        turns = math.floor(start / 360.0) * 360.0  # keep the output readable: 0 <= start < 360
        start, end = start - turns, end - turns
        if fill is not None:
            self.fill(v, v + Point.polar(start, radius), Arc(start, end, radius), "cycle", color=fill)
        path = self.draw(v + Point.polar(start, radius), Arc(start, end, radius), **options)
        if label is not None:
            mid = (start + end) / 2.0
            dist = label_distance if label_distance is not None else radius + 0.28
            self.node(label, at=v + Point.polar(mid, dist))
        return path

    def right_angle_mark(self, a: Any, vertex: Any, b: Any, *, size: float = 0.22, **options: Any) -> Path:
        """The small square marking a right angle at ``vertex``."""
        v, pa, pb = as_point(vertex), as_point(a), as_point(b)
        u = (pa - v).unit() * size
        w = (pb - v).unit() * size
        return self.draw(v + u, v + u + w, v + w, **options)

    def tick_marks(self, a: Any, b: Any, count: int = 1, *, size: float = 0.12, spacing: float = 0.07, **options: Any) -> list[Path]:
        """Equal-length marks (``count`` small strokes) across segment ``ab``."""
        pa, pb = as_point(a), as_point(b)
        d = (pb - pa).unit()
        n = d.perp() * size
        mid = pa.lerp(pb, 0.5)
        paths = []
        for i in range(count):
            c = mid + d * ((i - (count - 1) / 2.0) * spacing)
            paths.append(self.draw(c - n, c + n, **options))
        return paths


def _center(c: Any) -> Any:
    """Nodes stand for their border in paths; arcs need their centre."""
    coord = as_coord(c)
    if isinstance(coord, Node):
        return coord.center
    if isinstance(coord, NodeRef) and coord.anchor_name is None:
        return coord.center
    return coord


def _opposite(position: str) -> str:
    """Anchor that puts a label outside a box at ``position``: ``north west``
    -> ``south west`` (above the corner), ``east`` -> ``west``."""
    words = position.split()
    if "north" in words or "south" in words:
        swap = {"north": "south", "south": "north"}
    else:
        swap = {"east": "west", "west": "east"}
    return " ".join(swap.get(w, w) for w in words)


def _edge_label(label: Any, side: str | None, pos: float, style: Any) -> PathNode:
    opts = Options()
    opts.set("pos", pos)
    if side is None or side == "on":
        opts.add("line label")  # theme style: page-colored background
    elif side == "swap":
        opts.add("auto, swap")
    elif side == "sloped":
        opts.add("sloped, above")
    else:
        opts.add(side)
    opts.add(style)
    if "font" not in opts:
        opts.set("font", r"\small")
    return PathNode(label, opts)


class Scope(Container, Element):
    """``\\begin{scope}[options] ... \\end{scope}``; also a context manager::

        with pic.scope(xshift=4, rotate=30) as s:
            s.circle((0, 0), 1)
    """

    def __init__(self, *args: Any, **options: Any):
        Container.__init__(self)
        self.options = Options(*args, **options)

    def __enter__(self) -> "Scope":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def render(self, ctx: "RenderContext") -> list[str]:
        inner = self.render_children(ctx)
        return [f"\\begin{{scope}}{self.options.render()}", *_indent(inner), r"\end{scope}"]
