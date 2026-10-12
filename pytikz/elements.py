"""The building blocks of a picture: nodes, paths and raw TeX.

Elements are rendered lazily, when the picture is written out. This lets
components (plot axes, for instance) look at everything that was added to them
before deciding on coordinates.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ._fmt import brace, fmt, node_text
from .coords import AnchorMixin, Coord, NodeRef, as_coord
from .options import Options, _Raw
from .render import current, global_name

if TYPE_CHECKING:  # pragma: no cover
    from .render import RenderContext

__all__ = [
    "Element",
    "Raw",
    "Comment",
    "Node",
    "Coordinate",
    "Placement",
    "Path",
    "PathOp",
    "To",
    "Curve",
    "Arc",
    "CircleOp",
    "EllipseOp",
    "Rel",
    "PathNode",
    "PlotOp",
    "GridOp",
    "RawOp",
    "to",
    "curve",
    "arc",
    "circle",
    "ellipse",
    "rel",
    "label",
    "raw_op",
    "CONNECTORS",
]

MAX_LINE = 100


class Element:
    """Base class: something that renders to lines of TikZ code."""

    def render(self, ctx: "RenderContext") -> list[str]:  # pragma: no cover
        raise NotImplementedError


class Raw(Element):
    """Verbatim TeX code."""

    def __init__(self, tex: str):
        self.tex = tex

    def render(self, ctx: "RenderContext") -> list[str]:
        return self.tex.strip("\n").splitlines()


class Comment(Element):
    """A ``%`` comment in the generated code."""

    def __init__(self, text: str):
        self.text = text

    def render(self, ctx: "RenderContext") -> list[str]:
        return [f"% {line}".rstrip() for line in self.text.splitlines()] or ["%"]


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

class Placement:
    """Relative node placement from the ``positioning`` library.

    Renders as ``right=1cm of a`` (``distance`` may be a pair for diagonal
    directions: ``above right=0.5cm and 1cm of a``).
    """

    def __init__(self, direction: str, target: Any, distance: Any = None):
        self.direction = direction.replace("_", " ")
        self.target = as_coord(target)
        self.distance = distance

    def key(self) -> str:
        return self.direction

    def value(self) -> str:
        target = self.target.bare()
        if target is None:
            raise TypeError("relative placement needs a node or a numeric point")
        d = self.distance
        if d is None:
            text = f"of {target}"
        elif isinstance(d, (tuple, list)):
            a, b = (_length(x) for x in d)
            text = f"{a} and {b} of {target}"
        else:
            text = f"{_length(d)} of {target}"
        return brace(text)  # "of 1,2" (a numeric point) needs braces


def _length(value: Any) -> str:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return fmt(value) + "cm"
    return str(value)


class Node(Element, AnchorMixin, Coord):
    """A TikZ node: ``\\node[options] (name) at (x,y) {text};``.

    Nodes are coordinates themselves, so they can be passed wherever a point is
    expected, and expose anchors such as ``node.north`` or ``node.anchor(30)``.
    """

    def __init__(
        self,
        text: Any = "",
        at: Any = None,
        *,
        name: str | None = None,
        options: Any = None,
        placement: Placement | None = None,
    ):
        self.text = text
        self.at = as_coord(at) if at is not None else None
        self._name = name
        self.options = Options(options)
        self.placement = placement
        self._referenced = False
        if name is not None and any(ch in name for ch in ".,;()[]{}"):
            raise ValueError(f"invalid node name {name!r}")

    # -- naming / referencing -------------------------------------------------------
    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        ctx = current()
        if ctx is not None:
            return ctx.auto_name(self)
        # Asked for a name outside of rendering (e.g. to build raw TikZ text):
        # fix a process-wide unique one.
        self._name = global_name("node")
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def _mark_referenced(self) -> None:
        self._referenced = True

    def _ref(self, anchor: Any) -> NodeRef:
        return NodeRef(self, anchor)

    def tikz(self) -> str:
        return f"({self.name})"

    def bare(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Node({self.text!r}, name={self._name!r})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    # -- rendering ------------------------------------------------------------------------
    def _render_options(self) -> Options:
        opts = self.options.copy()
        if self.placement is not None:
            opts.set(self.placement.key(), _Raw(self.placement.value()))
        return opts

    def render(self, ctx: "RenderContext") -> list[str]:
        opts = self._render_options()
        text, multiline = node_text(self.text)
        if multiline and "align" not in opts and "text width" not in opts:
            opts = Options({"align": "center"}).merged(opts)  # first, so styles can override it
        parts = [r"\node" + opts.render()]
        if self._name is not None:
            ctx.claim(self._name, self)
        if self._name is not None or self._referenced:
            parts.append(f"({self.name})")
        if self.at is not None:
            parts.append(f"at {self.at.tikz()}")
        parts.append("{" + text + "};")
        return " ".join(parts).split("\n")


class Coordinate(Node):
    """``\\coordinate (name) at (x,y);`` -- a named point without a shape."""

    def __init__(self, at: Any, name: str | None = None):
        super().__init__("", at, name=name)
        self._referenced = True

    def render(self, ctx: "RenderContext") -> list[str]:
        if self._name is not None:
            ctx.claim(self._name, self)
        opts = self._render_options()
        at = f" at {self.at.tikz()}" if self.at is not None else ""
        return [f"\\coordinate{opts.render()} ({self.name}){at};"]


# ---------------------------------------------------------------------------
# path operations
# ---------------------------------------------------------------------------

#: Strings that are path connectors rather than node names.
CONNECTORS = frozenset({"--", "|-", "-|", "..", "to", "rectangle", "grid", "edge"})


class PathOp:
    """An operation inside a path.

    ``kind`` decides how it joins its neighbours: ``"connector"`` sits between
    two coordinates, ``"coord"`` behaves like a point (a ``--`` is inserted
    between two consecutive ones), ``"arc"`` continues from the current point
    and ends at a new one, ``"op"`` stands on its own (``circle``) and
    ``"label"`` is an inline node.
    """

    kind = "op"

    def tokens(self) -> list[str]:  # pragma: no cover
        raise NotImplementedError


class To(PathOp):
    """``to[options]`` -- e.g. ``To(bend_left=30)`` or ``To(out=0, in_=90)``."""

    kind = "connector"

    def __init__(self, *args: Any, **kwargs: Any):
        self.options = Options(*args, **kwargs)

    def tokens(self) -> list[str]:
        return ["to" + self.options.render()]


class Curve(PathOp):
    """Bezier segment ``.. controls (c1) and (c2) ..``."""

    kind = "connector"

    def __init__(self, control1: Any, control2: Any = None):
        self.c1 = as_coord(control1)
        self.c2 = as_coord(control2) if control2 is not None else None

    def tokens(self) -> list[str]:
        if self.c2 is None:
            return ["..", "controls", self.c1.tikz(), ".."]
        return ["..", "controls", self.c1.tikz(), "and", self.c2.tikz(), ".."]


class Arc(PathOp):
    """``arc[start angle=a, end angle=b, radius=r]`` from the current point."""

    kind = "arc"

    def __init__(
        self,
        start: float,
        end: float,
        radius: Any = 1.0,
        y_radius: Any = None,
        **options: Any,
    ):
        self.options = Options(**options)
        self.options.set("start angle", start)
        self.options.set("end angle", end)
        if y_radius is None:
            self.options.set("radius", radius)
        else:
            self.options.set("x radius", radius)
            self.options.set("y radius", y_radius)

    def tokens(self) -> list[str]:
        return ["arc" + self.options.render()]


class CircleOp(PathOp):
    """``circle[radius=r]`` around the current point."""

    def __init__(self, radius: Any):
        self.radius = radius

    def tokens(self) -> list[str]:
        return [f"circle[{Options(radius=self.radius).inner()}]"]


class EllipseOp(PathOp):
    """``ellipse[x radius=a, y radius=b]`` around the current point."""

    def __init__(self, x_radius: Any, y_radius: Any):
        self.rx, self.ry = x_radius, y_radius

    def tokens(self) -> list[str]:
        return [f"ellipse[{Options(x_radius=self.rx, y_radius=self.ry).inner()}]"]


class Rel(PathOp):
    """A relative coordinate: ``++(dx,dy)`` (moves on) or ``+(dx,dy)``."""

    kind = "coord"

    def __init__(self, offset: Any, update: bool = True):
        self.offset = as_coord(offset)
        self.update = update

    def tokens(self) -> list[str]:
        return [("++" if self.update else "+") + self.offset.tikz()]


class PathNode(PathOp):
    """An inline node such as ``node[midway, above] {label}``."""

    kind = "label"

    def __init__(self, text: Any = "", *args: Any, pos: float | None = None, name: str | None = None, **kwargs: Any):
        self.text = text
        self.options = Options(*args, **kwargs)
        if pos is not None:
            self.options.set("pos", pos)
        self.name = name

    def tokens(self) -> list[str]:
        text, multiline = node_text(self.text)
        opts = self.options.copy()
        if multiline and "align" not in opts and "text width" not in opts:
            opts = Options({"align": "center"}).merged(opts)
        name = f" ({self.name})" if self.name else ""
        return [f"node{opts.render()}{name} {{{text}}}"]


class PlotOp(PathOp):
    """``plot[options] coordinates {...}`` through a list of points."""

    def __init__(self, points: Iterable[Any], *args: Any, connect: bool = False, **kwargs: Any):
        self.points = [as_coord(p) for p in points]
        self.options = Options(*args, **kwargs)
        self.kind = "connector-plot" if connect else "op"

    def tokens(self) -> list[str]:
        coords = [p.tikz() for p in self.points]
        if not coords:
            return []
        head = ["--"] if self.kind == "connector-plot" else []
        coords[0] = "{" + coords[0]
        coords[-1] = coords[-1] + "}"
        return head + ["plot" + self.options.render(), "coordinates"] + coords


class GridOp(PathOp):
    """``grid[step=s]`` between the previous and the next coordinate."""

    kind = "connector"

    def __init__(self, step: Any = 1.0, **options: Any):
        self.options = Options(**options)
        if isinstance(step, (tuple, list)):
            self.options.set("xstep", step[0])
            self.options.set("ystep", step[1])
        else:
            self.options.set("step", step)

    def tokens(self) -> list[str]:
        return ["grid" + self.options.render()]


class Move(PathOp):
    """Start a new subpath: the next point is not joined to the previous one."""

    kind = "move"

    def tokens(self) -> list[str]:
        return []


class RawOp(PathOp):
    """Verbatim path text, e.g. ``RawOp("-- ++(1,0)")``."""

    def __init__(self, text: str, kind: str = "op"):
        self.text = text
        self.kind = kind

    def tokens(self) -> list[str]:
        return [self.text]


# Lower-case helpers read nicely inside ``pic.draw(...)`` calls.
def to(*args: Any, **kwargs: Any) -> To:
    """``to[...]`` connector: ``pic.draw(a, to(bend_left=20), b)``."""
    return To(*args, **kwargs)


def curve(control1: Any, control2: Any = None) -> Curve:
    """Bezier connector: ``pic.draw(a, curve(c1, c2), b)``."""
    return Curve(control1, control2)


def arc(start: float, end: float, radius: Any = 1.0, y_radius: Any = None, **options: Any) -> Arc:
    """Arc from the current point: ``pic.draw((1,0), arc(0, 90, 1))``."""
    return Arc(start, end, radius, y_radius, **options)


def circle(radius: Any) -> CircleOp:
    """Circle around the current point: ``pic.clip((0, 0), circle(2))``."""
    return CircleOp(radius)


def ellipse(x_radius: Any, y_radius: Any) -> EllipseOp:
    """Ellipse around the current point."""
    return EllipseOp(x_radius, y_radius)


def rel(dx: Any, dy: float | None = None, update: bool = True) -> Rel:
    """Relative step: ``pic.draw((0,0), rel(1,0), rel(0,1))``."""
    offset = (dx, dy) if dy is not None else dx
    return Rel(offset, update)


def label(text: Any, *args: Any, pos: float | None = None, **kwargs: Any) -> PathNode:
    """Inline label: ``pic.draw(a, label("$f$", above=True), b)``."""
    return PathNode(text, *args, pos=pos, **kwargs)


def raw_op(text: str) -> RawOp:
    return RawOp(text)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def _classify(item: Any) -> tuple[str, list[str]]:
    if isinstance(item, PathOp):
        return item.kind, item.tokens()
    if isinstance(item, str):
        text = item.strip()
        if text in CONNECTORS:
            return "connector", [text]
        if text in ("cycle", "-- cycle", "--cycle"):
            return "cycle", ["cycle"]
    return "coord", [as_coord(item).tikz()]


def _prepare(item: Any) -> Any:
    """Convert coordinates up front (this also marks referenced nodes)."""
    if isinstance(item, PathOp):
        return item
    if isinstance(item, str) and item.strip() in CONNECTORS | {"cycle", "-- cycle", "--cycle"}:
        return item
    return as_coord(item)


def path_tokens(items: Sequence[Any]) -> list[str]:
    """Turn path items into TikZ tokens, inserting ``--`` between points."""
    tokens: list[str] = []
    last = None
    pending: list[str] = []
    for item in items:
        kind, toks = _classify(item)
        if kind == "label":
            if last == "coord":
                pending.extend(toks)
            else:
                tokens.extend(toks)
            continue
        if kind in ("coord", "cycle"):
            if last == "coord":
                tokens.append("--")
            tokens.extend(pending)
            pending = []
            tokens.extend(toks)
            last = "coord" if kind == "coord" else "cycle"
        elif kind == "connector":
            tokens.extend(toks)
            tokens.extend(pending)
            pending = []
            last = "connector"
        elif kind == "move":
            tokens.extend(pending)
            pending = []
            last = None
        elif kind == "arc":
            # continues from the current point; what follows connects to its end
            tokens.extend(pending)
            pending = []
            tokens.extend(toks)
            last = "coord"
        elif kind == "connector-plot":
            tokens.extend(pending)
            pending = []
            tokens.extend(toks)
            last = "coord"
        else:
            tokens.extend(pending)
            pending = []
            tokens.extend(toks)
            last = "op"
    tokens.extend(pending)
    return tokens


def wrap_tokens(head: str, tokens: Sequence[str], tail: str = ";", width: int = MAX_LINE) -> list[str]:
    """Join tokens into lines no longer than ``width`` (when possible)."""
    lines: list[str] = []
    current_line = head
    for tok in tokens:
        candidate = f"{current_line} {tok}" if current_line else tok
        if len(candidate) > width and current_line.strip() and current_line != head:
            lines.append(current_line)
            current_line = "    " + tok
        else:
            current_line = candidate
    lines.append(current_line + tail)
    return lines


class Path(Element):
    """A path command: ``\\draw``, ``\\fill``, ``\\filldraw``, ``\\path``,
    ``\\clip`` or ``\\shade``."""

    COMMANDS = ("draw", "fill", "filldraw", "path", "clip", "shade", "shadedraw")

    def __init__(self, items: Sequence[Any], command: str = "draw", options: Any = None):
        if command not in self.COMMANDS:
            raise ValueError(f"unknown path command {command!r}")
        self.items = [_prepare(item) for item in items]
        self.command = command
        self.options = Options(options)

    def render(self, ctx: "RenderContext") -> list[str]:
        tokens = path_tokens(self.items)
        head = f"\\{self.command}{self.options.render()}"
        return wrap_tokens(head, tokens)

    def __repr__(self) -> str:
        return f"Path({self.command!r}, {len(self.items)} items)"
