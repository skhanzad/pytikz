"""Coordinates: numeric points computed in Python and symbolic TikZ references.

Two worlds meet here:

* :class:`Point` is a plain 2D vector. Arithmetic happens in Python and the
  result is written as ``(x,y)``. Use it for anything you can compute
  (geometry, layouts, data).
* Node anchors such as ``node.north`` are only known to TikZ. Combining them
  with arithmetic builds a ``calc`` expression instead, so
  ``a.east + (0.5, 0)`` is written as ``($(a.east)+(0.5,0)$)``.

Anywhere a coordinate is expected you may pass a ``Point``, a ``(x, y)`` tuple,
a node, a node name such as ``"a.north"``, or a complex number.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Callable, Iterator, Union

from ._fmt import fmt
from .units import Length, to_cm

__all__ = [
    "Coord",
    "Point",
    "Polar",
    "Point3",
    "NodeRef",
    "Calc",
    "Partway",
    "Perpendicular",
    "RawCoord",
    "LazyCoord",
    "AnchorMixin",
    "as_coord",
    "as_point",
    "is_numeric_coord",
    "between",
    "midpoint",
    "towards",
    "project",
    "xy",
    "ORIGIN",
]


class Coord:
    """Base class of everything that renders as a TikZ coordinate."""

    __slots__ = ()

    def tikz(self) -> str:
        """The coordinate in TikZ syntax, including the parentheses."""
        raise NotImplementedError

    def bare(self) -> str | None:
        """Text usable inside ``(a |- b)``; ``None`` if not possible."""
        return None

    def __str__(self) -> str:
        return self.tikz()

    # -- symbolic arithmetic (overridden by Point for numeric work) ----------
    def __add__(self, other: Any) -> "Coord":
        return Calc.combine(self, 1.0, other)

    def __radd__(self, other: Any) -> "Coord":
        return Calc.combine(other, 1.0, self)

    def __sub__(self, other: Any) -> "Coord":
        return Calc.combine(self, -1.0, other)

    def __rsub__(self, other: Any) -> "Coord":
        return Calc.combine(other, -1.0, self)

    def __mul__(self, k: float) -> "Coord":
        if not _is_number(k):
            return NotImplemented
        return Calc([(float(k), self)])

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Coord":
        if not _is_number(k):
            return NotImplemented
        return self * (1.0 / k)

    def __neg__(self) -> "Coord":
        return self * -1.0

    # -- geometric helpers ----------------------------------------------------
    def to(self, other: Any, t: float = 0.5) -> "Coord":
        """The point a fraction ``t`` of the way from here to ``other``."""
        return between(self, other, t)

    def towards(self, other: Any, distance: float | Length | str) -> "Coord":
        """The point ``distance`` (cm by default) from here towards ``other``."""
        return towards(self, other, distance)

    def shift(self, dx: float = 0.0, dy: float = 0.0) -> "Coord":
        """This coordinate moved by ``(dx, dy)``."""
        return self + Point(dx, dy)

    def xy(self, other: Any) -> "Coord":
        """The point with this coordinate's x and ``other``'s y (TikZ ``|-``)."""
        return xy(self, other)


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


class Point(Coord):
    """A 2D point/vector with numeric arithmetic.

    >>> Point(1, 2) + (3, 4)
    Point(4, 6)
    >>> (Point(1, 0).rotate(90)).round(6)
    Point(0, 1)
    """

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    # -- construction ---------------------------------------------------------
    @classmethod
    def polar(cls, angle: float, radius: float = 1.0) -> "Point":
        """The point at ``angle`` degrees and distance ``radius`` from the origin."""
        a = math.radians(angle)
        return cls(radius * math.cos(a), radius * math.sin(a))

    # -- sequence protocol so ``x, y = p`` works ---------------------------------
    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y)[index]

    def __eq__(self, other: object) -> bool:
        other_point = _maybe_point(other)
        return other_point is not None and (self.x, self.y) == (other_point.x, other_point.y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"Point({fmt(self.x, 6)}, {fmt(self.y, 6)})"

    # -- rendering ----------------------------------------------------------------
    def tikz(self) -> str:
        return f"({fmt(self.x)},{fmt(self.y)})"

    def bare(self) -> str:
        return f"{fmt(self.x)},{fmt(self.y)}"

    # -- arithmetic -----------------------------------------------------------------
    def __add__(self, other: Any) -> Coord:
        p = _maybe_point(other)
        if p is not None:
            return Point(self.x + p.x, self.y + p.y)
        return Coord.__add__(self, other)

    def __radd__(self, other: Any) -> Coord:
        p = _maybe_point(other)
        if p is not None:
            return Point(p.x + self.x, p.y + self.y)
        return Coord.__radd__(self, other)

    def __sub__(self, other: Any) -> Coord:
        p = _maybe_point(other)
        if p is not None:
            return Point(self.x - p.x, self.y - p.y)
        return Coord.__sub__(self, other)

    def __rsub__(self, other: Any) -> Coord:
        p = _maybe_point(other)
        if p is not None:
            return Point(p.x - self.x, p.y - self.y)
        return Coord.__rsub__(self, other)

    def __mul__(self, k: float) -> "Point":
        if not _is_number(k):
            return NotImplemented
        return Point(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Point":
        if not _is_number(k):
            return NotImplemented
        return Point(self.x / k, self.y / k)

    def __neg__(self) -> "Point":
        return Point(-self.x, -self.y)

    def __abs__(self) -> float:
        return self.norm()

    # -- vector helpers -----------------------------------------------------------
    def norm(self) -> float:
        """Euclidean length."""
        return math.hypot(self.x, self.y)

    length = norm

    def unit(self) -> "Point":
        """This vector scaled to length 1."""
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("the zero vector has no direction")
        return Point(self.x / n, self.y / n)

    def dot(self, other: Any) -> float:
        p = as_point(other)
        return self.x * p.x + self.y * p.y

    def cross(self, other: Any) -> float:
        """z component of the 3D cross product (signed parallelogram area)."""
        p = as_point(other)
        return self.x * p.y - self.y * p.x

    def angle(self) -> float:
        """Direction in degrees, in ``(-180, 180]``."""
        return math.degrees(math.atan2(self.y, self.x))

    def angle_to(self, other: Any) -> float:
        """Direction in degrees from this point to ``other``."""
        return (as_point(other) - self).angle()

    def distance(self, other: Any) -> float:
        return (as_point(other) - self).norm()

    def rotate(self, angle: float, about: Any = None) -> "Point":
        """Rotate counter-clockwise by ``angle`` degrees around ``about``."""
        c = as_point(about) if about is not None else Point(0, 0)
        a = math.radians(angle)
        dx, dy = self.x - c.x, self.y - c.y
        return Point(
            c.x + dx * math.cos(a) - dy * math.sin(a),
            c.y + dx * math.sin(a) + dy * math.cos(a),
        )

    def perp(self) -> "Point":
        """This vector rotated by +90 degrees."""
        return Point(-self.y, self.x)

    def lerp(self, other: Any, t: float) -> "Point":
        """Linear interpolation: ``self`` at t=0, ``other`` at t=1."""
        p = as_point(other)
        return Point(self.x + (p.x - self.x) * t, self.y + (p.y - self.y) * t)

    def scale(self, sx: float, sy: float | None = None) -> "Point":
        return Point(self.x * sx, self.y * (sx if sy is None else sy))

    def round(self, ndigits: int = 4) -> "Point":
        return Point(round(self.x, ndigits) + 0.0, round(self.y, ndigits) + 0.0)

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


ORIGIN = Point(0, 0)


class Polar(Point):
    """A point given in polar form; rendered as ``(angle:radius)``."""

    __slots__ = ("angle_deg", "radius")

    def __init__(self, angle: float, radius: float | Length = 1.0):
        r = to_cm(radius) if isinstance(radius, (Length, str)) else float(radius)
        self.angle_deg = float(angle)
        self.radius = radius
        a = math.radians(angle)
        super().__init__(r * math.cos(a), r * math.sin(a))

    def tikz(self) -> str:
        r = self.radius if isinstance(self.radius, (Length, str)) else fmt(self.radius)
        return f"({fmt(self.angle_deg)}:{r})"

    def bare(self) -> str:
        return self.tikz()[1:-1]

    def __repr__(self) -> str:
        return f"Polar({fmt(self.angle_deg)}, {self.radius!r})"


class Point3(Coord):
    """A 3D TikZ coordinate ``(x,y,z)`` (projected by TikZ's own z-vector)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def tikz(self) -> str:
        return f"({fmt(self.x)},{fmt(self.y)},{fmt(self.z)})"

    def bare(self) -> str:
        return self.tikz()[1:-1]

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"Point3({fmt(self.x)}, {fmt(self.y)}, {fmt(self.z)})"


# ---------------------------------------------------------------------------
# Node references
# ---------------------------------------------------------------------------

def _anchor_text(anchor: Any) -> str:
    if _is_number(anchor):
        return fmt(anchor)
    return str(anchor).replace("_", " ")


class AnchorMixin:
    """Anchor properties shared by nodes and node references."""

    __slots__ = ()

    def _ref(self, anchor: Any) -> "NodeRef":  # pragma: no cover - overridden
        raise NotImplementedError

    def anchor(self, anchor: Any) -> "NodeRef":
        """Reference an anchor by name (``"north east"``) or border angle (``30``)."""
        return self._ref(_anchor_text(anchor))

    @property
    def north(self) -> "NodeRef":
        return self._ref("north")

    @property
    def south(self) -> "NodeRef":
        return self._ref("south")

    @property
    def east(self) -> "NodeRef":
        return self._ref("east")

    @property
    def west(self) -> "NodeRef":
        return self._ref("west")

    @property
    def north_east(self) -> "NodeRef":
        return self._ref("north east")

    @property
    def north_west(self) -> "NodeRef":
        return self._ref("north west")

    @property
    def south_east(self) -> "NodeRef":
        return self._ref("south east")

    @property
    def south_west(self) -> "NodeRef":
        return self._ref("south west")

    @property
    def center(self) -> "NodeRef":
        return self._ref("center")

    @property
    def base(self) -> "NodeRef":
        return self._ref("base")

    @property
    def mid(self) -> "NodeRef":
        return self._ref("mid")

    @property
    def text_anchor(self) -> "NodeRef":
        return self._ref("text")


class NodeRef(AnchorMixin, Coord):
    """A reference to a named node (or one of its anchors)."""

    __slots__ = ("target", "anchor_name")

    def __init__(self, target: Any, anchor: str | None = None):
        # ``target`` is either a node name or a Node object whose (possibly
        # automatic) name is looked up when the picture is rendered.
        if hasattr(target, "_mark_referenced"):
            target._mark_referenced()
        elif not isinstance(target, str):
            raise TypeError(f"cannot reference {target!r} as a node")
        self.target = target
        self.anchor_name = anchor

    @property
    def name(self) -> str:
        if isinstance(self.target, str):
            return self.target
        return self.target.name

    def _ref(self, anchor: Any) -> "NodeRef":
        return NodeRef(self.target, anchor)

    def bare(self) -> str:
        return f"{self.name}.{self.anchor_name}" if self.anchor_name else self.name

    def tikz(self) -> str:
        return f"({self.bare()})"

    def __repr__(self) -> str:
        return f"NodeRef({self.bare()!r})"


# ---------------------------------------------------------------------------
# calc expressions
# ---------------------------------------------------------------------------

class Calc(Coord):
    """A linear combination of coordinates, computed by TikZ's calc library."""

    __slots__ = ("terms",)

    def __init__(self, terms: list[tuple[float, Coord]]):
        self.terms = terms

    @staticmethod
    def combine(left: Any, sign: float, right: Any) -> Coord:
        terms = _terms(as_coord(left)) + [(sign * k, c) for k, c in _terms(as_coord(right))]
        return Calc(terms).simplified()

    def simplified(self) -> Coord:
        """Fold numeric terms into one offset; return a Point if fully numeric."""
        offset = Point(0, 0)
        symbolic: list[tuple[float, Coord]] = []
        for k, c in self.terms:
            if isinstance(c, Point):
                offset = offset + c * k
            else:
                symbolic.append((k, c))
        if not symbolic:
            return offset
        if offset.x != 0 or offset.y != 0:
            symbolic.append((1.0, Point(offset.x, offset.y)))
        return Calc(symbolic)

    def __mul__(self, k: float) -> Coord:
        if not _is_number(k):
            return NotImplemented
        return Calc([(c * k, t) for c, t in self.terms])

    __rmul__ = __mul__

    def tikz(self) -> str:
        if len(self.terms) == 1 and self.terms[0][0] == 1.0:
            return self.terms[0][1].tikz()
        parts = []
        for i, (k, c) in enumerate(self.terms):
            text = c.tikz()
            if i == 0:
                if k == 1.0:
                    parts.append(text)
                else:
                    parts.append(f"{fmt(k)}*{text}")
            elif k == 1.0:
                parts.append(f"+{text}")
            elif k == -1.0:
                parts.append(f"-{text}")
            elif k < 0:
                parts.append(f"-{fmt(-k)}*{text}")
            else:
                parts.append(f"+{fmt(k)}*{text}")
        return "($" + "".join(parts) + "$)"

    def __repr__(self) -> str:
        return f"Calc({self.tikz()!r})"


def _terms(c: Coord) -> list[tuple[float, Coord]]:
    if isinstance(c, Calc):
        return list(c.terms)
    return [(1.0, c)]


class Partway(Coord):
    """``($(a)!t!(b)$)``: the point a fraction ``t`` from ``a`` to ``b``.

    ``t`` may also be a distance such as ``"1cm"`` and ``a`` may carry an
    angle, which rotates ``b`` around ``a`` (TikZ's ``!t!angle:b`` syntax).
    """

    __slots__ = ("a", "b", "t", "angle")

    def __init__(self, a: Any, b: Any, t: float | Length | str, angle: float | None = None):
        self.a, self.b, self.t, self.angle = as_coord(a), as_coord(b), t, angle

    def tikz(self) -> str:
        t = fmt(self.t) if _is_number(self.t) else str(self.t)
        rot = f"{fmt(self.angle)}:" if self.angle is not None else ""
        return f"(${self.a.tikz()}!{t}!{rot}{self.b.tikz()}$)"

    def __repr__(self) -> str:
        return f"Partway({self.tikz()!r})"


class Projection(Coord):
    """``($(a)!(p)!(b)$)``: the orthogonal projection of ``p`` on line ``ab``."""

    __slots__ = ("p", "a", "b")

    def __init__(self, p: Any, a: Any, b: Any):
        self.p, self.a, self.b = as_coord(p), as_coord(a), as_coord(b)

    def tikz(self) -> str:
        return f"(${self.a.tikz()}!{self.p.tikz()}!{self.b.tikz()}$)"


class Perpendicular(Coord):
    """``(a |- b)``: x from ``a`` and y from ``b`` (``|-`` in TikZ)."""

    __slots__ = ("a", "b", "op")

    def __init__(self, a: Any, b: Any, op: str = "|-"):
        self.a, self.b, self.op = as_coord(a), as_coord(b), op
        for c in (self.a, self.b):
            if isinstance(c, (Calc, Partway, Projection)):
                raise TypeError("|- needs node references or plain points, not calc expressions")

    def tikz(self) -> str:
        a, b = self.a.bare(), self.b.bare()
        if a is None or b is None:
            raise TypeError("|- needs node references or plain points")
        return f"({a} {self.op} {b})"


class RawCoord(Coord):
    """A coordinate given verbatim, e.g. ``RawCoord("(a.north)")``."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        text = text.strip()
        if not text.startswith("("):
            text = f"({text})"
        self.text = text

    def tikz(self) -> str:
        return self.text

    def bare(self) -> str | None:
        inner = self.text[1:-1]
        return None if "$" in inner else inner

    def __repr__(self) -> str:
        return f"RawCoord({self.text!r})"


class LazyCoord(Coord):
    """A coordinate computed only when the picture is rendered.

    Plot axes use this so that data coordinates can be mapped to the canvas
    after all data (and hence the axis limits) is known.
    """

    __slots__ = ("func",)

    def __init__(self, func: Callable[[], Any]):
        self.func = func

    def resolve(self) -> Coord:
        return as_coord(self.func())

    def tikz(self) -> str:
        return self.resolve().tikz()

    def bare(self) -> str | None:
        return self.resolve().bare()

    def __add__(self, other: Any) -> Coord:
        return LazyCoord(lambda: self.resolve() + other)

    def __radd__(self, other: Any) -> Coord:
        return LazyCoord(lambda: other + self.resolve())

    def __sub__(self, other: Any) -> Coord:
        return LazyCoord(lambda: self.resolve() - other)

    def __rsub__(self, other: Any) -> Coord:
        return LazyCoord(lambda: other - self.resolve())

    def __mul__(self, k: float) -> Coord:
        return LazyCoord(lambda: self.resolve() * k)

    __rmul__ = __mul__


# ---------------------------------------------------------------------------
# conversion helpers
# ---------------------------------------------------------------------------

CoordLike = Union[Coord, tuple, list, complex, str]


def _maybe_point(value: Any) -> Point | None:
    if isinstance(value, Point):
        return value
    if isinstance(value, complex):
        return Point(value.real, value.imag)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(_is_number(v) for v in value):
        return Point(value[0], value[1])
    if type(value).__module__ == "numpy" and getattr(value, "shape", None) == (2,):
        return Point(float(value[0]), float(value[1]))
    return None


def as_coord(value: Any) -> Coord:
    """Convert any coordinate-like value to a :class:`Coord`."""
    if isinstance(value, Coord):
        if hasattr(value, "_mark_referenced"):
            value._mark_referenced()
        return value
    p = _maybe_point(value)
    if p is not None:
        return p
    if isinstance(value, (tuple, list)) and len(value) == 3 and all(_is_number(v) for v in value):
        return Point3(*value)
    if isinstance(value, str):
        return RawCoord(value) if value.strip().startswith("(") else NodeRef(value.strip())
    if type(value).__module__ == "numpy" and getattr(value, "shape", None) == (3,):
        return Point3(*map(float, value))
    raise TypeError(f"cannot use {value!r} as a coordinate")


def as_point(value: Any) -> Point:
    """Convert to a numeric :class:`Point`; raises for symbolic coordinates."""
    if isinstance(value, LazyCoord):
        value = value.resolve()
    p = _maybe_point(value)
    if p is None:
        raise TypeError(
            f"{value!r} is not a numeric point; node positions are only known to TikZ"
        )
    return p


def is_numeric_coord(value: Any) -> bool:
    """True if ``value`` is a coordinate whose position Python knows."""
    return _maybe_point(value) is not None


def between(a: Any, b: Any, t: float = 0.5) -> Coord:
    """The point a fraction ``t`` of the way from ``a`` to ``b``."""
    pa, pb = _maybe_point(a), _maybe_point(b)
    if pa is not None and pb is not None and _is_number(t):
        return pa.lerp(pb, t)
    return Partway(a, b, t)


def midpoint(a: Any, b: Any) -> Coord:
    """The midpoint of ``a`` and ``b``."""
    return between(a, b, 0.5)


def towards(a: Any, b: Any, distance: float | Length | str) -> Coord:
    """The point at ``distance`` from ``a`` in the direction of ``b``."""
    pa, pb = _maybe_point(a), _maybe_point(b)
    if pa is not None and pb is not None and not isinstance(distance, str):
        d = to_cm(distance)
        return pa + (pb - pa).unit() * d
    if _is_number(distance):
        distance = Length(distance, "cm")
    return Partway(a, b, str(distance))


def project(p: Any, a: Any, b: Any) -> Coord:
    """Orthogonal projection of ``p`` onto the line through ``a`` and ``b``."""
    pp, pa, pb = _maybe_point(p), _maybe_point(a), _maybe_point(b)
    if pp is not None and pa is not None and pb is not None:
        d = pb - pa
        t = (pp - pa).dot(d) / d.dot(d)
        return pa + d * t
    return Projection(p, a, b)


def xy(x_from: Any, y_from: Any) -> Coord:
    """The point with the x of ``x_from`` and the y of ``y_from``."""
    pa, pb = _maybe_point(x_from), _maybe_point(y_from)
    if pa is not None and pb is not None:
        return Point(pa.x, pb.y)
    return Perpendicular(x_from, y_from, "|-")
