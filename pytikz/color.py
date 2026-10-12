"""Colors.

A :class:`Color` renders as an xcolor expression such as ``Blue``,
``Blue!20`` or ``Blue!60!black``. Colors created from hex strings or RGB
values get a name (``cFF8800`` unless you choose one) and their
``\\definecolor`` line is emitted automatically in every picture that uses
them::

    accent = Color("#E4572E")            # -> \\definecolor{cE4572E}{HTML}{E4572E}
    brand = Color("#0B7285", name="Brand")
    pic.node("x", fill=accent.lighten(0.85), draw=accent)

Strings work too: ``fill="Blue!15"`` uses the theme palette color ``Blue``.
"""

from __future__ import annotations

import colorsys
import re
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable, Iterator, Union

from ._fmt import fmt

__all__ = [
    "Color",
    "ColorLike",
    "rgb",
    "hex_color",
    "blend",
    "XCOLOR_BASE",
    "registered_colors",
    "parse_hex",
    "as_color",
    "active_palette",
    "xmix",
    "color_text",
]

#: Colors that xcolor always knows (no definition needed), with their RGB.
XCOLOR_BASE: dict[str, tuple[float, float, float]] = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "gray": (0.5, 0.5, 0.5),
    "darkgray": (0.25, 0.25, 0.25),
    "lightgray": (0.75, 0.75, 0.75),
    "brown": (0.75, 0.5, 0.25),
    "lime": (0.75, 1.0, 0.0),
    "olive": (0.5, 0.5, 0.0),
    "orange": (1.0, 0.5, 0.0),
    "pink": (1.0, 0.75, 0.75),
    "purple": (0.75, 0.0, 0.25),
    "teal": (0.0, 0.5, 0.5),
    "violet": (0.5, 0.0, 0.5),
}

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")

#: Every named color created in this process: name -> (model, spec, rgb).
_REGISTRY: dict[str, tuple[str, str, tuple[float, float, float]]] = {}


def registered_colors() -> dict[str, tuple[str, str, tuple[float, float, float]]]:
    """A copy of the global registry of named color definitions."""
    return dict(_REGISTRY)


def parse_hex(text: str) -> tuple[float, float, float]:
    """``"#3B82F6"`` or ``"3b82f6"`` or ``"#38f"`` -> RGB floats in [0, 1]."""
    h = text.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise ValueError(f"not a hex color: {text!r}")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _to_hex(rgb_: tuple[float, float, float]) -> str:
    return "".join(f"{max(0, min(255, round(c * 255))):02X}" for c in rgb_)


def _pct(fraction: float) -> str:
    return fmt(max(0.0, min(100.0, fraction * 100.0)), 1)


class Color:
    """A TikZ/xcolor color.

    ``Color("#3B82F6")``, ``Color((0.2, 0.5, 0.9))``, ``Color((51, 128, 230))``
    and ``Color("Blue!20")`` are all valid. Pass ``name=`` to choose the name
    used in the generated ``\\definecolor``.
    """

    __slots__ = ("expr", "_rgb")

    def __init__(self, spec: Any, name: str | None = None):
        if isinstance(spec, Color):
            self.expr, self._rgb = spec.expr, spec._rgb
            if name is not None:
                if spec.rgb is None:
                    raise ValueError(f"cannot name {spec.expr!r}: its RGB value is unknown")
                self._define(name, spec.rgb)
            return
        rgb_value: tuple[float, float, float] | None = None
        if isinstance(spec, str) and spec.strip().startswith("#"):
            rgb_value = parse_hex(spec)
        elif isinstance(spec, (tuple, list)) and len(spec) == 3:
            vals = [float(v) for v in spec]
            if any(v > 1.0 for v in vals):
                vals = [v / 255.0 for v in vals]
            rgb_value = (vals[0], vals[1], vals[2])
        elif isinstance(spec, str):
            self.expr = spec.strip()
            self._rgb = None
            if name is not None:
                value = self.rgb
                if value is None:
                    raise ValueError(f"cannot name {spec!r}: its RGB value is unknown")
                self._define(name, value)
            return
        else:
            raise TypeError(f"cannot interpret {spec!r} as a color")
        self._define(name or ("c" + _to_hex(rgb_value)), rgb_value)

    def _define(self, name: str, rgb_value: tuple[float, float, float]) -> None:
        if not _NAME_RE.match(name):
            raise ValueError(
                f"color names may only contain letters and digits (got {name!r})"
            )
        if name in XCOLOR_BASE:
            raise ValueError(f"{name!r} is a built-in xcolor name; choose another")
        previous = _REGISTRY.get(name)
        if previous is not None and previous[1] != _to_hex(rgb_value):
            warnings.warn(f"color {name!r} is redefined (#{previous[1]} -> #{_to_hex(rgb_value)}); "
                          "every picture uses the latest definition", stacklevel=3)
        _REGISTRY[name] = ("HTML", _to_hex(rgb_value), rgb_value)
        self.expr = name
        self._rgb = rgb_value

    @property
    def rgb(self) -> tuple[float, float, float] | None:
        """RGB floats in [0, 1], or ``None`` if unknown.

        Expressions such as ``Blue!20`` are evaluated against the palette of
        the theme that is active (the one of the picture being rendered, or
        the default theme).
        """
        if self._rgb is not None:
            return self._rgb
        return _resolve_rgb(self.expr)

    # -- constructors -----------------------------------------------------------
    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, name: str | None = None) -> "Color":
        """Hue in degrees, saturation and value in [0, 1]."""
        return cls(colorsys.hsv_to_rgb((h % 360) / 360.0, s, v), name=name)

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, name: str | None = None) -> "Color":
        """Hue in degrees, saturation and lightness in [0, 1]."""
        return cls(colorsys.hls_to_rgb((h % 360) / 360.0, l, s), name=name)

    # -- xcolor mixing (keeps readable expressions) ---------------------------------
    def tint(self, percent: float) -> "Color":
        """xcolor's ``color!percent``: ``percent`` % of this color, rest white."""
        return self._mixed(xmix(self.expr, percent), percent / 100.0, WHITE_RGB)

    def lighten(self, amount: float) -> "Color":
        """Mix with ``amount`` (0..1) of white."""
        return self._mixed(xmix(self.expr, (1 - amount) * 100), 1 - amount, WHITE_RGB)

    def darken(self, amount: float) -> "Color":
        """Mix with ``amount`` (0..1) of black."""
        return self._mixed(xmix(self.expr, (1 - amount) * 100, "black"), 1 - amount, BLACK_RGB)

    def mix(self, other: "ColorLike", amount: float = 0.5) -> "Color":
        """Blend towards ``other``: 0 gives this color, 1 gives ``other``."""
        other = as_color(other)
        target = other.expr
        if "!" in target:
            # xcolor cannot nest expressions on the right: use a defined color
            if other.rgb is None:
                raise ValueError(f"cannot mix with {other.expr!r}: its RGB value is unknown")
            target = other.resolved().expr
        return self._mixed(xmix(self.expr, (1 - amount) * 100, target), 1 - amount, other.rgb)

    def _mixed(self, expr: str, weight: float, other_rgb: Any) -> "Color":
        new = Color.__new__(Color)
        new.expr = expr
        new._rgb = None
        if self._rgb is not None and other_rgb is not None and "!" not in self.expr:
            new._rgb = tuple(  # type: ignore[assignment]
                weight * a + (1 - weight) * b for a, b in zip(self._rgb, other_rgb)
            )
        return new

    # -- python side helpers -------------------------------------------------------------
    @property
    def hex(self) -> str | None:
        """``"#RRGGBB"`` if the RGB value is known."""
        return "#" + _to_hex(self.rgb) if self.rgb is not None else None

    def resolved(self) -> "Color":
        """An equivalent color defined by its RGB value (``cRRGGBB``)."""
        if self.rgb is None:
            raise ValueError(f"RGB value of {self.expr!r} is unknown")
        return Color(self.rgb)

    def luminance(self) -> float:
        """Relative luminance (WCAG); 0 is black, 1 is white."""
        if self.rgb is None:
            return 0.5

        def lin(c: float) -> float:
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = (lin(c) for c in self.rgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def is_dark(self) -> bool:
        """Whether white text reads better than black text on this color."""
        return self.luminance() < 0.32

    def __str__(self) -> str:
        return self.expr

    def __repr__(self) -> str:
        extra = f", {self.hex}" if self.rgb is not None else ""
        return f"Color({self.expr!r}{extra})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Color) and other.expr == self.expr

    def __hash__(self) -> int:
        return hash(self.expr)


ColorLike = Union[Color, str, tuple]

WHITE_RGB = (1.0, 1.0, 1.0)
BLACK_RGB = (0.0, 0.0, 0.0)


def xmix(color: Any, percent: float, other: str | None = None) -> str:
    """``color!percent[!other]`` that stays valid when ``color`` is itself an
    xcolor expression: ``xmix("Blue!40", 50)`` -> ``Blue!40!white!50``."""
    base = str(color)
    if base.count("!") % 2 == 1:  # ends with a percentage: close it with white
        base += "!white"
    text = f"{base}!{fmt(max(0.0, min(100.0, percent)), 1)}"
    return f"{text}!{other}" if other else text


def color_text(value: Any) -> str:
    """TikZ text for a color given as a Color, palette/xcolor string, hex
    string (``"#E4572E"``) or RGB tuple."""
    if isinstance(value, Color):
        return value.expr
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}", text):
            return Color(text).expr
        return text
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return Color(tuple(value)).expr
    if type(value).__module__ == "numpy" and getattr(value, "shape", None) == (3,):
        return Color(tuple(float(v) for v in value)).expr
    return str(value)


def as_color(value: ColorLike) -> Color:
    """Coerce a string, tuple or :class:`Color` into a :class:`Color`."""
    if isinstance(value, Color):
        return value
    return Color(value)


def rgb(r: float, g: float, b: float, name: str | None = None) -> Color:
    """A color from RGB components (floats in [0, 1] or ints in [0, 255])."""
    return Color((r, g, b), name=name)


def hex_color(code: str, name: str | None = None) -> Color:
    """A color from a hex code such as ``"#3B82F6"``."""
    if not code.startswith("#"):
        code = "#" + code
    return Color(code, name=name)


def blend(a: ColorLike, b: ColorLike, t: float) -> Color:
    """RGB interpolation between ``a`` (t=0) and ``b`` (t=1) as a defined color."""
    ca, cb = as_color(a), as_color(b)
    if ca.rgb is None or cb.rgb is None:
        return ca.mix(cb, t)
    return Color(tuple((1 - t) * x + t * y for x, y in zip(ca.rgb, cb.rgb)))


_ACTIVE_PALETTE: ContextVar[dict[str, str] | None] = ContextVar("pytikz_palette", default=None)


@contextmanager
def active_palette(palette: dict[str, str]) -> Iterator[None]:
    """Resolve palette names (``Blue``...) against ``palette`` inside the block."""
    token = _ACTIVE_PALETTE.set(palette)
    try:
        yield
    finally:
        _ACTIVE_PALETTE.reset(token)


def _lookup(name: str) -> tuple[float, float, float] | None:
    palette = _ACTIVE_PALETTE.get()
    if palette is None:
        from .theme import get_theme

        palette = get_theme(None).palette
    if name in palette:
        return parse_hex(palette[name])
    if name in _REGISTRY:
        return _REGISTRY[name][2]
    return XCOLOR_BASE.get(name)


def _resolve_rgb(expr: str) -> tuple[float, float, float] | None:
    """Evaluate simple xcolor expressions like ``Blue!20!black``."""
    parts = expr.replace(" ", "").split("!")
    if not parts or not parts[0]:
        return None
    current = _lookup(parts[0])
    if current is None:
        return None
    i = 1
    while i < len(parts):
        try:
            pct = float(parts[i]) / 100.0
        except ValueError:
            return None
        other_name = parts[i + 1] if i + 1 < len(parts) else "white"
        other = _lookup(other_name) if other_name else WHITE_RGB
        if other is None:
            return None
        current = tuple(pct * a + (1 - pct) * b for a, b in zip(current, other))  # type: ignore[assignment]
        i += 2
    return current


def color_list(values: Iterable[ColorLike]) -> list[Color]:
    return [as_color(v) for v in values]

