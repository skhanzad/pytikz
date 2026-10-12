"""TikZ option lists (the ``[draw, fill=blue!20, rounded corners=3pt]`` part).

Options can be given in three interchangeable ways::

    Options("draw, thick")                     # raw TikZ text
    Options(draw=True, line_width=1.2)         # Python keywords
    Options({"shorten >": "2pt"})              # exact TikZ keys

Keyword names are translated by replacing ``_`` with a space
(``rounded_corners`` -> ``rounded corners``); a trailing underscore is dropped
so Python keywords work (``in_=90`` -> ``in=90``). ``True`` gives a bare key,
``False``/``None`` removes it. Plain numbers get a default unit that depends on
the key: centimetres for sizes and shifts, points for line widths, separations
and corner radii.
"""

from __future__ import annotations

import numbers
from typing import Any, Iterable, Iterator

from ._fmt import brace, fmt, split_top_level
from .units import Length

__all__ = ["Options", "normalize_key", "format_value", "CM_KEYS", "PT_KEYS"]

#: Keys whose plain-number values are centimetres.
CM_KEYS = frozenset(
    {
        "minimum width",
        "minimum height",
        "minimum size",
        "text width",
        "node distance",
        "xshift",
        "yshift",
        "radius",
        "x radius",
        "y radius",
        "x",
        "y",
        "z",
        "level distance",
        "sibling distance",
        "step",
    }
)

#: Keys whose plain-number values are TeX points.
PT_KEYS = frozenset(
    {
        "line width",
        "rounded corners",
        "inner sep",
        "outer sep",
        "inner xsep",
        "inner ysep",
        "outer xsep",
        "outer ysep",
        "shorten <",
        "shorten >",
        "double distance",
        "label distance",
        "pin distance",
        "text height",
        "text depth",
        "amplitude",
        "segment length",
        "raise",
        "inner frame sep",
        "outer frame sep",
    }
)

#: Keys whose values are colors (hex strings and RGB tuples are converted).
COLOR_KEYS = frozenset(
    {
        "color",
        "fill",
        "draw",
        "text",
        "top color",
        "bottom color",
        "left color",
        "right color",
        "middle color",
        "inner color",
        "outer color",
        "ball color",
        "pattern color",
    }
)

#: Python-friendly spellings of keys that cannot be written as identifiers.
KEY_ALIASES = {
    "shorten_start": "shorten <",
    "shorten_end": "shorten >",
    "text_color": "text",
}

_UNSET = object()


def normalize_key(key: str) -> str:
    """Translate a Python keyword argument name to a TikZ key."""
    if key in KEY_ALIASES:
        return KEY_ALIASES[key]
    return key.rstrip("_").replace("_", " ")


def _default_unit(key: str) -> str:
    if key in CM_KEYS:
        return "cm"
    if key in PT_KEYS:
        return "pt"
    return ""


def format_value(key: str, value: Any) -> str:
    """Render a single option value to TikZ text."""
    # Local imports avoid import cycles; these modules import options.
    from .color import Color
    from .coords import Coord

    if key in COLOR_KEYS and not isinstance(value, (bool, numbers.Real)):
        from .color import color_text

        return brace(color_text(value))
    if isinstance(value, Options):
        return "{" + value.inner() + "}"
    if isinstance(value, dict):
        return "{" + Options(value).inner() + "}"
    if isinstance(value, Color):
        return brace(value.expr)
    if isinstance(value, Length):
        return str(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return fmt(value) + _default_unit(key)
    if isinstance(value, Coord):
        return "{" + value.tikz() + "}"
    if isinstance(value, tuple) and key in ("shift", "at"):
        from .coords import as_coord

        return "{" + as_coord(value).tikz() + "}"
    if isinstance(value, (list, tuple)):
        return brace(" ".join(str(v) for v in value))
    return brace(str(value))


class Options:
    """An ordered TikZ option list. Later settings of a key replace earlier ones."""

    __slots__ = ("_items",)

    def __init__(self, *args: Any, **kwargs: Any):
        self._items: list[tuple[str, Any]] = []
        for arg in args:
            self.add(arg)
        for key, value in kwargs.items():
            self.set(normalize_key(key), value)

    # -- building -----------------------------------------------------------
    def add(self, arg: Any) -> "Options":
        """Add raw text, a dict of exact keys, another :class:`Options`, a
        :class:`~pytikz.Style` or a list of any of these."""
        from .style import Style

        if arg is None or arg is False:
            return self
        if isinstance(arg, Options):
            for key, value in arg._items:
                self._set_exact(key, value)
        elif isinstance(arg, Style):
            self._set_exact(arg.name, True)
        elif isinstance(arg, dict):
            for key, value in arg.items():
                self._set_exact(str(key), value)
        elif isinstance(arg, str):
            for part in split_top_level(arg):
                key, eq, value = _split_key_value(part)
                self._set_exact(key, _Raw(value) if eq else True)
        elif isinstance(arg, (list, tuple, set, frozenset)):
            for item in arg:
                self.add(item)
        else:
            raise TypeError(f"cannot use {arg!r} as a TikZ option")
        return self

    def set(self, key: str, value: Any = True) -> "Options":
        """Set an exact TikZ ``key`` (no underscore translation)."""
        return self._set_exact(key, value)

    def update(self, **kwargs: Any) -> "Options":
        """Set keys from Python keyword arguments."""
        for key, value in kwargs.items():
            self.set(normalize_key(key), value)
        return self

    def setdefault(self, key: str, value: Any) -> "Options":
        """Set ``key`` only if it is not present yet."""
        if key not in self:
            self.set(key, value)
        return self

    def _set_exact(self, key: str, value: Any) -> "Options":
        key = key.strip()
        if type(value).__module__ == "numpy" and type(value).__name__ in ("bool_", "bool"):
            value = bool(value)
        if key == "style" and not isinstance(value, (_Raw, Options, dict)):
            # ``style="thick, box"`` adds bare styles rather than TikZ's style= key.
            if value is not None and value is not False:
                self.add(value)
            return self
        if key == "arrow":
            if value is True:
                value = "->"
            if value is None or value is False:
                return self
            key, value = str(value), True
        for i, (existing, _) in enumerate(self._items):
            if existing == key:
                if value is None or value is False:
                    del self._items[i]
                else:
                    self._items[i] = (key, value)
                return self
        if value is not None and value is not False:
            self._items.append((key, value))
        return self

    def remove(self, key: str) -> "Options":
        self._items = [(k, v) for k, v in self._items if k != key]
        return self

    def copy(self) -> "Options":
        new = Options()
        new._items = list(self._items)
        return new

    def merged(self, *others: Any, **kwargs: Any) -> "Options":
        """A copy with ``others`` and ``kwargs`` applied on top."""
        new = self.copy()
        for other in others:
            new.add(other)
        new.update(**kwargs)
        return new

    # -- queries ------------------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return any(k == key for k, _ in self._items)

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self._items:
            if k == key:
                return v.text if isinstance(v, _Raw) else v
        return default

    def keys(self) -> list[str]:
        return [k for k, _ in self._items]

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Options) and self.inner() == other.inner()

    # -- rendering ----------------------------------------------------------
    def inner(self) -> str:
        """The comma separated option text without the square brackets."""
        parts = []
        for key, value in self._items:
            if value is True:
                parts.append(key)
            elif isinstance(value, _Raw):
                parts.append(f"{key}={value.text}")
            else:
                parts.append(f"{key}={format_value(key, value)}")
        return ", ".join(parts)

    def render(self) -> str:
        """``[...]`` or an empty string when there are no options."""
        text = self.inner()
        return f"[{text}]" if text else ""

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return f"Options({self.inner()!r})"


class _Raw:
    """An option value that is already TikZ text."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Raw) and other.text == self.text


def _split_key_value(part: str) -> tuple[str, bool, str]:
    depth = 0
    for i, ch in enumerate(part):
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "=" and depth == 0:
            return part[:i].strip(), True, part[i + 1 :].strip()
    return part.strip(), False, ""


def as_options(value: Any) -> Options:
    """Coerce ``value`` (None, str, dict, list or Options) to :class:`Options`."""
    if isinstance(value, Options):
        return value.copy()
    return Options(value)


def iter_options(values: Iterable[Any]) -> Options:
    opts = Options()
    for value in values:
        opts.add(value)
    return opts
