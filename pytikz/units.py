"""Lengths with explicit units.

Plain numbers passed as option values get a sensible default unit (see
:mod:`pytikz.options`), but you can always be explicit::

    pic.node("x", minimum_width=cm(2), inner_sep=pt(3))
"""

from __future__ import annotations

import numbers

from ._fmt import fmt

__all__ = ["Length", "cm", "mm", "pt", "em", "ex", "inch", "to_cm"]

#: Conversion factors to centimetres for absolute units.
_TO_CM = {
    "cm": 1.0,
    "mm": 0.1,
    "pt": 2.54 / 72.27,
    "bp": 2.54 / 72.0,
    "in": 2.54,
    "pc": 12 * 2.54 / 72.27,
    # Font relative units, approximated for a 10pt font.
    "em": 10 * 2.54 / 72.27,
    "ex": 4.3 * 2.54 / 72.27,
}


class Length:
    """A number with a unit, e.g. ``Length(2, "cm")`` renders as ``2cm``."""

    __slots__ = ("value", "unit")

    def __init__(self, value: float, unit: str = "cm"):
        if unit not in _TO_CM:
            raise ValueError(f"unknown unit {unit!r}; use one of {sorted(_TO_CM)}")
        self.value = float(value)
        self.unit = unit

    def __str__(self) -> str:
        return fmt(self.value) + self.unit

    def __repr__(self) -> str:
        return f"Length({fmt(self.value)}, {self.unit!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Length) and self.to_cm() == other.to_cm()

    def __hash__(self) -> int:
        return hash(round(self.to_cm(), 9))

    def __mul__(self, k: float) -> "Length":
        return Length(self.value * k, self.unit)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Length":
        return Length(self.value / k, self.unit)

    def __neg__(self) -> "Length":
        return Length(-self.value, self.unit)

    def __add__(self, other: "Length") -> "Length":
        other = _as_length(other)
        if other.unit == self.unit:
            return Length(self.value + other.value, self.unit)
        return Length(self.to_cm() + other.to_cm(), "cm")

    def __sub__(self, other: "Length") -> "Length":
        return self + (-_as_length(other))

    def to_cm(self) -> float:
        """The length in centimetres (font units are approximate)."""
        return self.value * _TO_CM[self.unit]

    def to_pt(self) -> float:
        """The length in TeX points."""
        return self.to_cm() / _TO_CM["pt"]


def _as_length(value: object) -> Length:
    if isinstance(value, Length):
        return value
    if isinstance(value, numbers.Real):
        return Length(value, "cm")
    raise TypeError(f"expected a Length, got {value!r}")


def cm(value: float) -> Length:
    """``value`` centimetres."""
    return Length(value, "cm")


def mm(value: float) -> Length:
    """``value`` millimetres."""
    return Length(value, "mm")


def pt(value: float) -> Length:
    """``value`` TeX points (1/72.27 inch)."""
    return Length(value, "pt")


def em(value: float) -> Length:
    """``value`` em (width of an M in the current font)."""
    return Length(value, "em")


def ex(value: float) -> Length:
    """``value`` ex (height of an x in the current font)."""
    return Length(value, "ex")


def inch(value: float) -> Length:
    """``value`` inches."""
    return Length(value, "in")


def to_cm(value: object, default_unit: str = "cm") -> float:
    """Convert a number, :class:`Length` or string like ``"3pt"`` to cm."""
    if isinstance(value, Length):
        return value.to_cm()
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value) * _TO_CM[default_unit]
    if isinstance(value, str):
        text = value.strip()
        for unit in sorted(_TO_CM, key=len, reverse=True):
            if text.endswith(unit):
                try:
                    return float(text[: -len(unit)]) * _TO_CM[unit]
                except ValueError:
                    break
        try:
            return float(text) * _TO_CM[default_unit]
        except ValueError:
            pass
    raise ValueError(f"cannot interpret {value!r} as a length")
