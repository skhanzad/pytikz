"""Tick placement and tick label formatting for plot axes."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable

__all__ = [
    "nice_step",
    "linear_ticks",
    "log_ticks",
    "format_number",
    "format_tick",
    "format_log_tick",
    "pi_ticks",
    "decimals_for",
]


def nice_step(span: float, target: int = 6) -> float:
    """A 'nice' step (1, 2, 2.5 or 5 times a power of ten) giving about
    ``target`` intervals over ``span``."""
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / max(1, target)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag * 1.0000001:
            return m * mag
    return 10 * mag


def linear_ticks(lo: float, hi: float, target: int = 6, step: float | None = None) -> list[float]:
    """Tick positions inside ``[lo, hi]`` at multiples of a nice step."""
    if hi < lo:
        lo, hi = hi, lo
    step = step or nice_step(hi - lo, target)
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    ticks = [round(k * step, 12) for k in range(first, last + 1)]
    return [0.0 if abs(t) < step * 1e-9 else t for t in ticks]


def log_ticks(lo: float, hi: float) -> list[float]:
    """Powers of ten in ``[lo, hi]`` (positive values only)."""
    if lo <= 0 or hi <= 0:
        raise ValueError("log axes need positive limits")
    a, b = math.ceil(math.log10(lo) - 1e-9), math.floor(math.log10(hi) + 1e-9)
    ticks = [10.0 ** k for k in range(a, b + 1)]
    if len(ticks) < 2:  # less than a decade: add 2x and 5x
        extra = []
        for k in range(a - 1, b + 1):
            for m in (1, 2, 5):
                v = m * 10.0 ** k
                if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                    extra.append(v)
        ticks = sorted(set(extra))
    return ticks


def decimals_for(step: float) -> int:
    """Decimals needed to show multiples of ``step`` exactly."""
    if step <= 0 or not math.isfinite(step):
        return 0
    for d in range(0, 10):
        if abs(round(step * 10**d) - step * 10**d) < 1e-6 * max(1.0, step * 10**d):
            return d
    return 6


def format_number(value: float, decimals: int) -> str:
    """Text for a tick value, using a math minus sign for negatives."""
    text = f"{abs(value):.{decimals}f}"
    if abs(value) >= 10000 and decimals == 0:
        # thin space thousands separator: 12\,500
        text = f"{abs(value):,.0f}".replace(",", r"\,")
    if value < 0 and float(text.replace(r"\,", "")) != 0:
        return "$-$" + text
    return text


def format_tick(value: float, step: float) -> str:
    return format_number(value, decimals_for(step))


def format_log_tick(value: float) -> str:
    k = math.log10(value)
    if abs(k - round(k)) < 1e-9:
        return f"$10^{{{int(round(k))}}}$"
    return format_number(value, decimals_for(value))


def pi_ticks(lo: float, hi: float, denominator: int | None = None, symbol: str = r"\pi") -> list[tuple[float, str]]:
    """Ticks at multiples of pi/denominator with labels like ``$\\frac{\\pi}{2}$``."""
    span = hi - lo
    if denominator is None:
        n = span / math.pi
        denominator = 1 if n > 6 else 2 if n > 1.5 else 4
    step = math.pi / denominator
    first, last = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    out = []
    for k in range(first, last + 1):
        frac = Fraction(k, denominator)
        out.append((k * step, _pi_label(frac, symbol)))
    return out


def _pi_label(frac: Fraction, symbol: str) -> str:
    if frac == 0:
        return "0"
    sign = "-" if frac < 0 else ""
    num, den = abs(frac.numerator), frac.denominator
    top = symbol if num == 1 else f"{num}{symbol}"
    if den == 1:
        return f"${sign}{top}$"
    return f"${sign}\\frac{{{top}}}{{{den}}}$"


Formatter = Callable[[float], str]
