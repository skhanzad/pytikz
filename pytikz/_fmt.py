"""Low-level formatting helpers shared by the whole package.

Everything that turns Python values into TikZ source text eventually goes
through one of these functions, so the generated code is consistent: numbers
are short (no ``1.0000001`` or ``-0``), option values are braced only when
TikZ needs it, and plain text can be escaped for LaTeX.
"""

from __future__ import annotations

import math
import numbers

__all__ = [
    "PRECISION",
    "fmt",
    "tex_escape",
    "smart_escape",
    "number_text",
    "is_braced",
    "brace",
    "split_top_level",
    "node_text",
    "label_text",
    "inline_text",
]

#: Number of decimals written for coordinates, lengths and other numbers.
PRECISION = 4

_OPEN = "{[("
_CLOSE = "}])"


def fmt(value: float, precision: int = PRECISION) -> str:
    """Format a number for TikZ: compact, no exponent and never ``-0``.

    >>> fmt(1.0), fmt(0.333333333), fmt(-0.00001), fmt(2.50)
    ('1', '0.3333', '0', '2.5')
    """
    if isinstance(value, bool):
        raise TypeError("expected a number, got a bool")
    if isinstance(value, numbers.Integral):
        return str(int(value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"cannot write the non-finite number {value!r} to TikZ")
    text = f"{value:.{precision}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", "", "-"):
        text = "0"
    return text


_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(text: str) -> str:
    """Escape LaTeX special characters so ``text`` is typeset literally."""
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(text))


def is_braced(value: str) -> bool:
    """True if ``value`` is a single ``{...}`` group (the outer braces match)."""
    if len(value) < 2 or value[0] != "{" or value[-1] != "}":
        return False
    depth = 0
    for i, ch in enumerate(value):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and i != len(value) - 1:
                return False
    return depth == 0


def _has_top_level(value: str, chars: str) -> bool:
    """Whether ``value`` contains one of ``chars`` outside ``{...}``.

    Only braces protect commas from pgfkeys; parentheses and brackets do not
    (``fit=(a) (1,2)`` must be written ``fit={(a) (1,2)}``).
    """
    depth = 0
    for ch in value:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0 and ch in chars:
            return True
    return False


def brace(value: str) -> str:
    """Wrap an option value in braces when TikZ's key parser needs it."""
    if value == "":
        return "{}"
    if is_braced(value):
        return value
    if (
        value != value.strip()
        or _has_top_level(value, ",=")
        or "[" in value
        or "]" in value
    ):
        return "{" + value + "}"
    return value


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split ``text`` at ``sep`` characters that are not nested in brackets.

    >>> split_top_level("draw, fill={red!20}, label={[x]a,b}")
    ['draw', 'fill={red!20}', 'label={[x]a,b}']
    """
    parts: list[str] = []
    depth = 0
    in_math = False
    current: list[str] = []
    for ch in text:
        if ch == "$" and depth == 0:
            in_math = not in_math
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
        if ch == sep and depth == 0 and not in_math:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def smart_escape(text: str) -> str:
    """Escape ``& % # _ ^`` in the text-mode parts of ``text``.

    These characters are always errors (or swallow the rest of the line) in
    plain node text, so ``"Add & Norm"`` or ``"conv_1"`` just work. Math
    (``$...$``, ``\\(...\\)``), already escaped characters and text containing
    LaTeX environments are left untouched.
    """
    if r"\begin{" in text or r"\verb" in text:
        return text
    out: list[str] = []
    in_math = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            pair = text[i : i + 2]
            if pair in (r"\(", r"\["):
                in_math = True
            elif pair in (r"\)", r"\]"):
                in_math = False
            out.append(pair)
            i += 2
            continue
        if ch == "$":
            in_math = not in_math
        elif not in_math and ch in "&%#_":
            out.append("\\" + ch)
            i += 1
            continue
        elif not in_math and ch == "^":
            out.append(r"\textasciicircum{}")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def number_text(value: float) -> str:
    """A number as node text: integers stay integers, others keep 6
    significant digits (``0.1 + 0.2`` -> ``0.3``, ``1e-5`` -> ``1e-05``)."""
    if isinstance(value, numbers.Integral):
        return str(int(value))
    value = float(value)
    if value.is_integer():
        return str(int(value))
    text = f"{value:.6g}"
    return "$-$" + text[1:] if text.startswith("-") else text


def node_text(text: object) -> tuple[str, bool]:
    """Prepare node contents.

    In plain text, Python newlines become LaTeX line breaks. Text that
    already contains ``\\\\`` or a LaTeX environment keeps its newlines as
    ordinary spaces (so ``aligned`` blocks and the like work). Returns the
    text and whether it has top-level line breaks (TikZ then needs an
    ``align`` option).
    """
    if text is None:
        return "", False
    if isinstance(text, numbers.Number) and not isinstance(text, bool):
        return number_text(text), False
    text = smart_escape(str(text).strip("\n"))
    if "\n" in text:
        if r"\\" in text or r"\begin{" in text:
            # keep the source layout, but never produce a paragraph break
            lines = [line for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)
        else:
            text = r"\\".join(text.split("\n"))
    if r"\begin{" in text:
        return text, False
    return text, r"\\" in text


def label_text(text: object) -> str:
    """User text for component labels: escaping and newlines like node text.

    Use it in nodes that also get ``align=center``.
    """
    return node_text(text)[0]


def inline_text(text: object) -> str:
    """User text for places where a line break would be wrong (table cells)."""
    if text is None:
        return ""
    return smart_escape(" ".join(str(text).split("\n")))
