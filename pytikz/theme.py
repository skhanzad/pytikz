"""Themes: palette, fonts, line weights and the predefined styles.

Every picture has a theme. The theme's palette colors (``Blue``, ``Orange``,
...) can be used by name anywhere, e.g. ``fill="Blue!15"``; only the colors a
picture actually uses are written to the output. Components ask the theme for
semantic colors (``theme.role("input")``) so a whole figure can be restyled by
switching themes::

    pic = Picture(theme="dark")

Built-in themes: ``"modern"`` (default), ``"paper"`` (serif, print friendly),
``"dark"`` (for slides) and ``"mono"`` (grayscale).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .color import xmix
from .options import Options
from .style import Style

__all__ = ["Theme", "THEMES", "get_theme", "register_theme", "ARG"]

#: Color name holding the argument of parameterised styles (``block=<color>``).
ARG = "pytikzarg"


MODERN_PALETTE = {
    "Blue": "3A76D8",
    "Sky": "2E9FD9",
    "Teal": "16A394",
    "Green": "3BA55C",
    "Lime": "8DB33A",
    "Yellow": "EAB531",
    "Orange": "EE8133",
    "Red": "DF4B4B",
    "Pink": "D9508E",
    "Purple": "8459D1",
    "Indigo": "5A5FD6",
    "Brown": "9C6B47",
    "Gray": "6B7280",
    "Ink": "1F2937",
}

DARK_PALETTE = {
    "Blue": "60A5FA",
    "Sky": "38BDF8",
    "Teal": "2DD4BF",
    "Green": "4ADE80",
    "Lime": "A3E635",
    "Yellow": "FACC15",
    "Orange": "FB923C",
    "Red": "F87171",
    "Pink": "F472B6",
    "Purple": "A78BFA",
    "Indigo": "818CF8",
    "Brown": "D6A07A",
    "Gray": "9CA3AF",
    "Ink": "E5E7EB",
    "Paper": "111827",
}

MONO_PALETTE = {
    "Blue": "4B5563",
    "Sky": "6B7280",
    "Teal": "4B5563",
    "Green": "6B7280",
    "Lime": "9CA3AF",
    "Yellow": "9CA3AF",
    "Orange": "374151",
    "Red": "111827",
    "Pink": "6B7280",
    "Purple": "374151",
    "Indigo": "4B5563",
    "Brown": "6B7280",
    "Gray": "6B7280",
    "Ink": "111111",
}

DEFAULT_ROLES = {
    # neural networks
    "input": "Green",
    "hidden": "Blue",
    "output": "Red",
    "bias": "Gray",
    # architecture blocks
    "conv": "Orange",
    "pool": "Red",
    "fc": "Purple",
    "norm": "Yellow",
    "attention": "Orange",
    "ffn": "Sky",
    "embedding": "Pink",
    "activation": "Teal",
    "softmax": "Green",
    "latent": "Purple",
    # general purpose
    "primary": "Blue",
    "secondary": "Orange",
    "accent": "Red",
    "positive": "Green",
    "negative": "Red",
    "muted": "Gray",
    "highlight": "Orange",
}


@dataclass
class Theme:
    """Visual defaults for a picture.

    The ``fill``/``stroke`` helpers produce the theme's standard light fill and
    darker outline for a palette color, e.g. ``theme.fill("Blue")`` ->
    ``"Blue!14"``. In parameterised styles start with :meth:`arg_color` and
    use :data:`ARG` as the color.
    """

    name: str
    palette: dict[str, str]
    font: str = r"\sffamily"
    preamble: tuple[str, ...] = ()
    ink: str = "Ink"
    paper: str | None = None
    line_width: float = 0.6
    arrow_tip: str = "Stealth[length=5.5pt, width=4pt, inset=1.2pt]"
    corner_radius: float = 3.0
    fill_tint: int = 14
    fill_base: str = "white"
    stroke_mix: int = 78
    stroke_base: str = "black"
    muted_text: str = "Ink!65"
    grid_color: str = "Gray!22"
    cycle: tuple[str, ...] = ("Blue", "Orange", "Green", "Red", "Purple", "Teal", "Pink", "Brown")
    roles: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLES))

    # -- color helpers --------------------------------------------------------
    def role(self, name: str) -> str:
        """Palette color for a semantic role (``"input"``, ``"conv"``, ...).

        Unknown names are returned unchanged so palette names work too.
        """
        return self.roles.get(name, name)

    def fill(self, color: Any, tint: float | None = None) -> str:
        """Light background fill for ``color``."""
        color = self.role(str(color))
        tint = self.fill_tint if tint is None else tint
        return xmix(color, tint, None if self.fill_base == "white" else self.fill_base)

    def stroke(self, color: Any) -> str:
        """Outline color for shapes filled with ``fill(color)``."""
        return xmix(self.role(str(color)), self.stroke_mix, self.stroke_base)

    @staticmethod
    def arg_color() -> str:
        """Start of a parameterised style definition: stores the style's
        argument as the color :data:`ARG`, so ``block=Blue!50`` works too."""
        return "/utils/exec={\\colorlet{" + ARG + "}{#1}}"

    def text_on(self, dark_background: bool) -> str:
        """A readable text color on a dark (or light) background."""
        if self.paper is None:
            return "white" if dark_background else self.ink
        return self.ink if dark_background else self.paper

    def color(self, index: int) -> str:
        """The ``index``-th color of the plotting color cycle."""
        return self.cycle[index % len(self.cycle)]

    # -- picture level defaults --------------------------------------------------
    def picture_options(self) -> Options:
        opts = Options()
        opts.set("color", self.ink)
        if self.font:
            # Switching the family at the start of the picture (rather than
            # with ``font=``) lets nodes use ``font=\small`` without losing it.
            opts.set("execute at begin picture", "{" + self.font + "}")
        opts.set("line width", f"{_num(self.line_width)}pt")
        opts.set(">", "{" + self.arrow_tip + "}")
        opts.set("line join", "round")
        return opts

    def styles(self) -> list[Style]:
        """Styles every picture can use; only used ones are emitted."""
        r = _num(self.corner_radius)
        return [
            Style("dot", "circle, fill, inner sep=0pt, minimum size=4.5pt"),
            Style(
                "block",
                f"{self.arg_color()}, draw={self.stroke(ARG)}, fill={self.fill(ARG)}, "
                f"rounded corners={r}pt, align=center, inner sep=5pt, "
                "minimum height=8mm, minimum width=20mm",
                default="Blue",
            ),
            Style(
                "box",
                f"draw, rounded corners={r}pt, align=center, inner sep=5pt, minimum height=7mm",
            ),
            Style(
                "tag",
                f"{self.arg_color()}, draw={self.stroke(ARG)}, fill={self.fill(ARG)}, rounded corners=2pt, "
                "inner sep=2.5pt, font=\\scriptsize",
                default="Gray",
            ),
            Style("muted", f"text={self.muted_text}"),
            Style("line label", f"fill={self.paper or 'white'}, inner sep=1.5pt"),
            Style(
                "op",
                f"circle, draw, fill={self.paper or 'white'}, inner sep=0pt, minimum size=5mm, "
                "font=\\normalsize",
            ),
            Style(
                "group",
                f"{self.arg_color()}, draw={self.stroke(ARG)}, fill={self.fill(ARG, self.fill_tint / 2)}, "
                f"rounded corners={_num(self.corner_radius * 2)}pt, dashed, inner sep=8pt",
                default="Gray",
            ),
        ]

    def with_(self, **changes: Any) -> "Theme":
        """A modified copy, e.g. ``THEMES["modern"].with_(font=r"\\rmfamily")``.

        The palette and roles are copied, so changing them does not affect
        the original theme.
        """
        changes.setdefault("palette", dict(self.palette))
        changes.setdefault("roles", dict(self.roles))
        return replace(self, **changes)


def _num(value: float) -> str:
    from ._fmt import fmt

    return fmt(value, 2)


THEMES: dict[str, Theme] = {}


def register_theme(theme: Theme) -> Theme:
    """Make ``theme`` available by name (``Picture(theme=theme.name)``)."""
    THEMES[theme.name] = theme
    return theme


register_theme(
    Theme(
        name="modern",
        palette=dict(MODERN_PALETTE),
        font=r"\sffamily",
        preamble=(r"\usepackage[scaled=0.95]{helvet}",),
    )
)

register_theme(
    Theme(
        name="paper",
        palette=dict(MODERN_PALETTE),
        font="",
        preamble=(r"\usepackage{lmodern}",),
        line_width=0.5,
        fill_tint=12,
        arrow_tip="Latex[length=5pt, width=3.6pt]",
    )
)

register_theme(
    Theme(
        name="dark",
        palette=dict(DARK_PALETTE),
        font=r"\sffamily",
        preamble=(r"\usepackage[scaled=0.95]{helvet}",),
        paper="Paper",
        fill_tint=22,
        fill_base="Paper",
        stroke_mix=85,
        stroke_base="white",
        muted_text="Ink!70!Paper",
        grid_color="Gray!35!Paper",
    )
)

register_theme(
    Theme(
        name="mono",
        palette=dict(MONO_PALETTE),
        font="",
        preamble=(r"\usepackage{lmodern}",),
        fill_tint=12,
        stroke_mix=100,
        cycle=("Ink", "Gray", "Ink!60", "Gray!60"),
        line_width=0.5,
    )
)


def get_theme(theme: str | Theme | None) -> Theme:
    """Look up a theme by name (``None`` gives the default ``"modern"``)."""
    if theme is None:
        return THEMES["modern"]
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(f"unknown theme {theme!r}; available: {sorted(THEMES)}") from None
