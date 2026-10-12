"""Colormaps for heatmaps, surfaces and contour plots.

``get_colormap("viridis")(0.3)`` returns a :class:`~pytikz.Color`. Values are
quantised to 256 levels, so a heatmap needs at most 256 color definitions.
Build your own with ``Colormap.from_colors("white", "Blue", "Ink")``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .color import Color, as_color, parse_hex

__all__ = ["Colormap", "COLORMAPS", "get_colormap"]

_DATA: dict[str, list[str]] = {
    # perceptually uniform (matplotlib / viridisLite samples)
    "viridis": ["440154", "482475", "414487", "355F8D", "2A788E", "21918C",
                "22A884", "44BF70", "7AD151", "BDDF26", "FDE725"],
    "magma": ["000004", "140E36", "3B0F70", "641A80", "8C2981", "B73779",
              "DE4968", "F7705C", "FE9F6D", "FECF92", "FCFDBF"],
    "inferno": ["000004", "160B39", "420A68", "6A176E", "932667", "BC3754",
                "DD513A", "F37819", "FCA50A", "F6D746", "FCFFA4"],
    "plasma": ["0D0887", "41049D", "6A00A8", "8F0DA4", "B12A90", "CC4778",
               "E16462", "F2844B", "FCA636", "FCCE25", "F0F921"],
    "cividis": ["00204D", "00336F", "39486B", "575C6D", "707173", "8A8779",
                "A69D75", "C4B56C", "E4CF5B", "FFEA46"],
    # ColorBrewer sequential
    "blues": ["F7FBFF", "DEEBF7", "C6DBEF", "9ECAE1", "6BAED6", "4292C6", "2171B5", "08519C", "08306B"],
    "reds": ["FFF5F0", "FEE0D2", "FCBBA1", "FC9272", "FB6A4A", "EF3B2C", "CB181D", "A50F15", "67000D"],
    "greens": ["F7FCF5", "E5F5E0", "C7E9C0", "A1D99B", "74C476", "41AB5D", "238B45", "006D2C", "00441B"],
    "purples": ["FCFBFD", "EFEDF5", "DADAEB", "BCBDDC", "9E9AC8", "807DBA", "6A51A3", "54278F", "3F007D"],
    "oranges": ["FFF5EB", "FEE6CE", "FDD0A2", "FDAE6B", "FD8D3C", "F16913", "D94801", "A63603", "7F2704"],
    "greys": ["FFFFFF", "F0F0F0", "D9D9D9", "BDBDBD", "969696", "737373", "525252", "252525", "000000"],
    "ylgnbu": ["FFFFD9", "EDF8B1", "C7E9B4", "7FCDBB", "41B6C4", "1D91C0", "225EA8", "253494", "081D58"],
    "ylorrd": ["FFFFCC", "FFEDA0", "FED976", "FEB24C", "FD8D3C", "FC4E2A", "E31A1C", "BD0026", "800026"],
    # diverging
    "rdbu": ["67001F", "B2182B", "D6604D", "F4A582", "FDDBC7", "F7F7F7",
             "D1E5F0", "92C5DE", "4393C3", "2166AC", "053061"],
    "rdylbu": ["A50026", "D73027", "F46D43", "FDAE61", "FEE090", "FFFFBF",
               "E0F3F8", "ABD9E9", "74ADD1", "4575B4", "313695"],
    "spectral": ["9E0142", "D53E4F", "F46D43", "FDAE61", "FEE08B", "FFFFBF",
                 "E6F598", "ABDDA4", "66C2A5", "3288BD", "5E4FA2"],
    "coolwarm": ["3B4CC0", "5977E3", "7B9FF9", "9EBEFF", "C0D4F5", "DDDCDC",
                 "F2CBB7", "F7AC8E", "EE8468", "D65244", "B40426"],
    "piyg": ["8E0152", "C51B7D", "DE77AE", "F1B6DA", "FDE0EF", "F7F7F7",
             "E6F5D0", "B8E186", "7FBC41", "4D9221", "276419"],
}


class Colormap:
    """Maps numbers in [0, 1] to colors by linear interpolation."""

    def __init__(self, name: str, colors: Sequence[Any], levels: int = 256):
        if len(colors) < 2:
            raise ValueError("a colormap needs at least two colors")
        self.name = name
        self._specs = list(colors)
        self.levels = levels

    @classmethod
    def from_colors(cls, *colors: Any, name: str = "custom", levels: int = 256) -> "Colormap":
        """A colormap through the given colors (hex codes, palette names, tuples)."""
        return cls(name, colors, levels)

    def _anchors(self) -> list[tuple[float, float, float]]:
        from .color import _ACTIVE_PALETTE

        key = id(_ACTIVE_PALETTE.get())
        cached = getattr(self, "_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        out = self._compute_anchors()
        self._cache = (key, out)
        return out

    def _compute_anchors(self) -> list[tuple[float, float, float]]:
        out = []
        for spec in self._specs:
            if isinstance(spec, str) and len(spec) == 6 and all(c in "0123456789abcdefABCDEF" for c in spec):
                out.append(parse_hex(spec))
                continue
            c = as_color(spec)
            if c.rgb is None:
                raise ValueError(f"colormap color {spec!r} has no known RGB value")
            out.append(c.rgb)
        return out

    def rgb(self, t: float) -> tuple[float, float, float]:
        """Interpolated RGB at ``t`` (clipped to [0, 1])."""
        t = 0.0 if t != t else min(1.0, max(0.0, float(t)))  # NaN -> 0
        if self.levels:
            t = round(t * (self.levels - 1)) / (self.levels - 1)
        anchors = self._anchors()
        x = t * (len(anchors) - 1)
        i = min(int(x), len(anchors) - 2)
        f = x - i
        a, b = anchors[i], anchors[i + 1]
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f)

    def __call__(self, t: float) -> Color:
        return Color(self.rgb(t))

    def sample(self, n: int) -> list[Color]:
        """``n`` evenly spaced colors."""
        if n == 1:
            return [self(0.5)]
        return [self(i / (n - 1)) for i in range(n)]

    def reversed(self) -> "Colormap":
        return Colormap(self.name + "_r", list(reversed(self._specs)), self.levels)

    def __repr__(self) -> str:
        return f"Colormap({self.name!r})"


COLORMAPS: dict[str, Colormap] = {name: Colormap(name, codes) for name, codes in _DATA.items()}


def get_colormap(cmap: str | Colormap | Sequence[Any] | None) -> Colormap:
    """Look up a colormap by name (append ``_r`` to reverse it), or build one
    from a list of colors."""
    if cmap is None:
        return COLORMAPS["viridis"]
    if isinstance(cmap, Colormap):
        return cmap
    if isinstance(cmap, str):
        key = cmap.lower()
        if key.endswith("_r") and key[:-2] in COLORMAPS:
            return COLORMAPS[key[:-2]].reversed()
        if key in COLORMAPS:
            return COLORMAPS[key]
        raise ValueError(f"unknown colormap {cmap!r}; available: {sorted(COLORMAPS)}")
    return Colormap.from_colors(*cmap)
