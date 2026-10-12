"""Cell grids, heatmaps and bracketed matrices."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .._fmt import fmt, label_text, number_text
from ..color import Color, as_color, color_text
from ..colormaps import get_colormap
from ..coords import NodeRef, Point, as_point
from ..elements import Element
from ..render import LazyName

if TYPE_CHECKING:  # pragma: no cover
    from ..render import RenderContext

__all__ = ["CellGrid", "Heatmap", "Matrix"]

_ANCHORS = {
    "center": (0.5, 0.5),
    "north": (0.5, 1.0),
    "south": (0.5, 0.0),
    "east": (1.0, 0.5),
    "west": (0.0, 0.5),
    "north west": (0.0, 1.0),
    "north east": (1.0, 1.0),
    "south west": (0.0, 0.0),
    "south east": (1.0, 0.0),
}


def _p(x: float, y: float) -> str:
    return f"({fmt(x, 3)},{fmt(y, 3)})"


class CellGrid(Element):
    """A rows x columns grid of square cells (row 0 at the top).

    Cells can be filled (one color, a 2D list, or a function ``(i, j) ->
    color``) and labelled (``values``). Use :meth:`cell` and :meth:`corner`
    for coordinates when annotating, e.g. to draw the receptive field of a
    convolution, and :meth:`outline` to frame a block of cells.

    ``depth > 0`` draws the grid as the front of a stack of ``depth`` layers
    (a multi-channel tensor).
    """

    def __init__(
        self,
        shape: tuple[int, int] | None = None,
        *,
        values: Any = None,
        at: Any = (0, 0),
        anchor: str = "center",
        cell: float = 0.6,
        fill: Any = None,
        line_color: Any = None,
        line_width: float = 0.5,
        font: str = r"\small",
        text_color: Any = None,
        depth: int = 0,
        depth_step: float = 0.14,
        label: Any = None,
        label_position: str = "below",
        name: str | None = None,
        number_format: str = "{:g}",
    ):
        if shape is None:
            if values is None:
                raise ValueError("give a shape or values")
            arr = values if isinstance(values, np.ndarray) else np.array(values, dtype=object)
            shape = (arr.shape[0], arr.shape[1])
        self.rows, self.cols = int(shape[0]), int(shape[1])
        self.values = values
        self.cell_size = float(cell)
        if anchor not in _ANCHORS:
            raise ValueError(f"unknown anchor {anchor!r}")
        fx, fy = _ANCHORS[anchor]
        p = as_point(at)
        self.x0 = p.x - fx * self.cols * self.cell_size
        self.y0 = p.y - fy * self.rows * self.cell_size  # bottom edge
        self.fill = fill
        self.line_color = line_color
        self.line_width = line_width
        self.font = font
        self.text_color = text_color
        self.depth = depth
        self.depth_step = depth_step
        self.label = label
        self.label_position = label_position
        self.name = name
        self.number_format = number_format
        self._outlines: list[tuple[int, int, int, int, dict[str, Any]]] = []
        self._cell_fills: dict[tuple[int, int], Any] = {}

    # -- geometry -------------------------------------------------------------------
    @property
    def width(self) -> float:
        return self.cols * self.cell_size

    @property
    def height(self) -> float:
        return self.rows * self.cell_size

    def cell(self, i: int, j: int) -> Point:
        """Centre of cell (row ``i``, column ``j``)."""
        s = self.cell_size
        return Point(self.x0 + (j + 0.5) * s, self.y0 + self.height - (i + 0.5) * s)

    def corner(self, i: int, j: int, which: str = "north west") -> Point:
        """A corner of a cell: ``"north west"``, ``"south east"``, ..."""
        fx, fy = _ANCHORS[which]
        s = self.cell_size
        return Point(self.x0 + (j + fx) * s, self.y0 + self.height - (i + 1 - fy) * s)

    def block_corners(self, i0: int, j0: int, i1: int, j1: int) -> dict[str, Point]:
        """The four outer corners of the block of cells ``[i0..i1] x [j0..j1]``."""
        return {
            "north west": self.corner(i0, j0, "north west"),
            "north east": self.corner(i0, j1, "north east"),
            "south west": self.corner(i1, j0, "south west"),
            "south east": self.corner(i1, j1, "south east"),
        }

    def anchor(self, which: str) -> Point:
        """A point on the grid's bounding box (``"north"``, ``"east"``, ...)."""
        fx, fy = _ANCHORS[which]
        return Point(self.x0 + fx * self.width, self.y0 + fy * self.height)

    @property
    def north(self) -> Point:
        return self.anchor("north")

    @property
    def south(self) -> Point:
        return self.anchor("south")

    @property
    def east(self) -> Point:
        return self.anchor("east")

    @property
    def west(self) -> Point:
        return self.anchor("west")

    @property
    def center(self) -> Point:
        return self.anchor("center")

    # -- decoration -------------------------------------------------------------------
    def outline(self, i0: int, j0: int, i1: int | None = None, j1: int | None = None, *, color: str = "Red",
                fill: Any = None, line_width: float = 1.4, inset: float = 0.0) -> "CellGrid":
        """Frame the block of cells from ``(i0, j0)`` to ``(i1, j1)`` (inclusive)."""
        i1 = i0 if i1 is None else i1
        j1 = j0 if j1 is None else j1
        self._outlines.append((i0, j0, i1, j1, {"color": color, "fill": fill, "width": line_width, "inset": inset}))
        return self

    def color_cell(self, i: int, j: int, color: Any) -> "CellGrid":
        """Fill a single cell."""
        self._cell_fills[(i, j)] = color
        return self

    def color_block(self, i0: int, j0: int, i1: int, j1: int, color: Any) -> "CellGrid":
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self._cell_fills[(i, j)] = color
        return self

    # -- rendering ----------------------------------------------------------------------
    def _fill_of(self, i: int, j: int) -> Any:
        if (i, j) in self._cell_fills:
            return self._cell_fills[(i, j)]
        f = self.fill
        if f is None:
            return None
        if callable(f):
            return f(i, j)
        if isinstance(f, (str, Color)):
            return f
        return f[i][j]

    def _value_text(self, i: int, j: int) -> str:
        if self.values is None:
            return ""
        v = self.values[i][j]
        if v is None:
            return ""
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            text = label_text(self.number_format.format(v))
            return "$-$" + text[1:] if text.startswith("-") else text
        return label_text(v)

    def _text_color(self, fill: Any, ctx: "RenderContext") -> str:
        if self.text_color is not None:
            return str(self.text_color)
        if fill is None:
            return ctx.theme.ink
        c = fill if isinstance(fill, Color) else as_color(color_text(fill))
        if c.rgb is None:
            return ctx.theme.ink
        return ctx.theme.text_on(c.is_dark())

    def render(self, ctx: "RenderContext") -> list[str]:
        theme = ctx.theme
        s = self.cell_size
        line = str(self.line_color) if self.line_color is not None else f"{theme.ink}!55"
        out = [f"% cell grid {self.name or ''} {self.rows}x{self.cols}".rstrip()]
        x0, y0 = self.x0, self.y0
        x1, y1 = x0 + self.width, y0 + self.height
        # stacked layers behind (multi-channel look)
        for k in range(self.depth, 0, -1):
            dx = dy = k * self.depth_step
            base = color_text(self._fill_of(0, 0) or theme.paper or "white")
            out.append(
                f"\\filldraw[fill={base}, draw={line}, line width={self.line_width:g}pt] "
                f"{_p(x0 + dx, y0 + dy)} rectangle {_p(x1 + dx, y1 + dy)};"
            )
        # cell fills, grouped by color
        groups: dict[str, list[str]] = {}
        for i in range(self.rows):
            for j in range(self.cols):
                f = self._fill_of(i, j)
                if f is None:
                    continue
                a = self.corner(i, j, "south west")
                groups.setdefault(color_text(f), []).append(f"{_p(a.x, a.y)} rectangle {_p(a.x + s, a.y + s)}")
        if self.depth or groups:
            out.append(f"\\fill[{theme.paper or 'white'}] {_p(x0, y0)} rectangle {_p(x1, y1)};" if self.depth else "")
        for color, rects in groups.items():
            out.append(f"\\fill[{color}] " + " ".join(rects) + ";")
        # grid lines
        out.append(
            f"\\draw[{line}, line width={self.line_width:g}pt] {_p(x0, y0)} grid[step={fmt(s, 4)}] {_p(x1, y1)};"
            if abs((x1 - x0) / s - round((x1 - x0) / s)) < 1e-9 and abs(x0 / s - round(x0 / s)) < 1e-9
            and abs(y0 / s - round(y0 / s)) < 1e-9
            else self._grid_lines(line)
        )
        # values
        if self.values is not None:
            for i in range(self.rows):
                for j in range(self.cols):
                    text = self._value_text(i, j)
                    if not text:
                        continue
                    c = self.cell(i, j)
                    tc = self._text_color(self._fill_of(i, j), ctx)
                    out.append(f"\\node[font={self.font}, text={tc}, inner sep=0pt] at {_p(c.x, c.y)} {{{text}}};")
        # outlines
        for i0, j0, i1, j1, st in self._outlines:
            nw = self.corner(i0, j0, "north west")
            se = self.corner(i1, j1, "south east")
            d = st["inset"]
            fill = f", fill={st['fill']}" if st["fill"] is not None else ""
            out.append(
                f"\\draw[{st['color']}, line width={st['width']:g}pt{fill}] "
                f"{_p(nw.x + d, se.y + d)} rectangle {_p(se.x - d, nw.y - d)};"
            )
        if self.label is not None:
            if self.label_position == "above":
                top = y1 + self.depth * self.depth_step
                out.append(f"\\node[anchor=south, font=\\small, align=center] at {_p((x0 + x1) / 2, top + 0.05)} {{{label_text(self.label)}}};")
            else:
                out.append(f"\\node[anchor=north, font=\\small, align=center] at {_p((x0 + x1) / 2, y0 - 0.08)} {{{label_text(self.label)}}};")
        return [ln for ln in out if ln]

    def _grid_lines(self, line: str) -> str:
        s = self.cell_size
        segs = []
        for i in range(self.rows + 1):
            y = self.y0 + i * s
            segs.append(f"{_p(self.x0, y)} -- {_p(self.x0 + self.width, y)}")
        for j in range(self.cols + 1):
            x = self.x0 + j * s
            segs.append(f"{_p(x, self.y0)} -- {_p(x, self.y0 + self.height)}")
        return f"\\draw[{line}, line width={self.line_width:g}pt] " + " ".join(segs) + ";"


class Heatmap(Element):
    """A colored matrix with optional row/column labels, values and colorbar.

    Good for attention weights, confusion matrices and correlation tables::

        pic.add(Heatmap(weights, row_labels=tokens, col_labels=tokens,
                        cmap="blues", values=True))
    """

    def __init__(
        self,
        data: Any,
        *,
        at: Any = (0, 0),
        cell: float = 0.6,
        cmap: Any = "blues",
        vmin: float | None = None,
        vmax: float | None = None,
        row_labels: Sequence[Any] | None = None,
        col_labels: Sequence[Any] | None = None,
        col_label_angle: float = 0.0,
        col_labels_position: str = "top",
        values: bool = False,
        number_format: str = "{:.2f}",
        font: str = r"\scriptsize",
        label_font: str = r"\small",
        gap: float = 0.0,
        colorbar: bool = False,
        colorbar_label: Any = None,
        title: Any = None,
        xlabel: Any = None,
        ylabel: Any = None,
        border: bool = False,
        name: str | None = None,
    ):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError("heatmap data must be 2D")
        self.origin = as_point(at)  # top-left corner of the cells
        self.cell_size = cell
        self.cmap = get_colormap(cmap)
        self.vmin = float(np.nanmin(self.data)) if vmin is None else vmin
        self.vmax = float(np.nanmax(self.data)) if vmax is None else vmax
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.col_label_angle = col_label_angle
        self.col_labels_position = col_labels_position
        self.values = values
        self.number_format = number_format
        self.font = font
        self.label_font = label_font
        self.gap = gap
        self.colorbar = colorbar
        self.colorbar_label = colorbar_label
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.border = border
        self.name = name
        self._outlines: list[tuple[int, int, int, int, str, float]] = []

    @property
    def rows(self) -> int:
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        return self.data.shape[1]

    def cell(self, i: int, j: int) -> Point:
        """Centre of cell ``(i, j)``; row 0 is at the top."""
        s = self.cell_size
        return Point(self.origin.x + (j + 0.5) * s, self.origin.y - (i + 0.5) * s)

    def outline(self, i0: int, j0: int, i1: int | None = None, j1: int | None = None, *, color: str = "Red",
                line_width: float = 1.5) -> "Heatmap":
        """Frame a block of cells."""
        self._outlines.append((i0, j0, i0 if i1 is None else i1, j0 if j1 is None else j1, color, line_width))
        return self

    def color_of(self, value: float) -> Color:
        span = (self.vmax - self.vmin) or 1.0
        return self.cmap((value - self.vmin) / span)

    def render(self, ctx: "RenderContext") -> list[str]:
        theme = ctx.theme
        s, g = self.cell_size, self.gap
        x0, ytop = self.origin.x, self.origin.y
        name = self.name or ctx.auto_name(self, "hm")
        out = [f"% heatmap {name} {self.rows}x{self.cols}"]
        groups: dict[str, list[str]] = {}
        colors: dict[tuple[int, int], Color] = {}
        for i in range(self.rows):
            for j in range(self.cols):
                v = self.data[i, j]
                if not math.isfinite(v):
                    continue
                c = self.color_of(v)
                colors[(i, j)] = c
                xa, yb = x0 + j * s, ytop - (i + 1) * s
                pad = g / 2
                extra = 0.003 if g == 0 else 0.0
                groups.setdefault(c.expr, []).append(
                    f"{_p(xa + pad, yb + pad)} rectangle {_p(xa + s - pad + extra, yb + s - pad + extra)}"
                )
        for color, rects in groups.items():
            from ..elements import wrap_tokens

            out += wrap_tokens(f"\\fill[{color}]", " ".join(rects).split(" "))
        if self.border:
            out.append(f"\\draw[{theme.ink}!60, line width=0.5pt] {_p(x0, ytop - self.rows * s)} rectangle {_p(x0 + self.cols * s, ytop)};")
        if self.values:
            for (i, j), c in colors.items():
                p = self.cell(i, j)
                tc = theme.text_on(c.is_dark())
                text = label_text(self.number_format.format(self.data[i, j]))
                if text.startswith("-"):
                    text = "$-$" + text[1:]
                out.append(f"\\node[font={self.font}, text={tc}, inner sep=0pt] at {_p(p.x, p.y)} {{{text}}};")
        for i0, j0, i1, j1, color, lw in self._outlines:
            out.append(
                f"\\draw[{color}, line width={lw:g}pt] {_p(x0 + j0 * s, ytop - (i1 + 1) * s)} "
                f"rectangle {_p(x0 + (j1 + 1) * s, ytop - i0 * s)};"
            )
        lf = f"font={self.label_font}, align=center"
        row_names, col_names = [], []
        if self.row_labels is not None:
            for i, lab in enumerate(self.row_labels[: self.rows]):
                p = self.cell(i, 0)
                nm = f"{name}-r{i}"
                row_names.append(nm)
                out.append(f"\\node[anchor=east, {lf}, inner sep=2pt] ({nm}) at {_p(x0 - 0.06, p.y)} {{{label_text(lab)}}};")
        if self.col_labels is not None:
            top = self.col_labels_position == "top"
            for j, lab in enumerate(self.col_labels[: self.cols]):
                p = self.cell(0, j)
                nm = f"{name}-c{j}"
                col_names.append(nm)
                y = ytop + 0.06 if top else ytop - self.rows * s - 0.06
                a = self.col_label_angle
                if a == 0:
                    anchor = "south" if top else "north"
                    out.append(f"\\node[anchor={anchor}, {lf}, inner sep=2pt] ({nm}) at {_p(p.x, y)} {{{label_text(lab)}}};")
                else:
                    anchor = "west" if top else "east"
                    out.append(f"\\node[anchor={anchor}, rotate={fmt(a)}, {lf}, inner sep=2pt] ({nm}) at {_p(p.x, y)} {{{label_text(lab)}}};")
        xb, yb = x0 + self.cols * s, ytop - self.rows * s
        if self.xlabel is not None:
            if col_names and self.col_labels_position != "top":
                out.append(f"\\node[fit={''.join(f'({n})' for n in col_names)}, inner sep=0pt] ({name}-cols) {{}};")
                out.append(f"\\node[anchor=north, font=\\small, inner sep=4pt, align=center] at ({fmt((x0 + xb) / 2, 3)},0 |- {name}-cols.south) {{{label_text(self.xlabel)}}};")
            else:
                out.append(f"\\node[anchor=north, font=\\small, inner sep=4pt, align=center] at {_p((x0 + xb) / 2, yb - 0.05)} {{{label_text(self.xlabel)}}};")
        if self.ylabel is not None:
            if row_names:
                out.append(f"\\node[fit={''.join(f'({n})' for n in row_names)}, inner sep=0pt] ({name}-rows) {{}};")
                out.append(f"\\node[anchor=south, rotate=90, font=\\small, inner sep=4pt, align=center] at ({name}-rows.west |- 0,{fmt((ytop + yb) / 2, 3)}) {{{label_text(self.ylabel)}}};")
            else:
                out.append(f"\\node[anchor=south, rotate=90, font=\\small, inner sep=4pt, align=center] at {_p(x0 - 0.05, (ytop + yb) / 2)} {{{label_text(self.ylabel)}}};")
        if self.title is not None:
            if col_names and self.col_labels_position == "top":
                out.append(f"\\node[fit={''.join(f'({n})' for n in col_names)}, inner sep=0pt] ({name}-cols) {{}};")
                out.append(f"\\node[anchor=south, font=\\small\\bfseries, inner sep=4pt, align=center] at ({fmt((x0 + xb) / 2, 3)},0 |- {name}-cols.north) {{{label_text(self.title)}}};")
            else:
                out.append(f"\\node[anchor=south, font=\\small\\bfseries, inner sep=4pt, align=center] at {_p((x0 + xb) / 2, ytop + 0.05)} {{{label_text(self.title)}}};")
        if self.colorbar:
            bx0 = xb + 0.3
            bx1 = bx0 + 0.3
            n = 40
            for k in range(n):
                ya = yb + (ytop - yb) * k / n
                yc = yb + (ytop - yb) * (k + 1) / n + (0.003 if k < n - 1 else 0)
                out.append(f"\\fill[{self.cmap((k + 0.5) / n).expr}] {_p(bx0, ya)} rectangle {_p(bx1, yc)};")
            out.append(f"\\draw[{theme.ink}!60, line width=0.4pt] {_p(bx0, yb)} rectangle {_p(bx1, ytop)};")
            from ..ticks import format_tick, linear_ticks, nice_step

            step = nice_step(self.vmax - self.vmin, 4)
            for v in linear_ticks(self.vmin, self.vmax, step=step):
                y = yb + (ytop - yb) * (v - self.vmin) / ((self.vmax - self.vmin) or 1)
                out.append(f"\\draw[{theme.ink}!60, line width=0.4pt] {_p(bx1, y)} -- {_p(bx1 + 0.07, y)};")
                out.append(f"\\node[anchor=west, font=\\scriptsize, inner sep=2pt] at {_p(bx1 + 0.08, y)} {{{format_tick(v, step)}}};")
            if self.colorbar_label is not None:
                out.append(f"\\node[anchor=north, rotate=90, font=\\small] at {_p(bx1 + 0.75, (ytop + yb) / 2)} {{{label_text(self.colorbar_label)}}};")
        return out


def _entry(value: Any) -> str:
    """Matrix entry text (math mode): numbers without float noise."""
    if value is None:
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return number_text(value).replace("$-$", "-")
    return str(value)


class Matrix(Element):
    """A bracketed matrix whose entries are nodes (``matrix of math nodes``).

    Entries are referenced with :meth:`entry`; :meth:`highlight_row`,
    :meth:`highlight_column` and :meth:`highlight` draw rounded boxes behind
    entries (for matrix multiplication figures and the like).
    """

    def __init__(
        self,
        entries: Sequence[Sequence[Any]],
        *,
        at: Any = (0, 0),
        name: str | None = None,
        delimiters: str = "[]",
        column_sep: float = 0.15,
        row_sep: float = 0.1,
        cell_width: float = 0.7,
        cell_height: float = 0.55,
        label: Any = None,
        font: str | None = None,
        anchor: str = "center",
    ):
        self.entries = [list(row) for row in entries]
        self.origin = as_point(at)
        self._lname = LazyName(self, "M", name)
        self.delimiters = delimiters
        self.column_sep = column_sep
        self.row_sep = row_sep
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.label = label
        self.font = font
        self.anchor = anchor
        self._highlights: list[tuple[list[tuple[int, int]], str, float]] = []

    @property
    def name(self) -> str:
        """Matrix name; entries are ``<name>-<row>-<column>`` (1-based)."""
        return self._lname.resolve()

    def entry(self, i: int, j: int) -> NodeRef:
        """Node of entry ``(i, j)`` (0-based)."""
        return NodeRef(self._lname.sub(f"-{i + 1}-{j + 1}"))

    @property
    def node(self) -> NodeRef:
        """The whole matrix node."""
        return NodeRef(self._lname.sub(""))

    def highlight(self, cells: Sequence[tuple[int, int]], color: str = "Orange", padding: float = 1.5) -> "Matrix":
        self._highlights.append((list(cells), color, padding))
        return self

    def highlight_row(self, i: int, color: str = "Orange") -> "Matrix":
        return self.highlight([(i, 0), (i, len(self.entries[i]) - 1)], color)

    def highlight_column(self, j: int, color: str = "Orange") -> "Matrix":
        return self.highlight([(0, j), (len(self.entries) - 1, j)], color)

    def render(self, ctx: "RenderContext") -> list[str]:
        ctx.require("matrix", "fit", "backgrounds")
        theme = ctx.theme
        left, right = self.delimiters[0], self.delimiters[-1]
        delim = ""
        if self.delimiters and self.delimiters != "none":
            lb = "\\{" if left == "{" else left
            rb = "\\}" if right == "}" else right
            delim = f", left delimiter={{{lb}}}, right delimiter={{{rb}}}"
        font = f", font={self.font}" if self.font else ""
        out = [
            f"\\matrix[matrix of math nodes, anchor={self.anchor}{delim}, column sep={fmt(self.column_sep)}cm, "
            f"row sep={fmt(self.row_sep)}cm, nodes={{minimum width={fmt(self.cell_width)}cm, "
            f"minimum height={fmt(self.cell_height)}cm, anchor=center, inner sep=1pt{font}}}] "
            f"({self.name}) at {self.origin.tikz()} {{"
        ]
        for row in self.entries:
            out.append("  " + " & ".join(_entry(e) for e in row) + r" \\")
        out.append("};")
        for cells, color, pad in self._highlights:
            fit = "".join(f"({self.name}-{i + 1}-{j + 1})" for i, j in cells)
            out.append(r"\begin{scope}[on background layer]")
            out.append(
                f"  \\node[fit={fit}, inner sep={fmt(pad)}pt, rounded corners=3pt, "
                f"fill={theme.fill(color, 25)}, draw={theme.stroke(color)}, line width=0.6pt] {{}};"
            )
            out.append(r"\end{scope}")
        if self.label is not None:
            out.append(f"\\node[anchor=north, font=\\small] at ({self.name}.south) {{{label_text(self.label)}}};")
        return out
