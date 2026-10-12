"""2D plot axes computed in Python and drawn with plain TikZ.

All data is transformed to picture coordinates in Python, so plots can mix
freely with nodes, arrows and annotations (``ax.point(x, y)`` gives the
picture coordinate of a data point)::

    ax = pic.add(tz.Axes(width=8, height=5, xlabel="$x$", ylabel="$f(x)$"))
    ax.function(np.sin, (-np.pi, np.pi), label=r"$\\sin x$")
    ax.scatter(xs, ys, color="Orange", label="samples")
    ax.legend("upper left")

Frame styles: ``"box"`` (full frame), ``"open"`` (left and bottom spines),
``"cross"`` (arrows through the origin, textbook style) and ``"none"``.
"""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from .._fmt import fmt, inline_text, label_text, node_text
from ..color import Color, color_text, xmix
from ..colormaps import get_colormap
from ..contour import auto_levels, contour_lines, superlevel_loops
from ..coords import LazyCoord, Point
from ..elements import Element, wrap_tokens
from ..geometry import clip_polygon, clip_polyline
from ..options import Options
from ..ticks import format_log_tick, format_tick, linear_ticks, log_ticks, nice_step, pi_ticks

if TYPE_CHECKING:  # pragma: no cover
    from ..render import RenderContext

__all__ = ["Axes"]

_DASHES = {
    "-": None,
    "solid": None,
    "--": "dashed",
    "dashed": "dashed",
    ":": "dotted",
    "dotted": "dotted",
    "-.": "dash dot",
    "dashdot": "dash dot",
    "dash dot": "dash dot",
    "densely dashed": "densely dashed",
    "densely dotted": "densely dotted",
    "loosely dashed": "loosely dashed",
}

_LEGEND_LOCATIONS = {
    "upper right": ("north east", (1, 1)),
    "upper left": ("north west", (0, 1)),
    "lower right": ("south east", (1, 0)),
    "lower left": ("south west", (0, 0)),
    "upper center": ("north", (0.5, 1)),
    "lower center": ("south", (0.5, 0)),
    "center right": ("east", (1, 0.5)),
    "center left": ("west", (0, 0.5)),
    "center": ("center", (0.5, 0.5)),
}

PREC = 3  # decimals for canvas coordinates (0.01 mm)


def _c(x: float, y: float) -> str:
    return f"({fmt(x, PREC)},{fmt(y, PREC)})"


def _simplify(points: list[tuple[float, float]], eps: float = 0.002) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker: drop points that do not change the drawing."""
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        (x1, y1), (x2, y2) = points[i], points[j]
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy)
        best, best_d = -1, eps
        for k in range(i + 1, j):
            px, py = points[k]
            if norm == 0:
                d = math.hypot(px - x1, py - y1)
            else:
                d = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm
            if d > best_d:
                best, best_d = k, d
        if best >= 0:
            keep[best] = True
            stack.append((i, best))
            stack.append((best, j))
    return [p for p, k in zip(points, keep) if k]


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def _eval_function(f: Callable, xs: np.ndarray) -> np.ndarray:
    """Evaluate ``f`` vectorised if possible, element-wise otherwise."""
    with np.errstate(all="ignore"):
        try:
            ys = np.asarray(f(xs), dtype=float)
            if ys.shape == xs.shape:
                return ys
        except Exception:  # noqa: BLE001 - fall back to scalar evaluation
            pass
        out = []
        for x in xs:
            try:
                out.append(float(f(float(x))))
            except (ValueError, ZeroDivisionError, OverflowError):
                out.append(float("nan"))
        return np.asarray(out)


def _eval_grid(f: Callable, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        try:
            Z = np.asarray(f(X, Y), dtype=float)
            if Z.shape == X.shape:
                return Z
        except Exception:  # noqa: BLE001
            pass
        Z = np.empty_like(X)
        for idx in np.ndindex(X.shape):
            try:
                Z[idx] = float(f(float(X[idx]), float(Y[idx])))
            except (ValueError, ZeroDivisionError, OverflowError):
                Z[idx] = float("nan")
        return Z


def _line_options(dash: str | None) -> Options:
    opts = Options()
    if dash:
        mapped = _DASHES.get(dash, dash)
        if mapped:
            opts.add(mapped)
    return opts


# ---------------------------------------------------------------------------
# artists
# ---------------------------------------------------------------------------

class _Artist:
    label: Any = None
    clip = True
    zorder = 2

    def bounds(self) -> tuple[float, float, float, float] | None:
        return None

    def data_points(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Data the axis limits should include (default: the bounds' corners)."""
        b = self.bounds()
        if b is None:
            return None
        return np.array([b[0], b[1]], dtype=float), np.array([b[2], b[3]], dtype=float)

    def occupied(self, ax: "Axes") -> tuple[np.ndarray, np.ndarray] | None:
        """Points (data coordinates) a legend should avoid covering."""
        return self.data_points()

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str | None:
        return None


def _break_at_jumps(cx: np.ndarray, cy: np.ndarray, box: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Insert gaps where a sampled curve jumps from above the box to below it
    (or back) between two samples: a pole such as 1/x or tan x."""
    y0, y1 = box[1], box[3]
    a, b = cy[:-1], cy[1:]
    with np.errstate(invalid="ignore"):
        jumps = np.where(((a > y1) & (b < y0)) | ((a < y0) & (b > y1)))[0]
    if jumps.size == 0:
        return cx, cy
    cx = np.insert(cx.astype(float), jumps + 1, np.nan)
    cy = np.insert(cy.astype(float), jumps + 1, np.nan)
    return cx, cy


def _bounds_of(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float] | None:
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not mask.any():
        return None
    return float(xs[mask].min()), float(xs[mask].max()), float(ys[mask].min()), float(ys[mask].max())


class _Line(_Artist):
    def __init__(self, ax: "Axes", x: Any, y: Any, *, color: Any, line_width: float, dash: str | None,
                 label: Any, smooth: bool, marker: str | None, marker_size: float, opacity: float | None,
                 arrow: str | None, options: Any, break_jumps: bool = False):
        self.x, self.y = _as_array(x), _as_array(y)
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same length")
        self.color_index = ax._next_color() if color is None else None
        self.color = color
        self.line_width = line_width
        self.dash = dash
        self.label = label
        self.smooth = smooth
        self.marker = marker
        self.marker_size = marker_size
        self.opacity = opacity
        self.arrow = arrow
        self.options = Options(options)
        self.break_jumps = break_jumps

    def bounds(self) -> tuple[float, float, float, float] | None:
        return _bounds_of(self.x, self.y)

    def data_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    def _color(self, ctx: "RenderContext") -> str:
        return color_text(self.color) if self.color is not None else ctx.theme.color(self.color_index)

    def _opts(self, ctx: "RenderContext") -> Options:
        opts = _line_options(self.dash)
        opts.set("color", self._color(ctx))
        opts.set("line width", f"{self.line_width:g}pt")
        if self.opacity is not None:
            opts.set("opacity", self.opacity)
        if self.arrow:
            opts.set(self.arrow)
        opts.add(self.options)
        return opts

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        cx, cy = ax._to_canvas_arrays(self.x, self.y)
        if self.break_jumps:
            cx, cy = _break_at_jumps(cx, cy, ax._box())
        lines: list[str] = []
        pieces = clip_polyline(cx, cy, ax._box())
        opts = self._opts(ctx).render()
        for piece in pieces:
            pts = _simplify(piece)
            if self.smooth and len(pts) > 2:
                tokens = ["plot[smooth]", "coordinates", "{" + _c(*pts[0])] + [_c(*p) for p in pts[1:-1]] + [_c(*pts[-1]) + "}"]
            else:
                tokens = [_c(*pts[0])]
                for p in pts[1:]:
                    tokens += ["--", _c(*p)]
            lines += wrap_tokens(f"\\draw{opts}", tokens)
        if self.marker:
            lines += _markers(ax, cx, cy, self.marker, self.marker_size, self._color(ctx), "white")
        return lines

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str:
        opts = self._opts(ctx)
        opts.remove("->").remove("<-").remove("<->")
        sample = rf"\draw{opts.render()} (0,0) -- (0.6,0);"
        if self.marker:
            sample += _marker_tokens(self.marker, [(0.3, 0.0)], self.marker_size, self._color(ctx), "white")
        return sample


_MARKERS = ("o", "s", "^", "v", "d", "x", "+", "*")


def _marker_paths(marker: str, pts: Sequence[tuple[float, float]], size: float, color: str,
                  edge: str | None) -> tuple[str, list[str]]:
    """``(command with options, path tokens)`` drawing ``marker`` at ``pts``."""
    r = size / 28.4527559  # pt -> cm
    filled = f"\\filldraw[fill={color}, draw={edge}, line width=0.4pt]" if edge else f"\\fill[{color}]"
    tokens: list[str] = []
    if marker == "o":
        for x, y in pts:
            tokens += [_c(x, y), f"circle[radius={size:g}pt]"]
        return filled, tokens
    if marker == "s":
        s = r * 0.9
        for x, y in pts:
            tokens += [_c(x - s, y - s), "rectangle", _c(x + s, y + s)]
        return filled, tokens
    if marker in ("^", "v", "d", "*"):
        for x, y in pts:
            if marker == "^":
                poly = [(x, y + 1.25 * r), (x - 1.1 * r, y - 0.65 * r), (x + 1.1 * r, y - 0.65 * r)]
            elif marker == "v":
                poly = [(x, y - 1.25 * r), (x - 1.1 * r, y + 0.65 * r), (x + 1.1 * r, y + 0.65 * r)]
            elif marker == "d":
                poly = [(x, y + 1.3 * r), (x + 0.95 * r, y), (x, y - 1.3 * r), (x - 0.95 * r, y)]
            else:
                poly = []
                for k in range(10):
                    rad = 1.35 * r if k % 2 == 0 else 0.55 * r
                    a = math.radians(90 + 36 * k)
                    poly.append((x + rad * math.cos(a), y + rad * math.sin(a)))
            for i, p in enumerate(poly):
                tokens += ([_c(*p)] if i == 0 else ["--", _c(*p)])
            tokens += ["--", "cycle"]
        return filled, tokens
    if marker in ("x", "+"):
        for x, y in pts:
            if marker == "x":
                tokens += [_c(x - r, y - r), "--", _c(x + r, y + r), _c(x - r, y + r), "--", _c(x + r, y - r)]
            else:
                tokens += [_c(x - 1.2 * r, y), "--", _c(x + 1.2 * r, y), _c(x, y - 1.2 * r), "--", _c(x, y + 1.2 * r)]
        return f"\\draw[{color}, line width=1pt, line cap=round]", tokens
    raise ValueError(f"unknown marker {marker!r}; use one of {_MARKERS}")


def _marker_tokens(marker: str, pts: Sequence[tuple[float, float]], size: float, color: str, edge: str | None) -> str:
    head, tokens = _marker_paths(marker, pts, size, color, edge)
    return head + " " + " ".join(tokens) + ";"


def _markers(ax: "Axes", cx: np.ndarray, cy: np.ndarray, marker: str, size: float, color: str, edge: str | None) -> list[str]:
    x0, y0, x1, y1 = ax._box()
    eps = 1e-6
    pts = [
        (float(x), float(y))
        for x, y in zip(cx, cy)
        if math.isfinite(x) and math.isfinite(y) and x0 - eps <= x <= x1 + eps and y0 - eps <= y <= y1 + eps
    ]
    if not pts:
        return []
    if edge and marker not in ("x", "+"):
        # one path per marker: later markers must cover earlier outlines
        out = []
        for p in pts:
            head, tokens = _marker_paths(marker, [p], size, color, edge)
            out.append(head + " " + " ".join(tokens) + ";")
        return out
    head, tokens = _marker_paths(marker, pts, size, color, edge)
    return wrap_tokens(head, tokens)


class _Scatter(_Artist):
    zorder = 3

    def __init__(self, ax: "Axes", x: Any, y: Any, *, color: Any, values: Any, cmap: Any, vmin: Any, vmax: Any,
                 marker: str, size: float, edgecolor: Any, label: Any, opacity: float | None):
        self.x, self.y = _as_array(x), _as_array(y)
        self.color_index = ax._next_color() if color is None and values is None else None
        self.color = color
        self.values = None if values is None else _as_array(values)
        self.cmap = get_colormap(cmap) if values is not None else None
        self.vmin, self.vmax = vmin, vmax
        self.marker = marker
        self.size = size
        self.edgecolor = edgecolor
        self.label = label
        self.opacity = opacity

    def bounds(self) -> tuple[float, float, float, float] | None:
        return _bounds_of(self.x, self.y)

    def data_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    def _color(self, ctx: "RenderContext") -> str:
        if self.color is not None:
            return color_text(self.color)
        if self.color_index is not None:
            return ctx.theme.color(self.color_index)
        return ctx.theme.ink

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        cx, cy = ax._to_canvas_arrays(self.x, self.y)
        edge = None if self.edgecolor in (None, "none") else color_text(self.edgecolor)
        lines: list[str] = []
        if self.opacity is not None:
            lines.append(rf"\begin{{scope}}[opacity={fmt(self.opacity)}]")
        if self.values is None:
            lines += _markers(ax, cx, cy, self.marker, self.size, self._color(ctx), edge)
        else:
            lo = float(np.nanmin(self.values)) if self.vmin is None else self.vmin
            hi = float(np.nanmax(self.values)) if self.vmax is None else self.vmax
            groups: dict[str, list[int]] = {}
            for i, v in enumerate(self.values):
                t = 0.5 if hi == lo else (v - lo) / (hi - lo)
                groups.setdefault(self.cmap(t).expr, []).append(i)  # type: ignore[union-attr]
            for color, idx in groups.items():
                lines += _markers(ax, cx[idx], cy[idx], self.marker, self.size, color, edge)
        if self.opacity is not None:
            lines.append(r"\end{scope}")
        return lines

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str:
        edge = None if self.edgecolor in (None, "none") else color_text(self.edgecolor)
        return _marker_tokens(self.marker, [(0.3, 0.0)], self.size, self._color(ctx), edge)


class _Fill(_Artist):
    zorder = 1

    def __init__(self, ax: "Axes", x: Any, y1: Any, y2: Any, *, color: Any, opacity: float, label: Any,
                 line: bool, options: Any):
        self.x = _as_array(x)
        self.y1 = np.broadcast_to(_as_array(y1) if np.ndim(y1) else np.full(self.x.shape, float(y1)), self.x.shape)
        self.y2 = np.broadcast_to(_as_array(y2) if np.ndim(y2) else np.full(self.x.shape, float(y2)), self.x.shape)
        self.color_index = ax._next_color() if color is None else None
        self.color = color
        self.opacity = opacity
        self.label = label
        self.line = line
        self.options = Options(options)

    def bounds(self) -> tuple[float, float, float, float] | None:
        b1, b2 = _bounds_of(self.x, self.y1), _bounds_of(self.x, self.y2)
        if b1 is None or b2 is None:
            return b1 or b2
        return min(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), max(b1[3], b2[3])

    def data_points(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate([self.x, self.x]), np.concatenate([self.y1, self.y2])

    def _color(self, ctx: "RenderContext") -> str:
        return color_text(self.color) if self.color is not None else ctx.theme.color(self.color_index)

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        cx, cy1 = ax._to_canvas_arrays(self.x, self.y1)
        _, cy2 = ax._to_canvas_arrays(self.x, self.y2)
        cy1, cy2 = ax._floor(self.y1, cy1), ax._floor(self.y2, cy2)
        ok = np.isfinite(cx) & np.isfinite(cy1) & np.isfinite(cy2)
        lines: list[str] = []
        color = self._color(ctx)
        opts = Options(fill=color)
        if self.opacity < 1:
            opts.set("fill opacity", self.opacity)
        opts.add(self.options)
        # split at gaps (NaN)
        start = None
        runs = []
        for i, good in enumerate(list(ok) + [False]):
            if good and start is None:
                start = i
            elif not good and start is not None:
                runs.append((start, i))
                start = None
        for a, b in runs:
            if b - a < 2:
                continue
            poly = list(zip(cx[a:b], cy1[a:b])) + list(zip(cx[a:b][::-1], cy2[a:b][::-1]))
            clipped = clip_polygon(poly, ax._box())
            if len(clipped) < 3:
                continue
            tokens = []
            for p in _simplify(clipped + [clipped[0]])[:-1]:
                tokens += [_c(*p), "--"]
            tokens.append("cycle")
            lines += wrap_tokens(f"\\fill{opts.render()}", tokens)
            if self.line:
                for ys in (cy1, cy2):
                    for piece in clip_polyline(cx[a:b], ys[a:b], ax._box()):
                        pts = _simplify(piece)
                        toks = [_c(*pts[0])]
                        for p in pts[1:]:
                            toks += ["--", _c(*p)]
                        lines += wrap_tokens(f"\\draw[{color}]", toks)
        return lines

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str:
        a = f", fill opacity={fmt(self.opacity)}" if self.opacity < 1 else ""
        return rf"\fill[{self._color(ctx)}{a}] (0,-0.1) rectangle (0.6,0.12);"


class _Bars(_Artist):
    zorder = 1

    def __init__(self, ax: "Axes", x: Any, heights: Any, *, width: float, bottom: Any, color: Any, label: Any,
                 opacity: float | None, edge: bool, horizontal: bool):
        self.x = _as_array(x)
        self.h = _as_array(heights)
        self.bottom = np.broadcast_to(_as_array(bottom) if np.ndim(bottom) else np.full(self.x.shape, float(bottom)), self.x.shape)
        self.width = width
        self.color_index = ax._next_color() if color is None else None
        self.color = color
        self.label = label
        self.opacity = opacity
        self.edge = edge
        self.horizontal = horizontal

    def _rects(self) -> list[tuple[float, float, float, float]]:
        out = []
        for x, h, b in zip(self.x, self.h, self.bottom):
            lo, hi = sorted((b, b + h))
            if self.horizontal:
                out.append((lo, x - self.width / 2, hi, x + self.width / 2))
            else:
                out.append((x - self.width / 2, lo, x + self.width / 2, hi))
        return out

    def bounds(self) -> tuple[float, float, float, float] | None:
        rs = self._rects()
        if not rs:
            return None
        return min(r[0] for r in rs), max(r[2] for r in rs), min(r[1] for r in rs), max(r[3] for r in rs)

    def data_points(self) -> tuple[np.ndarray, np.ndarray] | None:
        rs = self._rects()
        if not rs:
            return None
        xs = np.array([v for r in rs for v in (r[0], r[2])], dtype=float)
        ys = np.array([v for r in rs for v in (r[1], r[3])], dtype=float)
        return xs, ys

    def occupied(self, ax: "Axes") -> tuple[np.ndarray, np.ndarray] | None:
        xs, ys = [], []
        for x0, y0, x1, y1 in self._rects():
            for fx in np.linspace(0, 1, 5):
                for fy in np.linspace(0, 1, 5):
                    xs.append(x0 + (x1 - x0) * fx)
                    ys.append(y0 + (y1 - y0) * fy)
        return (np.array(xs), np.array(ys)) if xs else None

    def _colors(self, ctx: "RenderContext") -> list[str]:
        if self.color is None:
            return [ctx.theme.color(self.color_index)] * len(self.x)
        if isinstance(self.color, (list, tuple)) and len(self.color) == len(self.x) and not (
            len(self.color) == 3 and all(isinstance(c, (int, float)) for c in self.color)
        ):
            return [color_text(c) for c in self.color]
        return [color_text(self.color)] * len(self.x)

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        lines = []
        for (x0, y0, x1, y1), color in zip(self._rects(), self._colors(ctx)):
            cx, cy = ax._to_canvas_arrays(np.array([x0, x1]), np.array([y0, y1]))
            cx, cy = ax._floor(np.array([x0, x1]), cx, axis="x"), ax._floor(np.array([y0, y1]), cy)
            if not (np.all(np.isfinite(cx)) and np.all(np.isfinite(cy))):
                continue
            (px0, px1), (py0, py1) = cx, cy
            clipped = clip_polygon([(px0, py0), (px1, py0), (px1, py1), (px0, py1)], ax._box())
            if len(clipped) < 3:
                continue
            xs = [p[0] for p in clipped]
            ys = [p[1] for p in clipped]
            opts = Options(fill=xmix(color, 85) if self.edge else color)
            if self.edge:
                opts.set("draw", xmix(color, 75, "black"))
                opts.set("line width", "0.5pt")
            if self.opacity is not None:
                opts.set("fill opacity", self.opacity)
            cmd = "filldraw" if self.edge else "fill"
            lines.append(f"\\{cmd}{opts.render()} {_c(min(xs), min(ys))} rectangle {_c(max(xs), max(ys))};")
        return lines

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str:
        return rf"\fill[{self._colors(ctx)[0]}] (0.12,-0.12) rectangle (0.48,0.14);"


class _RefLine(_Artist):
    """Horizontal/vertical/sloped reference line across the axes."""

    zorder = 1

    def __init__(self, kind: str, value: Any, *, color: Any, line_width: float, dash: str | None,
                 label: Any, options: Any, extent: tuple[float | None, float | None] = (None, None)):
        self.kind = kind
        self.value = value
        self.color = color
        self.line_width = line_width
        self.dash = dash
        self.label = label
        self.options = Options(options)
        self.extent = extent

    def _opts(self, ctx: "RenderContext") -> Options:
        opts = _line_options(self.dash)
        opts.set("color", color_text(self.color) if self.color is not None else f"{ctx.theme.ink}!55")
        opts.set("line width", f"{self.line_width:g}pt")
        opts.add(self.options)
        return opts

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        (xa, xb), (ya, yb) = ax.xlim, ax.ylim
        if self.kind == "h":
            lo = self.extent[0] if self.extent[0] is not None else xa
            hi = self.extent[1] if self.extent[1] is not None else xb
            xs, ys = np.array([lo, hi]), np.array([self.value, self.value])
        elif self.kind == "v":
            lo = self.extent[0] if self.extent[0] is not None else ya
            hi = self.extent[1] if self.extent[1] is not None else yb
            xs, ys = np.array([self.value, self.value]), np.array([lo, hi])
        else:  # sloped: value = (point, slope)
            (px, py), slope = self.value
            xs = np.array([xa, xb])
            ys = py + slope * (xs - px)
        cx, cy = ax._to_canvas_arrays(xs, ys)
        out = []
        for piece in clip_polyline(cx, cy, ax._box()):
            out.append(f"\\draw{self._opts(ctx).render()} {_c(*piece[0])} -- {_c(*piece[-1])};")
        return out

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str:
        return rf"\draw{self._opts(ctx).render()} (0,0) -- (0.6,0);"


class _Text(_Artist):
    clip = False
    zorder = 5

    def __init__(self, x: float, y: float, text: Any, options: Any):
        self.x, self.y, self.text = x, y, text
        self.options = Options(options)

    def occupied(self, ax: "Axes") -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.x], dtype=float), np.array([self.y], dtype=float)

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        p = ax._to_canvas(self.x, self.y)
        text, multi = node_text(self.text)
        opts = self.options.copy()
        if multi and "align" not in opts:
            opts.set("align", "center")
        return [f"\\node{opts.render()} at {_c(p.x, p.y)} {{{text}}};"]


class _Annotation(_Artist):
    clip = False
    zorder = 5

    def __init__(self, text: Any, xy: tuple[float, float], xytext: tuple[float, float], arrow: Any, options: Any,
                 arrow_options: Any):
        self.text, self.xy, self.xytext = text, xy, xytext
        self.arrow = arrow
        self.options = Options(options)
        self.arrow_options = Options(arrow_options)

    def occupied(self, ax: "Axes") -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.xy[0], self.xytext[0]], dtype=float),
                np.array([self.xy[1], self.xytext[1]], dtype=float))

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        p = ax._to_canvas(*self.xy)
        q = ax._to_canvas(*self.xytext)
        text, multi = node_text(self.text)
        opts = self.options.copy()
        if multi and "align" not in opts:
            opts.set("align", "center")
        name = ctx.auto_name(self, "ann")
        lines = [f"\\node{opts.render()} ({name}) at {_c(q.x, q.y)} {{{text}}};"]
        if self.arrow:
            aopts = Options("->" if self.arrow is True else self.arrow, shorten_end="2pt")
            aopts.add(self.arrow_options)
            lines.append(f"\\draw{aopts.render()} ({name}) -- {_c(p.x, p.y)};")
        return lines


class _DataPath(_Artist):
    """Arbitrary path (polyline/polygon/arrow) in data coordinates."""

    def __init__(self, points: Sequence[Any], closed: bool, command: str, options: Any, label: Any = None,
                 clip: bool = True, autoscale: bool = True):
        self.points = [(float(p[0]), float(p[1])) for p in points]
        self.closed = closed
        self.command = command
        self.options = Options(options)
        self.label = label
        self.clip = clip
        self.autoscale = autoscale

    def bounds(self) -> tuple[float, float, float, float] | None:
        if not self.autoscale or not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), max(xs), min(ys), max(ys)

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        pts = [ax._to_canvas(x, y) for x, y in self.points]
        tokens: list[str] = []
        for i, p in enumerate(pts):
            if i:
                tokens.append("--")
            tokens.append(_c(p.x, p.y))
        if self.closed:
            tokens += ["--", "cycle"]
        return wrap_tokens(f"\\{self.command}{self.options.render()}", tokens)

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str:
        if self.closed:
            return rf"\{self.command}{self.options.render()} (0.05,-0.1) rectangle (0.55,0.12);"
        return rf"\draw{self.options.render()} (0,0) -- (0.6,0);"


class _Image(_Artist):
    """Grid of colored cells (``imshow``)."""

    zorder = 0

    def __init__(self, Z: Any, extent: tuple[float, float, float, float], cmap: Any, vmin: Any, vmax: Any,
                 opacity: float | None):
        self.Z = np.asarray(Z, dtype=float)
        if self.Z.ndim != 2:
            raise ValueError("imshow needs a 2D array")
        self.extent = extent
        self.cmap = get_colormap(cmap)
        self.vmin = float(np.nanmin(self.Z)) if vmin is None else vmin
        self.vmax = float(np.nanmax(self.Z)) if vmax is None else vmax
        self.opacity = opacity

    def bounds(self) -> tuple[float, float, float, float] | None:
        x0, x1, y0, y1 = self.extent
        return x0, x1, y0, y1

    def occupied(self, ax: "Axes") -> tuple[np.ndarray, np.ndarray]:
        x0, x1, y0, y1 = self.extent
        X, Y = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 9))
        return X.ravel(), Y.ravel()

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        ny, nx = self.Z.shape
        x0, x1, y0, y1 = self.extent
        xs = np.linspace(x0, x1, nx + 1)
        ys = np.linspace(y0, y1, ny + 1)
        cxs = [ax._to_canvas(x, y0).x for x in xs]
        cys = [ax._to_canvas(x0, y).y for y in ys]
        groups: dict[str, list[str]] = {}
        span = (self.vmax - self.vmin) or 1.0
        pad = 0.004
        for i in range(ny):
            for j in range(nx):
                v = self.Z[i, j]
                if not math.isfinite(v):
                    continue
                color = self.cmap((v - self.vmin) / span).expr
                groups.setdefault(color, []).append(
                    f"{_c(cxs[j], cys[i])} rectangle {_c(cxs[j + 1] + pad, cys[i + 1] + pad)}"
                )
        lines = []
        if self.opacity is not None:
            lines.append(rf"\begin{{scope}}[fill opacity={fmt(self.opacity)}]")
        for color, rects in groups.items():
            lines += wrap_tokens(f"\\fill[{color}]", " ".join(rects).split(" "))
        if self.opacity is not None:
            lines.append(r"\end{scope}")
        return lines


class _Contour(_Artist):
    zorder = 1

    def __init__(self, ax: "Axes", source: Any, x: Any, y: Any, *, levels: Any, filled: bool, cmap: Any,
                 colors: Any, line_width: float, opacity: float | None, resolution: int, dash: str | None,
                 lines_over: bool, line_color: Any, label: Any):
        self.ax = ax
        self.source = source
        self.x = None if x is None else _as_array(x)
        self.y = None if y is None else _as_array(y)
        self.levels = levels
        self.filled = filled
        self.cmap = get_colormap(cmap) if cmap is not None or colors is None else None
        self.colors = colors
        self.line_width = line_width
        self.opacity = opacity
        self.resolution = resolution
        self.dash = dash
        self.lines_over = lines_over
        self.line_color = line_color
        self.label = label

    def bounds(self) -> tuple[float, float, float, float] | None:
        if callable(self.source):
            return None
        ny, nx = np.asarray(self.source).shape
        x = self.x if self.x is not None else np.arange(nx, dtype=float)
        y = self.y if self.y is not None else np.arange(ny, dtype=float)
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str | None:
        if self.filled:
            return None
        if self.colors is not None:
            color = color_text(self.colors if isinstance(self.colors, (str, Color)) else list(self.colors)[0])
        else:
            color = (self.cmap or get_colormap("viridis"))(0.5).expr
        dash = _line_options(self.dash).render().strip("[]")
        extra = f", {dash}" if dash else ""
        return rf"\draw[{color}, line width={self.line_width:g}pt{extra}] (0,0) -- (0.6,0);"

    def grid(self, ax: "Axes") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if callable(self.source):
            (xa, xb), (ya, yb) = ax.xlim, ax.ylim
            n = self.resolution
            ny = max(10, int(round(n * ax.height / ax.width)))
            x = np.linspace(xa, xb, n)
            y = np.linspace(ya, yb, ny)
            X, Y = np.meshgrid(x, y)
            return x, y, _eval_grid(self.source, X, Y)
        Z = np.asarray(self.source, dtype=float)
        ny, nx = Z.shape
        x = self.x if self.x is not None else np.arange(nx, dtype=float)
        y = self.y if self.y is not None else np.arange(ny, dtype=float)
        return x, y, Z

    def _levels(self, Z: np.ndarray) -> list[float]:
        if self.levels is None:
            return auto_levels(Z, 10)
        if isinstance(self.levels, int):
            return auto_levels(Z, self.levels)
        return sorted(float(v) for v in self.levels)

    def level_colors(self, levels: list[float], filled: bool) -> list[str]:
        n = len(levels) + (1 if filled else 0)
        if self.colors is not None:
            cols = [self.colors] * n if isinstance(self.colors, (str, Color)) else list(self.colors)
            return [color_text(cols[i % len(cols)]) for i in range(n)]
        cmap = self.cmap or get_colormap("viridis")
        return [cmap(i / max(1, n - 1)).expr for i in range(n)]

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        x, y, Z = self.grid(ax)
        levels = self._levels(Z)
        cx = np.array([ax._to_canvas(v, y[0]).x for v in x])
        cy = np.array([ax._to_canvas(x[0], v).y for v in y])
        lines: list[str] = []
        box = ax._box()
        gx0, gx1 = max(box[0], float(cx.min())), min(box[2], float(cx.max()))
        gy0, gy1 = max(box[1], float(cy.min())), min(box[3], float(cy.max()))
        if self.filled:
            colors = self.level_colors(levels, True)
            lines.append(r"\begin{scope}")
            lines.append(f"  \\clip {_c(gx0, gy0)} rectangle {_c(gx1, gy1)};")
            op = f", fill opacity={fmt(self.opacity)}" if self.opacity is not None else ""
            if np.all(np.isfinite(Z)):
                lines.append(f"  \\fill[{colors[0]}{op}] {_c(gx0, gy0)} rectangle {_c(gx1, gy1)};")
            else:  # leave undefined (NaN) regions empty
                tokens = []
                for loop in superlevel_loops(cx, cy, np.isfinite(Z).astype(float), 0.5):
                    pts = _simplify(loop)
                    if len(pts) < 3:
                        continue
                    for i, p in enumerate(pts[:-1]):
                        tokens += ([_c(*p)] if i == 0 else ["--", _c(*p)])
                    tokens += ["--", "cycle"]
                if tokens:
                    lines += ["  " + ln for ln in wrap_tokens(f"\\fill[{colors[0]}, even odd rule{op}]", tokens)]
            for level, color in zip(levels, colors[1:]):
                loops = superlevel_loops(cx, cy, Z, level)
                tokens: list[str] = []
                for loop in loops:
                    pts = _simplify(loop)
                    if len(pts) < 3:
                        continue
                    for i, p in enumerate(pts[:-1]):
                        tokens += ([_c(*p)] if i == 0 else ["--", _c(*p)])
                    tokens += ["--", "cycle"]
                if tokens:
                    lines += ["  " + ln for ln in wrap_tokens(f"\\fill[{color}, even odd rule{op}]", tokens)]
            lines.append(r"\end{scope}")
        if not self.filled or self.lines_over:
            if self.filled:
                line_colors = [color_text(self.line_color or f"{ctx.theme.ink}!45")] * len(levels)
            else:
                line_colors = self.level_colors(levels, False)
            opts_base = _line_options(self.dash)
            opts_base.set("line width", f"{self.line_width:g}pt")
            for level, color in zip(levels, line_colors):
                for line in contour_lines(cx, cy, Z, level):
                    for piece in clip_polyline([p[0] for p in line], [p[1] for p in line], box):
                        pts = _simplify(piece)
                        tokens = [_c(*pts[0])]
                        for p in pts[1:]:
                            tokens += ["--", _c(*p)]
                        lines += wrap_tokens(f"\\draw{opts_base.merged(color=color).render()}", tokens)
        return lines


class _Regions(_Artist):
    """Filled areas where a classifier ``f(x, y) -> class index`` is constant."""

    zorder = 0

    def __init__(self, f: Callable, colors: Sequence[Any], opacity: float, resolution: int,
                 boundary: bool, boundary_color: Any, boundary_width: float):
        self.f = f
        self.colors = list(colors)
        self.opacity = opacity
        self.resolution = resolution
        self.boundary = boundary
        self.boundary_color = boundary_color
        self.boundary_width = boundary_width

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        (xa, xb), (ya, yb) = ax.xlim, ax.ylim
        n = self.resolution
        ny = max(10, int(round(n * ax.height / ax.width)))
        x = np.linspace(xa, xb, n)
        y = np.linspace(ya, yb, ny)
        X, Y = np.meshgrid(x, y)
        C = np.rint(_eval_grid(self.f, X, Y))
        cx = np.array([ax._to_canvas(v, ya).x for v in x])
        cy = np.array([ax._to_canvas(xa, v).y for v in y])
        box = ax._box()
        out = [r"\begin{scope}", f"  \\clip {_c(box[0], box[1])} rectangle {_c(box[2], box[3])};"]
        op = f", fill opacity={fmt(self.opacity)}" if self.opacity < 1 else ""
        classes = sorted({int(c) for c in np.unique(C[np.isfinite(C)])})
        for k in classes:
            color = color_text(self.colors[k % len(self.colors)])
            indicator = (C == k).astype(float)
            tokens: list[str] = []
            for loop in superlevel_loops(cx, cy, indicator, 0.5):
                pts = _simplify(loop, 0.004)
                if len(pts) < 3:
                    continue
                for i, p in enumerate(pts[:-1]):
                    tokens += ([_c(*p)] if i == 0 else ["--", _c(*p)])
                tokens += ["--", "cycle"]
            if tokens:
                out += ["  " + ln for ln in wrap_tokens(f"\\fill[{color}{op}, even odd rule]", tokens)]
        out.append(r"\end{scope}")
        if self.boundary:
            color = str(self.boundary_color or f"{ctx.theme.ink}!70")
            for k in classes:
                indicator = (C == k).astype(float)
                for line in contour_lines(cx, cy, indicator, 0.5):
                    for piece in clip_polyline([p[0] for p in line], [p[1] for p in line], box):
                        pts = _simplify(piece, 0.004)
                        if len(pts) < 2:
                            continue
                        tokens = [_c(*pts[0])]
                        for p in pts[1:]:
                            tokens += ["--", _c(*p)]
                        out += wrap_tokens(f"\\draw[{color}, line width={self.boundary_width:g}pt]", tokens)
        return out


class _Quiver(_Artist):
    zorder = 2

    def __init__(self, X: Any, Y: Any, U: Any, V: Any, *, scale: float | None, normalize: bool, color: Any,
                 cmap: Any, line_width: float, pivot: str, length: float, label: Any, tip: str):
        self.X, self.Y = _as_array(X), _as_array(Y)
        self.U, self.V = _as_array(U), _as_array(V)
        self.scale = scale
        self.normalize = normalize
        self.color = color
        self.cmap = get_colormap(cmap) if cmap is not None else None
        self.line_width = line_width
        self.pivot = pivot
        self.length = length
        self.label = label
        self.tip = tip

    def bounds(self) -> tuple[float, float, float, float] | None:
        return _bounds_of(self.X, self.Y)

    def data_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X, self.Y

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str:
        color = color_text(self.color) if self.color is not None else ctx.theme.ink
        return rf"\draw[{self.tip}, {color}, line width={self.line_width:g}pt] (0.05,0) -- (0.55,0);"

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        sx = ax.width / (ax._fx(ax.xlim[1]) - ax._fx(ax.xlim[0]))
        sy = ax.height / (ax._fy(ax.ylim[1]) - ax._fy(ax.ylim[0]))
        du, dv = self.U * sx, self.V * sy  # canvas direction
        mags = np.hypot(du, dv)
        data_mag = np.hypot(self.U, self.V)
        finite = mags[np.isfinite(mags) & (mags > 0)]
        if finite.size == 0:
            return []
        # spacing between arrows (canvas) to choose a default length
        cx, cy = ax._to_canvas_arrays(self.X, self.Y)
        if self.normalize:
            target = self.length
            factor = np.where(mags > 0, target / np.where(mags > 0, mags, 1), 0)
        else:
            k = self.scale if self.scale is not None else self.length / float(finite.max())
            factor = np.full_like(mags, k)
        box = ax._box()
        lo_m, hi_m = float(np.nanmin(data_mag)), float(np.nanmax(data_mag))
        lines = []
        for i in range(len(cx)):
            if not (np.isfinite(mags[i]) and mags[i] > 0):
                continue
            ex, ey = du[i] * factor[i], dv[i] * factor[i]
            if self.pivot == "middle":
                x0, y0 = cx[i] - ex / 2, cy[i] - ey / 2
            else:
                x0, y0 = cx[i], cy[i]
            if not (box[0] - 1e-6 <= cx[i] <= box[2] + 1e-6 and box[1] - 1e-6 <= cy[i] <= box[3] + 1e-6):
                continue
            if self.cmap is not None:
                t = 0.5 if hi_m == lo_m else (data_mag[i] - lo_m) / (hi_m - lo_m)
                color = self.cmap(t).expr
            else:
                color = color_text(self.color) if self.color is not None else ctx.theme.ink
            lines.append(
                f"\\draw[{self.tip}, {color}, line width={self.line_width:g}pt] "
                f"{_c(x0, y0)} -- {_c(x0 + ex, y0 + ey)};"
            )
        return lines


class _Streamlines(_Artist):
    zorder = 2

    def __init__(self, f: Callable, *, density: float, color: Any, cmap: Any, line_width: float,
                 arrow_size: float, min_length: float, label: Any, seed: int):
        self.f = f
        self.density = density
        self.color = color
        self.cmap = get_colormap(cmap) if cmap is not None else None
        self.line_width = line_width
        self.arrow_size = arrow_size
        self.min_length = min_length
        self.label = label
        self.seed = seed

    def _trajectories(self, ax: "Axes") -> list[np.ndarray]:
        (xa, xb), (ya, yb) = ax.xlim, ax.ylim
        n = max(8, int(round(25 * self.density)))
        ny = max(6, int(round(n * ax.height / ax.width)))
        occupied = np.zeros((ny, n), dtype=bool)

        def cell(px: float, py: float) -> tuple[int, int]:
            return (min(ny - 1, max(0, int(py * ny))), min(n - 1, max(0, int(px * n))))

        def field(px: float, py: float) -> tuple[float, float]:
            x, y = xa + px * (xb - xa), ya + py * (yb - ya)
            try:
                u, v = self.f(x, y)
            except (ValueError, ZeroDivisionError, OverflowError):
                return 0.0, 0.0
            u, v = float(u) / (xb - xa), float(v) / (yb - ya)
            s = math.hypot(u, v)
            if not math.isfinite(s) or s < 1e-12:
                return 0.0, 0.0
            return u / s, v / s

        step = 0.3 / max(n, ny)

        def integrate(px: float, py: float, direction: float, own: list[tuple[int, int]]) -> list[tuple[float, float]]:
            pts = []
            visited = set(own)
            for _ in range(int(6.0 / step)):
                k1 = field(px, py)
                if k1 == (0.0, 0.0):
                    break
                mx, my = px + 0.5 * step * direction * k1[0], py + 0.5 * step * direction * k1[1]
                k2 = field(mx, my)
                if k2 == (0.0, 0.0):
                    break
                px, py = px + step * direction * k2[0], py + step * direction * k2[1]
                if not (0 <= px <= 1 and 0 <= py <= 1):
                    break
                c = cell(px, py)
                if (occupied[c] or c in visited) and c not in own[-6:]:
                    break  # ran into another line, or closed on itself
                if not own or own[-1] != c:
                    own.append(c)
                    visited.add(c)
                pts.append((px, py))
            return pts

        rng = np.random.default_rng(self.seed)
        seeds = [(i, j) for i in range(ny) for j in range(n)]
        rng.shuffle(seeds)
        lines = []
        for i, j in seeds:
            if occupied[i, j]:
                continue
            px, py = (j + 0.5) / n, (i + 0.5) / ny
            start = [cell(px, py)]
            back = integrate(px, py, -1.0, list(start))
            # going forward, the backward part is a wall except right at the seed
            fwd = integrate(px, py, 1.0, [cell(*q) for q in back[::-1]] + start)
            traj = back[::-1] + [(px, py)] + fwd
            if len(traj) * step < self.min_length:
                continue
            for q in traj:
                occupied[cell(*q)] = True
            arr = np.array(traj)
            arr[:, 0] = xa + arr[:, 0] * (xb - xa)
            arr[:, 1] = ya + arr[:, 1] * (yb - ya)
            lines.append(arr)
        return lines

    def legend_sample(self, ax: "Axes", ctx: "RenderContext") -> str:
        color = color_text(self.color) if self.color is not None else f"{ctx.theme.ink}!70"
        return rf"\draw[{color}, line width={self.line_width:g}pt, ->] (0,0) -- (0.6,0);"

    def render(self, ax: "Axes", ctx: "RenderContext") -> list[str]:
        out = []
        trajs = self._trajectories(ax)
        speeds = []
        for t in trajs:
            sp = []
            for x, y in t[:: max(1, len(t) // 10)]:
                try:
                    u, v = self.f(x, y)
                    sp.append(math.hypot(u, v))
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass
            speeds.append(float(np.mean(sp)) if sp else 0.0)
        lo, hi = (min(speeds), max(speeds)) if speeds else (0, 1)
        for t, s in zip(trajs, speeds):
            cx, cy = ax._to_canvas_arrays(t[:, 0], t[:, 1])
            if self.cmap is not None:
                color = self.cmap(0.5 if hi == lo else (s - lo) / (hi - lo)).expr
            else:
                color = color_text(self.color) if self.color is not None else f"{ctx.theme.ink}!70"
            for piece in clip_polyline(cx, cy, ax._box()):
                pts = _simplify(piece, 0.004)
                if len(pts) < 2:
                    continue
                tokens = [_c(*pts[0])]
                for p in pts[1:]:
                    tokens += ["--", _c(*p)]
                deco = (
                    "postaction={decorate}, decoration={markings, mark=at position 0.5 with "
                    f"{{\\arrow{{Stealth[length={self.arrow_size:g}pt, width={self.arrow_size * 0.8:g}pt]}}}}}}"
                )
                out += wrap_tokens(f"\\draw[{color}, line width={self.line_width:g}pt, {deco}]", tokens)
        return out


# ---------------------------------------------------------------------------
# the axes
# ---------------------------------------------------------------------------

class Axes(Element):
    """A coordinate system for plotting data.

    Parameters
    ----------
    at:
        Picture coordinate of the lower-left corner of the plotting area.
    width, height:
        Size of the plotting area in cm.
    xlim, ylim:
        Axis limits; computed from the data when omitted.
    xlabel, ylabel, title:
        Labels (LaTeX).
    frame:
        ``"open"`` (left/bottom spines, default), ``"box"``, ``"cross"`` or ``"none"``.
    grid:
        ``True`` for light grid lines at the major ticks.
    xticks, yticks:
        ``"auto"``, ``None`` (no ticks), a list of values, a list of
        ``(value, label)`` pairs, a dict, a step size or ``"pi"``.
    xscale, yscale:
        ``"linear"`` or ``"log"``.
    equal:
        Use the same scale on both axes (adjusts the limits).
    margins:
        Fraction of the data range added around autoscaled limits.
    """

    def __init__(
        self,
        at: Any = (0, 0),
        width: float = 8.0,
        height: float = 5.0,
        *,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        xlabel: Any = None,
        ylabel: Any = None,
        title: Any = None,
        frame: str = "open",
        grid: bool | str = False,
        xticks: Any = "auto",
        yticks: Any = "auto",
        xticklabels: bool = True,
        yticklabels: bool = True,
        xscale: str = "linear",
        yscale: str = "linear",
        equal: bool = False,
        margins: float | tuple[float, float] = (0.04, 0.06),
        name: str | None = None,
        background: Any = None,
        tick_font: str = r"\footnotesize",
        label_font: str = r"\small",
        title_font: str = r"\small\bfseries",
    ):
        if frame not in ("open", "box", "cross", "none"):
            raise ValueError("frame must be 'open', 'box', 'cross' or 'none'")
        p = Point(*at) if not isinstance(at, Point) else at
        self.x0, self.y0 = p.x, p.y
        self.width, self.height = float(width), float(height)
        self._xlim_user, self._ylim_user = xlim, ylim
        self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
        self.frame = frame
        self.grid = grid
        self.xticks, self.yticks = xticks, yticks
        self.xticklabels, self.yticklabels = xticklabels, yticklabels
        self.xscale, self.yscale = xscale, yscale
        self.equal = equal
        self.margins = margins if isinstance(margins, tuple) else (margins, margins)
        self._name = name
        self.background = background
        self.tick_font, self.label_font, self.title_font = tick_font, label_font, title_font
        self.artists: list[_Artist] = []
        self._colors_used = 0
        self._legend: dict[str, Any] | None = None
        self._colorbar: dict[str, Any] | None = None
        self.xlim: tuple[float, float] = (0.0, 1.0)
        self.ylim: tuple[float, float] = (0.0, 1.0)

    # -- helpers --------------------------------------------------------------------
    def _next_color(self) -> int:
        i = self._colors_used
        self._colors_used += 1
        return i

    def _fx(self, v: Any) -> Any:
        return np.log10(v) if self.xscale == "log" else v

    def _fy(self, v: Any) -> Any:
        return np.log10(v) if self.yscale == "log" else v

    def _to_canvas_arrays(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(all="ignore"):
            fx0, fx1 = self._fx(self.xlim[0]), self._fx(self.xlim[1])
            fy0, fy1 = self._fy(self.ylim[0]), self._fy(self.ylim[1])
            cx = self.x0 + (self._fx(np.asarray(xs, dtype=float)) - fx0) / (fx1 - fx0) * self.width
            cy = self.y0 + (self._fy(np.asarray(ys, dtype=float)) - fy0) / (fy1 - fy0) * self.height
        return cx, cy

    def _to_canvas(self, x: float, y: float) -> Point:
        cx, cy = self._to_canvas_arrays(np.array([x]), np.array([y]))
        return Point(float(cx[0]), float(cy[0]))

    def _box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x0 + self.width, self.y0 + self.height)

    def _floor(self, values: Any, canvas: np.ndarray, axis: str = "y") -> np.ndarray:
        """On log axes, move non-positive values just past the low edge
        (instead of -inf/NaN) so that fills and bars reach the axis."""
        if (self.yscale if axis == "y" else self.xscale) != "log":
            return canvas
        edge = (self.y0 if axis == "y" else self.x0) - 1.0
        return np.where(np.asarray(values, dtype=float) <= 0, edge, canvas)

    # -- public coordinate helpers ------------------------------------------------------
    def point(self, x: float, y: float) -> LazyCoord:
        """The picture coordinate of data point ``(x, y)`` (resolved at render time)."""
        return LazyCoord(lambda: self._to_canvas(x, y))

    @property
    def lower_left(self) -> Point:
        return Point(self.x0, self.y0)

    @property
    def upper_right(self) -> Point:
        return Point(self.x0 + self.width, self.y0 + self.height)

    @property
    def center(self) -> Point:
        return Point(self.x0 + self.width / 2, self.y0 + self.height / 2)

    # -- plotting ---------------------------------------------------------------------------
    def plot(
        self,
        x: Any,
        y: Any = None,
        *,
        color: Any = None,
        line_width: float = 1.3,
        dash: str | None = None,
        label: Any = None,
        smooth: bool = False,
        marker: str | None = None,
        marker_size: float = 2.0,
        opacity: float | None = None,
        arrow: str | None = None,
        options: Any = None,
    ) -> "Axes":
        """Draw a line through the points ``(x[i], y[i])``.

        ``dash`` accepts TikZ names (``"dashed"``) or matplotlib style
        (``"--"``, ``":"``, ``"-."``).
        """
        if y is None:
            y = x
            x = np.arange(len(_as_array(y)))
        self.artists.append(
            _Line(self, x, y, color=color, line_width=line_width, dash=dash, label=label, smooth=smooth,
                  marker=marker, marker_size=marker_size, opacity=opacity, arrow=arrow, options=options)
        )
        return self

    def function(
        self,
        f: Callable[[Any], Any],
        domain: tuple[float, float] | None = None,
        *,
        samples: int = 300,
        **kwargs: Any,
    ) -> "Axes":
        """Plot ``y = f(x)`` over ``domain`` (default: the x limits).

        ``f`` may be vectorised (``np.sin``) or scalar (``math.sin``);
        undefined points (``nan``/``inf``) break the curve, and so do poles
        (jumps across the whole visible y range, as in ``tan x``) once the y
        limits are set.
        """
        if domain is None:
            if self._xlim_user is None:
                raise ValueError("give a domain or set xlim before plotting a function")
            domain = self._xlim_user
        a, b = (float(v) for v in domain)
        if samples < 2:
            raise ValueError("samples must be at least 2")
        if self.xscale == "log":
            if b <= 0:
                raise ValueError("a log x axis needs a domain with positive values")
            a = a if a > 0 else b * 1e-6
            xs = np.logspace(math.log10(a), math.log10(b), samples)
        else:
            xs = np.linspace(a, b, samples)
        ys = _eval_function(f, xs)
        self.plot(xs, ys, **kwargs)
        self.artists[-1].break_jumps = True  # type: ignore[attr-defined]
        return self

    def parametric(self, f: Callable[[Any], Any], t_range: tuple[float, float], *, samples: int = 400,
                   **kwargs: Any) -> "Axes":
        """Plot the curve ``t -> (x(t), y(t))``."""
        ts = np.linspace(t_range[0], t_range[1], samples)
        pts = [f(t) for t in ts]
        return self.plot([p[0] for p in pts], [p[1] for p in pts], **kwargs)

    def scatter(
        self,
        x: Any,
        y: Any,
        *,
        color: Any = None,
        c: Any = None,
        cmap: Any = None,
        vmin: float | None = None,
        vmax: float | None = None,
        marker: str = "o",
        size: float = 2.2,
        edgecolor: Any = "white",
        label: Any = None,
        opacity: float | None = None,
    ) -> "Axes":
        """Markers at ``(x, y)``; ``c`` maps values through ``cmap``.

        Markers: ``"o"``, ``"s"``, ``"^"``, ``"v"``, ``"d"``, ``"x"``, ``"+"``, ``"*"``.
        ``size`` is the marker radius in points.
        """
        self.artists.append(
            _Scatter(self, x, y, color=color, values=c, cmap=cmap, vmin=vmin, vmax=vmax, marker=marker,
                     size=size, edgecolor=edgecolor, label=label, opacity=opacity)
        )
        return self

    def fill_between(
        self,
        x: Any,
        y1: Any,
        y2: Any = 0.0,
        *,
        color: Any = None,
        opacity: float = 0.25,
        label: Any = None,
        line: bool = False,
        options: Any = None,
    ) -> "Axes":
        """Shade the region between two curves (``y2`` may be a constant)."""
        self.artists.append(_Fill(self, x, y1, y2, color=color, opacity=opacity, label=label, line=line,
                                  options=options))
        return self

    def area(self, f: Callable[[Any], Any], a: float, b: float, *, baseline: float = 0.0, samples: int = 200,
             **kwargs: Any) -> "Axes":
        """Shade the area between ``f`` and ``baseline`` for ``a <= x <= b``."""
        xs = np.linspace(a, b, samples)
        return self.fill_between(xs, _eval_function(f, xs), baseline, **kwargs)

    def bar(
        self,
        x: Any,
        heights: Any,
        *,
        width: float = 0.8,
        bottom: Any = 0.0,
        color: Any = None,
        label: Any = None,
        opacity: float | None = None,
        edge: bool = True,
        horizontal: bool = False,
    ) -> "Axes":
        """Bars at positions ``x`` (``color`` may be a list, one per bar)."""
        self.artists.append(_Bars(self, x, heights, width=width, bottom=bottom, color=color, label=label,
                                  opacity=opacity, edge=edge, horizontal=horizontal))
        return self

    def hline(self, y: float, *, color: Any = None, line_width: float = 0.8, dash: str | None = "dashed",
              label: Any = None, xmin: float | None = None, xmax: float | None = None, options: Any = None) -> "Axes":
        """A horizontal reference line."""
        self.artists.append(_RefLine("h", y, color=color, line_width=line_width, dash=dash, label=label,
                                     options=options, extent=(xmin, xmax)))
        return self

    def vline(self, x: float, *, color: Any = None, line_width: float = 0.8, dash: str | None = "dashed",
              label: Any = None, ymin: float | None = None, ymax: float | None = None, options: Any = None) -> "Axes":
        """A vertical reference line."""
        self.artists.append(_RefLine("v", x, color=color, line_width=line_width, dash=dash, label=label,
                                     options=options, extent=(ymin, ymax)))
        return self

    def axline(self, point: tuple[float, float], slope: float, *, color: Any = None, line_width: float = 1.0,
               dash: str | None = None, label: Any = None, options: Any = None) -> "Axes":
        """An infinite straight line through ``point`` with ``slope``."""
        self.artists.append(_RefLine("s", (point, slope), color=color, line_width=line_width, dash=dash,
                                     label=label, options=options))
        return self

    def text(self, x: float, y: float, text: Any, **options: Any) -> "Axes":
        """Text at data point ``(x, y)`` (node options as keywords)."""
        self.artists.append(_Text(x, y, text, Options(**options)))
        return self

    def annotate(self, text: Any, xy: tuple[float, float], xytext: tuple[float, float], *, arrow: Any = True,
                 arrow_options: Any = None, **options: Any) -> "Axes":
        """Text at ``xytext`` with an arrow pointing to ``xy`` (data coordinates)."""
        self.artists.append(_Annotation(text, xy, xytext, arrow, Options(**options), arrow_options))
        return self

    def arrow(self, start: tuple[float, float], end: tuple[float, float], *, tip: str = "->",
              autoscale: bool = True, **options: Any) -> "Axes":
        """An arrow between two data points (e.g. a vector)."""
        opts = Options(tip, **options)
        self.artists.append(_DataPath([start, end], False, "draw", opts, clip=False, autoscale=autoscale))
        return self

    def line(self, points: Sequence[Any], *, label: Any = None, clip: bool = True, **options: Any) -> "Axes":
        """A polyline through data points, with any TikZ options."""
        self.artists.append(_DataPath(points, False, "draw", Options(**options), label=label, clip=clip))
        return self

    def polygon(self, points: Sequence[Any], *, label: Any = None, command: str = "filldraw", clip: bool = True,
                **options: Any) -> "Axes":
        """A closed polygon in data coordinates (filled and outlined by default)."""
        self.artists.append(_DataPath(points, True, command, Options(**options), label=label, clip=clip))
        return self

    def imshow(self, Z: Any, extent: tuple[float, float, float, float] | None = None, *, cmap: Any = "viridis",
               vmin: float | None = None, vmax: float | None = None, opacity: float | None = None) -> "Axes":
        """Color the cells of a 2D array (row 0 at the bottom).

        ``extent = (x0, x1, y0, y1)`` defaults to cell centres at integers.
        """
        Z = np.asarray(Z, dtype=float)
        if extent is None:
            ny, nx = Z.shape
            extent = (-0.5, nx - 0.5, -0.5, ny - 0.5)
        art = _Image(Z, extent, cmap, vmin, vmax, opacity)
        self.artists.append(art)
        self._last_mappable = ("image", art)
        return self

    def contour(
        self,
        f: Any,
        x: Any = None,
        y: Any = None,
        *,
        levels: Any = None,
        cmap: Any = None,
        colors: Any = None,
        line_width: float = 0.8,
        dash: str | None = None,
        resolution: int = 120,
        label: Any = None,
    ) -> "Axes":
        """Contour lines of ``f(x, y)`` (a function or a 2D array with axes ``x``, ``y``).

        ``levels`` is a count or a list of values. Lines are colored with
        ``cmap`` (default viridis) unless ``colors`` is given.
        """
        self.artists.append(_Contour(self, f, x, y, levels=levels, filled=False, cmap=cmap, colors=colors,
                                     line_width=line_width, opacity=None, resolution=resolution, dash=dash,
                                     lines_over=False, line_color=None, label=label))
        return self

    def contourf(
        self,
        f: Any,
        x: Any = None,
        y: Any = None,
        *,
        levels: Any = None,
        cmap: Any = "viridis",
        colors: Any = None,
        opacity: float | None = None,
        lines: bool = True,
        line_color: Any = None,
        line_width: float = 0.4,
        resolution: int = 150,
    ) -> "Axes":
        """Filled contours of ``f(x, y)``; ``lines`` also draws the level lines."""
        art = _Contour(self, f, x, y, levels=levels, filled=True, cmap=cmap, colors=colors,
                       line_width=line_width, opacity=opacity, resolution=resolution, dash=None,
                       lines_over=lines, line_color=line_color, label=None)
        self.artists.append(art)
        self._last_mappable = ("contour", art)
        return self

    def regions(
        self,
        f: Callable[[Any, Any], Any],
        colors: Sequence[Any] = ("Blue!25", "Orange!25", "Green!25", "Red!25", "Purple!25"),
        *,
        opacity: float = 1.0,
        resolution: int = 200,
        boundary: bool = False,
        boundary_color: Any = None,
        boundary_width: float = 0.8,
    ) -> "Axes":
        """Shade the regions where ``f(x, y)`` (a class index) is constant.

        Handy for decision regions of classifiers and Voronoi cells. Needs
        explicit or data-derived limits; ``boundary=True`` also draws the
        class boundaries.
        """
        self.artists.append(_Regions(f, colors, opacity, resolution, boundary, boundary_color, boundary_width))
        return self

    def quiver(
        self,
        X: Any,
        Y: Any,
        U: Any,
        V: Any,
        *,
        scale: float | None = None,
        normalize: bool = False,
        color: Any = None,
        cmap: Any = None,
        line_width: float = 0.7,
        pivot: str = "middle",
        length: float = 0.45,
        tip: str = "-{Stealth[length=3.5pt, width=3pt]}",
        label: Any = None,
    ) -> "Axes":
        """Arrows ``(U, V)`` at points ``(X, Y)``.

        By default the longest arrow is ``length`` cm; ``normalize=True``
        draws all arrows with that length (combine with ``cmap`` to show the
        magnitude by color).
        """
        self.artists.append(_Quiver(X, Y, U, V, scale=scale, normalize=normalize, color=color, cmap=cmap,
                                    line_width=line_width, pivot=pivot, length=length, label=label, tip=tip))
        return self

    def vector_field(self, f: Callable[[float, float], tuple[float, float]], *, n: int = 15, **kwargs: Any) -> "Axes":
        """Quiver plot of ``f(x, y) -> (u, v)`` on an ``n``-wide grid over the limits."""
        if self._xlim_user is None or self._ylim_user is None:
            raise ValueError("vector_field needs explicit xlim and ylim")
        (xa, xb), (ya, yb) = self._xlim_user, self._ylim_user
        ny = max(2, int(round(n * self.height / self.width)))
        dx, dy = (xb - xa) / n, (yb - ya) / ny
        xs = np.linspace(xa + dx / 2, xb - dx / 2, n)
        ys = np.linspace(ya + dy / 2, yb - dy / 2, ny)
        X, Y = np.meshgrid(xs, ys)
        UV = [f(float(x), float(y)) for x, y in zip(X.ravel(), Y.ravel())]
        U = np.array([uv[0] for uv in UV], dtype=float)
        V = np.array([uv[1] for uv in UV], dtype=float)
        kwargs.setdefault("length", 0.7 * min(self.width / n, self.height / ny))
        return self.quiver(X.ravel(), Y.ravel(), U, V, **kwargs)

    def streamlines(
        self,
        f: Callable[[float, float], tuple[float, float]],
        *,
        density: float = 1.0,
        color: Any = None,
        cmap: Any = None,
        line_width: float = 0.8,
        arrow_size: float = 4.5,
        min_length: float = 0.08,
        seed: int = 1,
        label: Any = None,
    ) -> "Axes":
        """Streamlines of the vector field ``f(x, y) -> (u, v)`` over the axis limits."""
        self.artists.append(_Streamlines(f, density=density, color=color, cmap=cmap, line_width=line_width,
                                         arrow_size=arrow_size, min_length=min_length, label=label, seed=seed))
        return self

    def legend(self, location: str = "best", *, columns: int = 1, frame: bool = True, font: str = r"\footnotesize",
               **options: Any) -> "Axes":
        """Show a legend of the labelled artists.

        ``location``: ``"best"``, ``"upper right"``, ``"upper left"``,
        ``"lower right"``, ``"lower left"``, ``"upper center"``, ...,
        ``"outside right"`` or ``"above"`` (one row above the plot).
        """
        self._legend = {"location": location, "columns": columns, "frame": frame, "font": font,
                        "options": Options(**options)}
        return self

    def colorbar(self, *, label: Any = None, width: float = 0.3, gap: float = 0.3, ticks: Any = "auto") -> "Axes":
        """A color bar for the last ``imshow``/``contourf`` to the right of the axes."""
        if not hasattr(self, "_last_mappable"):
            raise ValueError("colorbar needs an imshow() or contourf() first")
        self._colorbar = {"label": label, "width": width, "gap": gap, "ticks": ticks}
        return self

    # -- limits and ticks -------------------------------------------------------------------
    def _autoscale(self) -> None:
        data = [d for d in (a.data_points() for a in self.artists) if d is not None]

        def collect(axis: int, log: bool) -> tuple[float, float] | None:
            lo, hi = math.inf, -math.inf
            for d in data:
                v = np.asarray(d[axis], dtype=float).ravel()
                v = v[np.isfinite(v)]
                if log:
                    v = v[v > 0]  # log axes ignore non-positive values
                if v.size:
                    lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
            if lo > hi:
                return None
            return (math.log10(lo), math.log10(hi)) if log else (lo, hi)

        def finish(rng: tuple[float, float] | None, margin: float, inv: Callable[[float], float]) -> tuple[float, float]:
            if rng is None:
                return (inv(0.0), inv(1.0))
            lo, hi = rng
            if hi == lo:
                d = abs(lo) * 0.1 or 1.0
                lo, hi = lo - d, hi + d
            span = hi - lo
            return inv(lo - margin * span), inv(hi + margin * span)

        logx, logy = self.xscale == "log", self.yscale == "log"
        ix = (lambda v: 10.0 ** v) if logx else (lambda v: v)
        iy = (lambda v: 10.0 ** v) if logy else (lambda v: v)
        if self._xlim_user is not None:
            self.xlim = tuple(map(float, self._xlim_user))  # type: ignore[assignment]
        else:
            self.xlim = finish(collect(0, logx), self.margins[0], ix)
        if self._ylim_user is not None:
            self.ylim = tuple(map(float, self._ylim_user))  # type: ignore[assignment]
        else:
            self.ylim = finish(collect(1, logy), self.margins[1], iy)
        for name, lim, log in (("x", self.xlim, logx), ("y", self.ylim, logy)):
            if log and min(lim) <= 0:
                raise ValueError(f"log {name} axis needs positive limits, got {lim}")
        if self.equal and not (logx or logy):
            sx = (self.xlim[1] - self.xlim[0]) / self.width
            sy = (self.ylim[1] - self.ylim[0]) / self.height
            if sx > sy and self._ylim_user is None:
                c, half = sum(self.ylim) / 2, sx * self.height / 2
                self.ylim = (c - half, c + half)
            elif sy > sx and self._xlim_user is None:
                c, half = sum(self.xlim) / 2, sy * self.width / 2
                self.xlim = (c - half, c + half)

    def _ticks(self, spec: Any, lim: tuple[float, float], scale: str, length: float) -> list[tuple[float, str]]:
        lo, hi = min(lim), max(lim)
        if spec is None or spec is False:
            return []
        if isinstance(spec, dict):
            return [(float(k), label_text(v)) for k, v in spec.items() if lo - 1e-9 <= float(k) <= hi + 1e-9]
        if isinstance(spec, str):
            if spec == "pi":
                return pi_ticks(lo, hi)
            if spec != "auto":
                raise ValueError(f"unknown tick specification {spec!r}")
        elif isinstance(spec, numbers.Real) and not isinstance(spec, bool):
            values = linear_ticks(lo, hi, step=float(spec))
            return [(v, format_tick(v, float(spec))) for v in values]
        else:  # any iterable of values and/or (value, label) pairs
            items = list(spec)
            out = []
            vals = [float(v) for v in items if not isinstance(v, (tuple, list))]
            step = nice_step(hi - lo, 6) if vals else 1.0
            if len(vals) >= 2:
                diffs = [abs(b - a) for a, b in zip(sorted(vals), sorted(vals)[1:]) if b != a]
                step = min(diffs) if diffs else step
            for v in items:
                if isinstance(v, (tuple, list)):
                    out.append((float(v[0]), label_text(v[1])))
                else:
                    out.append((float(v), format_tick(float(v), step)))
            return out
        # auto
        if scale == "log":
            return [(v, format_log_tick(v)) for v in log_ticks(lo, hi)]
        target = max(3, int(round(length / 1.4)))
        step = nice_step(hi - lo, target)
        return [(v, format_tick(v, step)) for v in linear_ticks(lo, hi, step=step)]

    # -- rendering -----------------------------------------------------------------------------
    def render(self, ctx: "RenderContext") -> list[str]:
        self._autoscale()
        name = self._name or ctx.auto_name(self, "ax")
        if self._name:
            ctx.claim(self._name, self)
        theme = ctx.theme
        x0, y0, x1, y1 = self._box()
        xt = self._ticks(self.xticks, self.xlim, self.xscale, self.width)
        yt = self._ticks(self.yticks, self.ylim, self.yscale, self.height)
        xt = [(self._to_canvas(v, self.ylim[0]).x, lab) for v, lab in xt]
        yt = [(self._to_canvas(self.xlim[0], v).y, lab) for v, lab in yt]
        eps = 1e-6
        xt = [(p, lab) for p, lab in xt if x0 - eps <= p <= x1 + eps]
        yt = [(p, lab) for p, lab in yt if y0 - eps <= p <= y1 + eps]
        axis_color = f"{theme.ink}!80"
        out = [f"% axes {name}"]

        if self.background is not None:
            out.append(f"\\fill[{self.background}] {_c(x0, y0)} rectangle {_c(x1, y1)};")

        # grid
        if self.grid:
            segs = []
            for p, _ in xt:
                segs.append(f"{_c(p, y0)} -- {_c(p, y1)}")
            for p, _ in yt:
                segs.append(f"{_c(x0, p)} -- {_c(x1, p)}")
            if segs:
                out += wrap_tokens(f"\\draw[{theme.grid_color}, line width=0.4pt]", " ".join(segs).split(" "))

        # origin for the cross frame
        ox = self._to_canvas(0.0, 0.0).x if self.xlim[0] <= 0 <= self.xlim[1] and self.xscale == "linear" else x0
        oy = self._to_canvas(0.0, 0.0).y if self.ylim[0] <= 0 <= self.ylim[1] and self.yscale == "linear" else y0

        # data, back to front
        ordered = sorted(self.artists, key=lambda a: a.zorder)
        clipped = [a for a in ordered if a.clip]
        unclipped = [a for a in ordered if not a.clip]
        if clipped:
            out.append(r"\begin{scope}")
            out.append(f"  \\clip {_c(x0, y0)} rectangle {_c(x1, y1)};")
            for art in clipped:
                out += ["  " + ln for ln in art.render(self, ctx)]
            out.append(r"\end{scope}")

        # frame, ticks and tick labels
        tick = 0.1
        tl = f"font={self.tick_font}, text={theme.ink}!85"
        xt_names, yt_names = [], []
        if self.frame in ("open", "box"):
            if self.frame == "box":
                out.append(f"\\draw[{axis_color}, line width=0.5pt] {_c(x0, y0)} rectangle {_c(x1, y1)};")
            else:
                out.append(f"\\draw[{axis_color}, line width=0.5pt] {_c(x0, y1)} -- {_c(x0, y0)} -- {_c(x1, y0)};")
            inward = self.frame == "box"
            d = tick if inward else -tick
            segs = [f"{_c(p, y0)} -- {_c(p, y0 + d)}" for p, _ in xt] + [f"{_c(x0, p)} -- {_c(x0 + d, p)}" for p, _ in yt]
            if segs:
                out += wrap_tokens(f"\\draw[{axis_color}, line width=0.5pt]", " ".join(segs).split(" "))
            lab_y = y0 - (0.06 if inward else tick + 0.04)
            lab_x = x0 - (0.06 if inward else tick + 0.04)
            if self.xticklabels:
                for i, (p, lab) in enumerate(xt):
                    nm = f"{name}-xt{i}"
                    xt_names.append(nm)
                    out.append(f"\\node[anchor=north, {tl}, inner sep=2pt] ({nm}) at {_c(p, lab_y)} {{{lab}}};")
            if self.yticklabels:
                for i, (p, lab) in enumerate(yt):
                    nm = f"{name}-yt{i}"
                    yt_names.append(nm)
                    out.append(f"\\node[anchor=east, {tl}, inner sep=2pt] ({nm}) at {_c(lab_x, p)} {{{lab}}};")
        elif self.frame == "cross":
            over = 0.3
            out.append(f"\\draw[->, {axis_color}, line width=0.6pt] {_c(x0, oy)} -- {_c(x1 + over, oy)};")
            out.append(f"\\draw[->, {axis_color}, line width=0.6pt] {_c(ox, y0)} -- {_c(ox, y1 + over)};")
            segs = [f"{_c(p, oy - tick / 2)} -- {_c(p, oy + tick / 2)}" for p, _ in xt if abs(p - ox) > 1e-6]
            segs += [f"{_c(ox - tick / 2, p)} -- {_c(ox + tick / 2, p)}" for p, _ in yt if abs(p - oy) > 1e-6]
            if segs:
                out += wrap_tokens(f"\\draw[{axis_color}, line width=0.5pt]", " ".join(segs).split(" "))
            if self.xticklabels:
                for i, (p, lab) in enumerate(xt):
                    if abs(p - ox) < 1e-6:
                        continue
                    nm = f"{name}-xt{i}"
                    xt_names.append(nm)
                    out.append(f"\\node[anchor=north, {tl}, inner sep=2pt] ({nm}) at {_c(p, oy - tick / 2)} {{{lab}}};")
            if self.yticklabels:
                for i, (p, lab) in enumerate(yt):
                    if abs(p - oy) < 1e-6:
                        continue
                    nm = f"{name}-yt{i}"
                    yt_names.append(nm)
                    out.append(f"\\node[anchor=east, {tl}, inner sep=2pt] ({nm}) at {_c(ox - tick / 2, p)} {{{lab}}};")
            if (xt or yt) and (self.xticklabels or self.yticklabels) and abs(ox - x0) > 1e-6 and abs(oy - y0) > 1e-6:
                out.append(f"\\node[anchor=north east, {tl}, inner sep=2pt] at {_c(ox - 0.02, oy - 0.02)} {{0}};")

        # unclipped artists (text, annotations) above the frame
        for art in unclipped:
            out += art.render(self, ctx)

        # axis labels
        lf = f"font={self.label_font}, align=center"
        if self.frame == "cross":
            if self.xlabel is not None:
                out.append(f"\\node[anchor=west, {lf}, inner sep=3pt] at {_c(x1 + 0.3, oy)} {{{label_text(self.xlabel)}}};")
            if self.ylabel is not None:
                out.append(f"\\node[anchor=south, {lf}, inner sep=3pt] at {_c(ox, y1 + 0.3)} {{{label_text(self.ylabel)}}};")
        else:
            xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
            if self.xlabel is not None:
                if xt_names:
                    out.append(f"\\node[fit={''.join(f'({n})' for n in xt_names)}, inner sep=0pt] ({name}-xticks) {{}};")
                    out.append(f"\\node[anchor=north, {lf}, inner sep=4pt] at ({fmt(xc, PREC)},0 |- {name}-xticks.south) {{{label_text(self.xlabel)}}};")
                else:
                    out.append(f"\\node[anchor=north, {lf}, inner sep=4pt] at {_c(xc, y0 - 0.1)} {{{label_text(self.xlabel)}}};")
            if self.ylabel is not None:
                if yt_names:
                    out.append(f"\\node[fit={''.join(f'({n})' for n in yt_names)}, inner sep=0pt] ({name}-yticks) {{}};")
                    out.append(f"\\node[anchor=south, rotate=90, {lf}, inner sep=4pt] at ({name}-yticks.west |- 0,{fmt(yc, PREC)}) {{{label_text(self.ylabel)}}};")
                else:
                    out.append(f"\\node[anchor=south, rotate=90, {lf}, inner sep=4pt] at {_c(x0 - 0.1, yc)} {{{label_text(self.ylabel)}}};")
        if self.title is not None:
            top = y1 + (0.35 if self.frame == "cross" and self.ylabel is not None else 0.05)
            out.append(f"\\node[anchor=south, font={self.title_font}, inner sep=4pt, align=center] at {_c((x0 + x1) / 2, top)} {{{label_text(self.title)}}};")

        if self._colorbar is not None:
            out += self._render_colorbar(ctx, name)
        if self._legend is not None:
            out += self._render_legend(ctx)
        return out

    def _render_legend(self, ctx: "RenderContext") -> list[str]:
        entries = []
        for art in self.artists:
            if art.label is None:
                continue
            sample = art.legend_sample(self, ctx)
            if sample:
                entries.append((sample, art.label))
        if not entries:
            return []
        cfg = self._legend
        assert cfg is not None
        loc = cfg["location"]
        x0, y0, x1, y1 = self._box()
        pad = 0.12
        if loc == "best":
            loc = self._best_location()
        if loc == "outside right":
            anchor, at = "north west", (x1 + 0.25, y1)
        elif loc == "above":
            anchor, at = "south", ((x0 + x1) / 2, y1 + 0.1)
        else:
            if loc not in _LEGEND_LOCATIONS:
                raise ValueError(f"unknown legend location {loc!r}")
            anchor, (fx, fy) = _LEGEND_LOCATIONS[loc]
            ax_ = x0 + fx * self.width + (pad if fx == 0 else -pad if fx == 1 else 0)
            ay_ = y0 + fy * self.height + (pad if fy == 0 else -pad if fy == 1 else 0)
            at = (ax_, ay_)
        columns = len(entries) if loc == "above" and cfg["columns"] == 1 else cfg["columns"]
        colspec = "@{\\hspace{10pt}}".join(["l@{\\hspace{4pt}}l"] * columns)
        rows = []
        for i in range(0, len(entries), columns):
            chunk = entries[i : i + columns]
            cells = [f"\\tikz[baseline=-0.6ex]{{{s}}} & {inline_text(lab)}" for s, lab in chunk]
            rows.append(" & ".join(cells) + r" \\")
        opts = Options(anchor=anchor, font=cfg["font"], inner_sep="4pt", align="left")
        if cfg["frame"]:
            opts.set("fill", "white" if not ctx.theme.paper else ctx.theme.paper)
            opts.set("draw", f"{ctx.theme.ink}!18")
            opts.set("rounded corners", "2pt")
            opts.set("fill opacity", 0.92)
            opts.set("text opacity", 1)
        opts.add(cfg["options"])
        lines = [f"\\node{opts.render()} at {_c(*at)} {{%"]
        lines.append(f"  \\begin{{tabular}}{{@{{}}{colspec}@{{}}}}")
        lines += ["    " + r for r in rows]
        lines.append(r"  \end{tabular}%")
        lines.append("};")
        return lines

    def _best_location(self) -> str:
        """Pick the corner that overlaps the least data."""
        x0, y0, x1, y1 = self._box()
        counts = {}
        regions = {
            "upper right": (0.55, 1.0, 0.6, 1.0),
            "upper left": (0.0, 0.45, 0.6, 1.0),
            "lower right": (0.55, 1.0, 0.0, 0.4),
            "lower left": (0.0, 0.45, 0.0, 0.4),
        }
        pts_x, pts_y = [], []
        for art in self.artists:
            occupied = art.occupied(self)
            if occupied is None:
                continue
            xs, ys = (np.asarray(v, dtype=float).ravel() for v in occupied)
            if xs.shape != ys.shape:
                continue
            cx, cy = self._to_canvas_arrays(xs, ys)
            pts_x.append((cx - x0) / self.width)
            pts_y.append((cy - y0) / self.height)
        if not pts_x:
            return "upper right"
        px, py = np.concatenate(pts_x), np.concatenate(pts_y)
        for loc, (a, b, c, d) in regions.items():
            counts[loc] = int(np.sum((px >= a) & (px <= b) & (py >= c) & (py <= d)))
        order = ["upper right", "upper left", "lower right", "lower left"]
        return min(order, key=lambda k: (counts[k], order.index(k)))

    def _render_colorbar(self, ctx: "RenderContext", name: str) -> list[str]:
        cfg = self._colorbar
        assert cfg is not None
        kind, art = self._last_mappable
        x0, y0, x1, y1 = self._box()
        bx0 = x1 + cfg["gap"]
        bx1 = bx0 + cfg["width"]
        out = ["% colorbar"]
        if kind == "image":
            lo, hi, cmap = art.vmin, art.vmax, art.cmap
            n = 48
            for i in range(n):
                ya = y0 + self.height * i / n
                yb = y0 + self.height * (i + 1) / n + (0.004 if i < n - 1 else 0)
                color = cmap((i + 0.5) / n).expr
                out.append(f"\\fill[{color}] {_c(bx0, ya)} rectangle {_c(bx1, yb)};")
        else:
            x, y, Z = art.grid(self)
            levels = art._levels(Z)
            colors = art.level_colors(levels, True)
            lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))
            # band k lies between consecutive edges; levels outside the data
            # range give empty bands
            edges = [lo] + [min(hi, max(lo, v)) for v in levels] + [hi]
            for i, color in enumerate(colors):
                a = y0 + self.height * (edges[i] - lo) / ((hi - lo) or 1)
                b = y0 + self.height * (edges[i + 1] - lo) / ((hi - lo) or 1)
                if b - a <= 1e-9:
                    continue
                top = min(y1, b + 0.003)
                out.append(f"\\fill[{color}] {_c(bx0, a)} rectangle {_c(bx1, top)};")
        out.append(f"\\draw[{ctx.theme.ink}!70, line width=0.4pt] {_c(bx0, y0)} rectangle {_c(bx1, y1)};")
        tl = f"font={self.tick_font}, text={ctx.theme.ink}!85"
        values = self._ticks(cfg["ticks"], (lo, hi), "linear", self.height)
        names = []
        for i, (v, lab) in enumerate(values):
            yy = y0 + self.height * (v - lo) / ((hi - lo) or 1)
            out.append(f"\\draw[{ctx.theme.ink}!70, line width=0.4pt] {_c(bx1, yy)} -- {_c(bx1 + 0.08, yy)};")
            nm = f"{name}-cb{i}"
            names.append(nm)
            out.append(f"\\node[anchor=west, {tl}, inner sep=2pt] ({nm}) at {_c(bx1 + 0.1, yy)} {{{lab}}};")
        if cfg["label"] is not None:
            if names:
                out.append(f"\\node[fit={''.join(f'({n})' for n in names)}, inner sep=0pt] ({name}-cbticks) {{}};")
                out.append(f"\\node[anchor=north, rotate=90, font={self.label_font}, inner sep=4pt] at ({name}-cbticks.east |- 0,{fmt((y0 + y1) / 2, PREC)}) {{{label_text(cfg['label'])}}};")
            else:
                out.append(f"\\node[anchor=north, rotate=90, font={self.label_font}] at {_c(bx1 + 0.2, (y0 + y1) / 2)} {{{label_text(cfg['label'])}}};")
        return out
