"""3D plots: surfaces, curves and points with an orthographic camera.

Everything is projected in Python and depth-sorted (painter's algorithm), so
the output is plain 2D TikZ that compiles quickly and looks the same
everywhere::

    ax = pic.add(tz.Axes3D(xlim=(-2, 2), ylim=(-2, 2), azimuth=-60, elevation=30))
    ax.surface(lambda x, y: np.exp(-x**2 - y**2), cmap="viridis")
    ax.plot3d(xs, ys, zs, color="Red")
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from .._fmt import fmt, label_text
from ..color import Color
from ..colormaps import get_colormap
from ..contour import auto_levels, contour_lines
from ..coords import LazyCoord, Point, as_point
from ..elements import Element
from ..ticks import format_tick, linear_ticks, nice_step

if TYPE_CHECKING:  # pragma: no cover
    from ..render import RenderContext

__all__ = ["Axes3D"]


def _c(x: float, y: float) -> str:
    return f"({fmt(x, 3)},{fmt(y, 3)})"


def _arr(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _runs(points: np.ndarray) -> list[np.ndarray]:
    """Split an (n, 2) array of screen points at non-finite rows."""
    ok = np.all(np.isfinite(points), axis=1)
    runs, start = [], None
    for i, good in enumerate(list(ok) + [False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= 2:
                runs.append(points[start:i])
            start = None
    return runs


class _Prim:
    """A drawable piece with a depth (larger = closer to the viewer)."""

    __slots__ = ("depth", "lines")

    def __init__(self, depth: float, lines: list[str]):
        self.depth = depth
        self.lines = lines


class Axes3D(Element):
    """A 3D coordinate box seen from ``azimuth``/``elevation`` (degrees).

    Parameters
    ----------
    at:
        Picture position of the centre of the box.
    size:
        Box lengths ``(x, y, z)`` in cm before projection.
    xlim, ylim, zlim:
        Data ranges (``zlim`` defaults to the data range).
    panes, grid:
        Draw the three back panes with grid lines (matplotlib style).
    """

    def __init__(
        self,
        at: Any = (0, 0),
        *,
        size: tuple[float, float, float] = (4.0, 4.0, 3.0),
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        zlim: tuple[float, float] | None = None,
        azimuth: float = -60.0,
        elevation: float = 28.0,
        xlabel: Any = None,
        ylabel: Any = None,
        zlabel: Any = None,
        title: Any = None,
        panes: bool = True,
        grid: bool = True,
        ticks: bool = True,
        box: bool = False,
        pane_color: str | None = None,
        tick_font: str = r"\scriptsize",
        label_font: str = r"\small",
        name: str | None = None,
    ):
        self.center = as_point(at)
        self.size = tuple(float(s) for s in size)
        self._xlim, self._ylim, self._zlim = xlim, ylim, zlim
        self.azimuth, self.elevation = azimuth, elevation
        self.xlabel, self.ylabel, self.zlabel, self.title = xlabel, ylabel, zlabel, title
        self.panes, self.grid, self.ticks, self.box = panes, grid, ticks, box
        self.pane_color = pane_color
        self.tick_font, self.label_font = tick_font, label_font
        self.name = name
        self._artists: list[Callable[["RenderContext"], list[_Prim]]] = []
        self._bounds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._overlays: list[Callable[["RenderContext"], list[str]]] = []
        self._floor: list[Callable[["RenderContext"], list[str]]] = []
        self._set_camera()

    # -- camera ------------------------------------------------------------------
    def _set_camera(self) -> None:
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        self._right = np.array([-math.sin(az), math.cos(az), 0.0])
        self._up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
        self._view = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])

    @property
    def xlim(self) -> tuple[float, float]:
        return self._limits()[0]

    @property
    def ylim(self) -> tuple[float, float]:
        return self._limits()[1]

    @property
    def zlim(self) -> tuple[float, float]:
        return self._limits()[2]

    def _limits(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        out = []
        for k, user in enumerate((self._xlim, self._ylim, self._zlim)):
            if user is not None:
                out.append((float(user[0]), float(user[1])))
                continue
            vals = [b[k][np.isfinite(b[k])] for b in self._bounds]
            vals = [v for v in vals if v.size]
            if not vals:
                out.append((-1.0, 1.0))
                continue
            lo = float(min(v.min() for v in vals))
            hi = float(max(v.max() for v in vals))
            if hi == lo:
                lo, hi = lo - 1, hi + 1
            out.append((lo, hi))
        return out[0], out[1], out[2]

    def _box_coords(self, x: Any, y: Any, z: Any) -> np.ndarray:
        """Data -> centred box coordinates (cm), shape (..., 3)."""
        (x0, x1), (y0, y1), (z0, z1) = self._limits()
        lx, ly, lz = self.size
        u = ((_arr(x) - x0) / (x1 - x0) - 0.5) * lx
        v = ((_arr(y) - y0) / (y1 - y0) - 0.5) * ly
        w = ((_arr(z) - z0) / (z1 - z0) - 0.5) * lz
        return np.stack([u, v, w], axis=-1)

    def _screen(self, b: np.ndarray) -> np.ndarray:
        """Box coordinates -> picture coordinates, shape (..., 2)."""
        X = b @ self._right + self.center.x
        Y = b @ self._up + self.center.y
        return np.stack([X, Y], axis=-1)

    def _depth(self, b: np.ndarray) -> np.ndarray:
        return b @ self._view

    def project(self, x: float, y: float, z: float) -> Point:
        """Picture coordinates of data point ``(x, y, z)`` (limits must be known)."""
        s = self._screen(self._box_coords(x, y, z))
        return Point(float(s[0]), float(s[1]))

    def point(self, x: float, y: float, z: float) -> LazyCoord:
        """Like :meth:`project` but resolved at render time (safe with autoscaled z)."""
        return LazyCoord(lambda: self.project(x, y, z))

    # -- data ---------------------------------------------------------------------------
    def _grid(self, f: Any, X: Any, Y: Any, Z: Any, resolution: int | tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if f is not None:
            if self._xlim is None or self._ylim is None:
                raise ValueError("surfaces of functions need xlim and ylim")
            nx, ny = (resolution, resolution) if isinstance(resolution, int) else resolution
            xs = np.linspace(*self._xlim, nx)
            ys = np.linspace(*self._ylim, ny)
            X, Y = np.meshgrid(xs, ys)
            with np.errstate(all="ignore"):
                try:
                    Z = np.asarray(f(X, Y), dtype=float)
                    if Z.shape != X.shape:
                        raise ValueError
                except Exception:  # noqa: BLE001 - scalar fallback
                    Z = np.vectorize(lambda a, b: float(f(a, b)))(X, Y)
            return X, Y, Z
        if X is None or Y is None or Z is None:
            raise ValueError("give a function f(x, y) or arrays X, Y, Z")
        return _arr(X), _arr(Y), _arr(Z)

    def surface(
        self,
        f: Callable[[Any, Any], Any] | None = None,
        *,
        X: Any = None,
        Y: Any = None,
        Z: Any = None,
        resolution: int | tuple[int, int] = 32,
        cmap: Any = "viridis",
        color: str | None = None,
        values: Any = None,
        vmin: float | None = None,
        vmax: float | None = None,
        edges: Any = "dark",
        edge_width: float = 0.15,
        shade: bool = True,
        opacity: float | None = None,
    ) -> "Axes3D":
        """A surface ``z = f(x, y)`` or a parametric mesh ``(X, Y, Z)``.

        Colored by height through ``cmap`` (or by ``values``), or uniformly
        with ``color``. ``edges``: ``"dark"`` (darker shade of each face),
        ``"none"`` or any color.
        """
        Xg, Yg, Zg = self._grid(f, X, Y, Z, resolution)
        self._bounds.append((Xg.ravel(), Yg.ravel(), Zg.ravel()))
        cm = get_colormap(cmap) if color is None else None
        vals = Zg if values is None else _arr(values)

        def build(ctx: "RenderContext") -> list[_Prim]:
            B = self._box_coords(Xg, Yg, Zg)
            S = self._screen(B)
            D = self._depth(B)
            lo = float(np.nanmin(vals)) if vmin is None else vmin
            hi = float(np.nanmax(vals)) if vmax is None else vmax
            light = self._view + 0.6 * self._up - 0.4 * self._right
            light = light / np.linalg.norm(light)
            base_rgb = None
            if color is not None:
                base_rgb = Color(color).rgb if not isinstance(color, Color) else color.rgb
                if base_rgb is None:
                    base_rgb = (0.4, 0.5, 0.8)
            prims = []
            ny, nx = Zg.shape
            op = f", fill opacity={fmt(opacity)}" if opacity is not None else ""
            for i in range(ny - 1):
                for j in range(nx - 1):
                    quad = B[i : i + 2, j : j + 2].reshape(4, 3)
                    if not np.all(np.isfinite(quad)):
                        continue
                    if cm is not None:
                        t = float(np.mean(vals[i : i + 2, j : j + 2]))
                        t = 0.5 if hi == lo else (t - lo) / (hi - lo)
                        rgb = cm.rgb(round(min(1.0, max(0.0, t)) * 127) / 127)
                    else:
                        rgb = base_rgb
                    if shade:
                        n = np.cross(B[i + 1, j + 1] - B[i, j], B[i, j + 1] - B[i + 1, j])
                        norm = np.linalg.norm(n)
                        k = abs(float(n @ light) / norm) if norm > 0 else 1.0
                        factor = round((0.55 + 0.45 * k) * 32) / 32
                        rgb = tuple(min(1.0, c * factor + 0.04) for c in rgb)
                    fill = Color(rgb).expr
                    if edges == "dark":
                        draw = f"{fill}!65!black"
                    elif edges in (None, "none", False):
                        draw = fill
                    else:
                        draw = str(edges)
                    pts = S[[i, i, i + 1, i + 1], [j, j + 1, j + 1, j]]
                    path = " -- ".join(_c(*p) for p in pts)
                    prims.append(_Prim(float(np.mean(D[i : i + 2, j : j + 2])),
                                       [f"\\filldraw[fill={fill}, draw={draw}{op}] {path} -- cycle;"]))
            return prims

        self._artists.append(build)
        self._last_surface = dict(cmap=cm, values=vals, vmin=vmin, vmax=vmax)
        self._edge_width = edge_width
        return self

    def wireframe(self, f: Callable[[Any, Any], Any] | None = None, *, X: Any = None, Y: Any = None, Z: Any = None,
                  resolution: int | tuple[int, int] = 20, color: str = "Blue", line_width: float = 0.4) -> "Axes3D":
        """Grid lines of a surface (no hidden-line removal)."""
        Xg, Yg, Zg = self._grid(f, X, Y, Z, resolution)
        self._bounds.append((Xg.ravel(), Yg.ravel(), Zg.ravel()))

        def build(ctx: "RenderContext") -> list[_Prim]:
            S = self._screen(self._box_coords(Xg, Yg, Zg))
            D = self._depth(self._box_coords(Xg, Yg, Zg))
            prims = []
            ny, nx = Zg.shape
            for i in range(ny):
                for j in range(nx - 1):
                    prims.append(_Prim(float((D[i, j] + D[i, j + 1]) / 2),
                                       [f"\\draw[{color}, line width={line_width:g}pt] {_c(*S[i, j])} -- {_c(*S[i, j + 1])};"]))
            for j in range(nx):
                for i in range(ny - 1):
                    prims.append(_Prim(float((D[i, j] + D[i + 1, j]) / 2),
                                       [f"\\draw[{color}, line width={line_width:g}pt] {_c(*S[i, j])} -- {_c(*S[i + 1, j])};"]))
            return prims

        self._artists.append(build)
        return self

    def plot3d(self, x: Any, y: Any, z: Any, *, color: str = "Red", line_width: float = 1.4, dash: str | None = None,
               marker: bool = False, marker_size: float = 1.8, arrow: bool = False, on_top: bool = True,
               lift: float = 0.0) -> "Axes3D":
        """A 3D polyline. ``on_top=True`` draws it over everything (no occlusion);
        ``lift`` raises it slightly (in cm along z) to sit visibly on a surface."""
        xs, ys, zs = _arr(x).ravel(), _arr(y).ravel(), _arr(z).ravel()
        self._bounds.append((xs, ys, zs))
        style = f"{color}, line width={line_width:g}pt" + (f", {dash}" if dash else "")

        def pts(ctx: "RenderContext") -> tuple[np.ndarray, np.ndarray]:
            B = self._box_coords(xs, ys, zs)
            B[:, 2] += lift
            return self._screen(B), self._depth(B)

        if on_top:
            def overlay(ctx: "RenderContext") -> list[str]:
                S, _ = pts(ctx)
                tip = ", -{Stealth[length=5pt]}" if arrow else ""
                out = [f"\\draw[{style}{tip}] " + " -- ".join(_c(*p) for p in run) + ";" for run in _runs(S)]
                good = S[np.all(np.isfinite(S), axis=1)]
                if marker and len(good):
                    out.append(f"\\fill[{color}] " + " ".join(f"{_c(*p)} circle[radius={marker_size:g}pt]" for p in good) + ";")
                return out

            self._overlays.append(overlay)
        else:
            def build(ctx: "RenderContext") -> list[_Prim]:
                S, D = pts(ctx)
                prims = []
                for k in range(len(S) - 1):
                    if not (np.all(np.isfinite(S[k : k + 2])) and np.all(np.isfinite(D[k : k + 2]))):
                        continue
                    prims.append(_Prim(float((D[k] + D[k + 1]) / 2) + 0.05,
                                       [f"\\draw[{style}, line cap=round] {_c(*S[k])} -- {_c(*S[k + 1])};"]))
                if marker:
                    for k in range(len(S)):
                        if not (np.all(np.isfinite(S[k])) and np.isfinite(D[k])):
                            continue
                        prims.append(_Prim(float(D[k]) + 0.06, [f"\\fill[{color}] {_c(*S[k])} circle[radius={marker_size:g}pt];"]))
                return prims

            self._artists.append(build)
        return self

    def scatter3d(self, x: Any, y: Any, z: Any, *, color: str = "Blue", size: float = 2.0, c: Any = None,
                  cmap: Any = "viridis", edgecolor: str | None = "white") -> "Axes3D":
        """Points, depth-sorted with the surfaces."""
        xs, ys, zs = _arr(x).ravel(), _arr(y).ravel(), _arr(z).ravel()
        self._bounds.append((xs, ys, zs))
        vals = None if c is None else _arr(c).ravel()
        cm = get_colormap(cmap)

        def build(ctx: "RenderContext") -> list[_Prim]:
            B = self._box_coords(xs, ys, zs)
            S, D = self._screen(B), self._depth(B)
            prims = []
            lo, hi = (float(vals.min()), float(vals.max())) if vals is not None else (0, 1)
            for k in range(len(S)):
                col = color if vals is None else cm(0.5 if hi == lo else (vals[k] - lo) / (hi - lo)).expr
                edge = f", draw={edgecolor}, line width=0.3pt" if edgecolor else ""
                cmd = "filldraw" if edgecolor else "fill"
                prims.append(_Prim(float(D[k]) + 0.02, [f"\\{cmd}[fill={col}{edge}] {_c(*S[k])} circle[radius={size:g}pt];"]))
            return prims

        self._artists.append(build)
        return self

    def arrow3d(self, start: Sequence[float], end: Sequence[float], *, color: str = "Ink", line_width: float = 1.0,
                label: Any = None, label_position: str = "above") -> "Axes3D":
        """An arrow between two 3D points, drawn on top."""
        self._bounds.append((_arr([start[0], end[0]]), _arr([start[1], end[1]]), _arr([start[2], end[2]])))

        def overlay(ctx: "RenderContext") -> list[str]:
            p, q = self.project(*start), self.project(*end)
            lab = f" node[{label_position}, font=\\small, align=center] {{{label_text(label)}}}" if label is not None else ""
            return [f"\\draw[->, {color}, line width={line_width:g}pt] {p.tikz()} -- {q.tikz()}{lab};"]

        self._overlays.append(overlay)
        return self

    def text3d(self, x: float, y: float, z: float, text: Any, **options: Any) -> "Axes3D":
        """Text at a 3D point (node options as keywords)."""
        from ..options import Options

        opts = Options(**options)

        def overlay(ctx: "RenderContext") -> list[str]:
            return [f"\\node{opts.render()} at {self.project(x, y, z).tikz()} {{{label_text(text)}}};"]

        self._overlays.append(overlay)
        return self

    def contour_floor(self, f: Callable[[Any, Any], Any] | None = None, *, X: Any = None, Y: Any = None, Z: Any = None,
                      levels: Any = 10, cmap: Any = "viridis", color: str | None = None, line_width: float = 0.6,
                      resolution: int = 80) -> "Axes3D":
        """Contour lines of ``f`` drawn on the floor of the box (z = zmin)."""
        Xg, Yg, Zg = self._grid(f, X, Y, Z, resolution)
        cm = get_colormap(cmap)

        def draw(ctx: "RenderContext") -> list[str]:
            xs, ys = Xg[0, :], Yg[:, 0]
            lv = auto_levels(Zg, levels) if isinstance(levels, int) else list(levels)
            z0 = self.zlim[0]
            out = []
            finite = Zg[np.isfinite(Zg)]
            if finite.size == 0:
                return out
            lo, hi = float(finite.min()), float(finite.max())
            for level in lv:
                col = color or cm(0.5 if hi == lo else (level - lo) / (hi - lo)).expr
                for line in contour_lines(xs, ys, Zg, level):
                    arr = np.array(line)
                    S = self._screen(self._box_coords(arr[:, 0], arr[:, 1], np.full(len(arr), z0)))
                    for run in _runs(S):
                        out.append(f"\\draw[{col}, line width={line_width:g}pt] " + " -- ".join(_c(*p) for p in run) + ";")
            return out

        self._floor.append(draw)
        return self

    # -- rendering -------------------------------------------------------------------------------
    def _corner(self, ux: float, vy: float, wz: float) -> np.ndarray:
        lx, ly, lz = self.size
        return np.array([ux * lx / 2, vy * ly / 2, wz * lz / 2])

    def render(self, ctx: "RenderContext") -> list[str]:
        theme = ctx.theme
        name = self.name or ctx.auto_name(self, "ax3d")
        out = [f"% 3d axes {name}"]
        (x0, x1), (y0, y1), (z0, z1) = self._limits()
        v = self._view
        # back sides: -1 means the minimum side is at the back
        bx = -1.0 if v[0] > 0 else 1.0
        by = -1.0 if v[1] > 0 else 1.0
        bz = -1.0 if v[2] > 0 else 1.0
        fx, fy = -bx, -by
        pane_fill = self.pane_color or f"{theme.ink}!4"
        pane_line = f"{theme.ink}!25"
        grid_line = f"{theme.ink}!14"

        def P(ux: float, vy: float, wz: float) -> str:
            return _c(*self._screen(self._corner(ux, vy, wz)))

        def to_unit(val: float, lo: float, hi: float) -> float:
            return 2 * (val - lo) / (hi - lo) - 1

        xt = [(t, format_tick(t, s)) for s in [nice_step(x1 - x0, 4)] for t in linear_ticks(x0, x1, step=s)]
        yt = [(t, format_tick(t, s)) for s in [nice_step(y1 - y0, 4)] for t in linear_ticks(y0, y1, step=s)]
        zt = [(t, format_tick(t, s)) for s in [nice_step(z1 - z0, 4)] for t in linear_ticks(z0, z1, step=s)]

        if self.panes:
            out.append(f"\\filldraw[fill={pane_fill}, draw={pane_line}, line width=0.4pt] {P(-1, -1, bz)} -- {P(1, -1, bz)} -- {P(1, 1, bz)} -- {P(-1, 1, bz)} -- cycle;")
            out.append(f"\\filldraw[fill={pane_fill}, draw={pane_line}, line width=0.4pt] {P(bx, -1, -1)} -- {P(bx, 1, -1)} -- {P(bx, 1, 1)} -- {P(bx, -1, 1)} -- cycle;")
            out.append(f"\\filldraw[fill={pane_fill}, draw={pane_line}, line width=0.4pt] {P(-1, by, -1)} -- {P(1, by, -1)} -- {P(1, by, 1)} -- {P(-1, by, 1)} -- cycle;")
            if self.grid:
                segs = []
                for t, _ in xt:
                    u = to_unit(t, x0, x1)
                    segs.append(f"{P(u, -1, bz)} -- {P(u, 1, bz)}")
                    segs.append(f"{P(u, by, -1)} -- {P(u, by, 1)}")
                for t, _ in yt:
                    w = to_unit(t, y0, y1)
                    segs.append(f"{P(-1, w, bz)} -- {P(1, w, bz)}")
                    segs.append(f"{P(bx, w, -1)} -- {P(bx, w, 1)}")
                for t, _ in zt:
                    w = to_unit(t, z0, z1)
                    segs.append(f"{P(bx, -1, w)} -- {P(bx, 1, w)}")
                    segs.append(f"{P(-1, by, w)} -- {P(1, by, w)}")
                from ..elements import wrap_tokens

                out += wrap_tokens(f"\\draw[{grid_line}, line width=0.3pt]", " ".join(segs).split(" "))
        for floor in self._floor:
            out += floor(ctx)
        # depth sorted content
        prims: list[_Prim] = []
        for build in self._artists:
            prims += build(ctx)
        prims.sort(key=lambda p: p.depth)
        if prims:
            ew = getattr(self, "_edge_width", 0.15)
            out.append(f"\\begin{{scope}}[line width={ew:g}pt, line join=round]")
            for p in prims:
                out += ["  " + ln for ln in p.lines]
            out.append(r"\end{scope}")
        for overlay in self._overlays:
            out += overlay(ctx)
        if self.box:
            segs = []
            for a in (-1, 1):
                for b in (-1, 1):
                    segs.append(f"{P(-1, a, b)} -- {P(1, a, b)}")
                    segs.append(f"{P(a, -1, b)} -- {P(a, 1, b)}")
                    segs.append(f"{P(a, b, -1)} -- {P(a, b, 1)}")
            out.append(f"\\draw[{theme.ink}!50, line width=0.4pt] " + " ".join(segs) + ";")
        # ticks and labels
        if self.ticks:
            tl = f"font={self.tick_font}, text={theme.ink}!80, inner sep=1pt"
            zbase = bz  # floor level used by the x/y axes

            def outward(edge_mid: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, float]:
                d = self._screen(edge_mid + extra) - self._screen(edge_mid)
                n = np.linalg.norm(d)
                d = d / n if n > 0 else np.array([0.0, -1.0])
                return d, math.degrees(math.atan2(-d[1], -d[0]))

            # x axis on the front edge of the floor
            d, anchor = outward(self._corner(0, fy, zbase), np.array([0, fy * 0.4, 0]))
            for t, lab in xt:
                u = to_unit(t, x0, x1)
                p = self._screen(self._corner(u, fy, zbase)) + d * 0.12
                out.append(f"\\node[anchor={fmt(anchor % 360, 0)}, {tl}] at {_c(*p)} {{{lab}}};")
            if self.xlabel is not None:
                p = self._screen(self._corner(0, fy, zbase)) + d * 0.65
                out.append(f"\\node[anchor={fmt(anchor % 360, 0)}, font={self.label_font}, align=center] at {_c(*p)} {{{label_text(self.xlabel)}}};")
            # y axis
            d, anchor = outward(self._corner(fx, 0, zbase), np.array([fx * 0.4, 0, 0]))
            for t, lab in yt:
                w = to_unit(t, y0, y1)
                p = self._screen(self._corner(fx, w, zbase)) + d * 0.12
                out.append(f"\\node[anchor={fmt(anchor % 360, 0)}, {tl}] at {_c(*p)} {{{lab}}};")
            if self.ylabel is not None:
                p = self._screen(self._corner(fx, 0, zbase)) + d * 0.65
                out.append(f"\\node[anchor={fmt(anchor % 360, 0)}, font={self.label_font}, align=center] at {_c(*p)} {{{label_text(self.ylabel)}}};")
            # z axis on the left-most vertical edge
            corners = [(a, b) for a in (-1, 1) for b in (-1, 1)]
            a, b = min(corners, key=lambda ab: self._screen(self._corner(ab[0], ab[1], 0))[0])
            for t, lab in zt:
                w = to_unit(t, z0, z1)
                p = self._screen(self._corner(a, b, w)) + np.array([-0.1, 0.0])
                out.append(f"\\node[anchor=east, {tl}] at {_c(*p)} {{{lab}}};")
            if self.zlabel is not None:
                p = self._screen(self._corner(a, b, 0)) + np.array([-0.55, 0.0])
                out.append(f"\\node[anchor=east, font={self.label_font}, align=center] at {_c(*p)} {{{label_text(self.zlabel)}}};")
        if self.title is not None:
            tops = [self._screen(self._corner(a, b, 1))[1] for a in (-1, 1) for b in (-1, 1)]
            out.append(f"\\node[anchor=south, font=\\small\\bfseries, align=center] at {_c(self.center.x, max(tops) + 0.2)} {{{label_text(self.title)}}};")
        return out
