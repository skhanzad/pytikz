"""3D boxes for tensors and convolutional network architectures.

:class:`Cuboid` draws a single shaded box (a tensor, a feature map, a
volume). :class:`ConvNet` lays out a sequence of them, PlotNeuralNet style::

    cnn = pic.add(tz.ConvNet())
    cnn.input(224, 3, label="Image")
    cnn.conv(64, 224, label="conv1")
    cnn.pool(112)
    cnn.conv(128, 112, label="conv2")
    cnn.fc(4096, label="fc")
    cnn.fc(10, label="softmax", color="softmax")
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

from .._fmt import fmt, label_text
from ..color import color_text, xmix
from ..coords import Point, as_point
from ..elements import Element

if TYPE_CHECKING:  # pragma: no cover
    from ..render import RenderContext

__all__ = ["Cuboid", "ConvNet"]


def _p(p: Point) -> str:
    return f"({fmt(p.x, 3)},{fmt(p.y, 3)})"


class Cuboid(Element):
    """A box in oblique projection with lit top and side faces.

    ``at`` is the lower-left corner of the front face; ``width`` and
    ``height`` size the front face and ``depth`` recedes at ``angle``
    degrees (scaled by ``depth_scale``). ``grid=(nx, ny, nz)`` draws cell lines
    on the visible faces (handy for tensors).
    """

    def __init__(
        self,
        at: Any = (0, 0),
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        *,
        color: str = "Blue",
        angle: float = 40.0,
        depth_scale: float = 0.55,
        grid: tuple[int, int, int] | None = None,
        opacity: float | None = None,
        label: Any = None,
        line_width: float = 0.5,
        name: str | None = None,
    ):
        self.origin = as_point(at)
        self.width, self.height, self.depth = float(width), float(height), float(depth)
        self.color = color
        self.angle = angle
        self.depth_scale = depth_scale
        self.grid = grid
        self.opacity = opacity
        self.label = label
        self.line_width = line_width
        self.name = name

    @property
    def depth_vector(self) -> Point:
        return Point.polar(self.angle, self.depth * self.depth_scale)

    # handy points ---------------------------------------------------------------
    @property
    def front_center(self) -> Point:
        return self.origin + Point(self.width / 2, self.height / 2)

    @property
    def center(self) -> Point:
        return self.front_center + self.depth_vector / 2

    @property
    def right_center(self) -> Point:
        """Centre of the right (visible side) face."""
        return self.origin + Point(self.width, self.height / 2) + self.depth_vector / 2

    @property
    def left_center(self) -> Point:
        return self.origin + Point(0, self.height / 2) + self.depth_vector / 2

    @property
    def top_center(self) -> Point:
        return self.origin + Point(self.width / 2, self.height) + self.depth_vector / 2

    @property
    def bottom_center(self) -> Point:
        return self.origin + Point(self.width / 2, 0)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of the drawing."""
        d = self.depth_vector
        xs = [self.origin.x, self.origin.x + self.width + max(0.0, d.x), self.origin.x + min(0.0, d.x)]
        ys = [self.origin.y, self.origin.y + self.height + max(0.0, d.y), self.origin.y + min(0.0, d.y)]
        return min(xs), min(ys), max(xs), max(ys)

    def render(self, ctx: "RenderContext") -> list[str]:
        theme = ctx.theme
        c = color_text(theme.role(self.color))
        o = self.origin
        w, h = self.width, self.height
        d = self.depth_vector
        a, b = o, o + Point(w, 0)
        cc, dd = o + Point(w, h), o + Point(0, h)
        edge = xmix(c, 60, "black")
        op = f", fill opacity={fmt(self.opacity)}" if self.opacity is not None else ""
        lw = f"line width={self.line_width:g}pt"
        out = []
        if self.name:
            out.append(f"% cuboid {self.name}")
        # top, side, front (front last so its edges are crisp)
        out.append(f"\\filldraw[fill={xmix(c, 28)}{op}, draw={edge}, {lw}] {_p(dd)} -- {_p(cc)} -- {_p(cc + d)} -- {_p(dd + d)} -- cycle;")
        out.append(f"\\filldraw[fill={xmix(c, 62)}{op}, draw={edge}, {lw}] {_p(b)} -- {_p(b + d)} -- {_p(cc + d)} -- {_p(cc)} -- cycle;")
        out.append(f"\\filldraw[fill={xmix(c, 45)}{op}, draw={edge}, {lw}] {_p(a)} rectangle {_p(cc)};")
        if self.grid:
            nx, ny, nz = self.grid
            segs = []
            for k in range(1, nx):
                x = a + Point(w * k / nx, 0)
                segs.append(f"{_p(x)} -- {_p(x + Point(0, h))} -- {_p(x + Point(0, h) + d)}")
            for k in range(1, ny):
                y = a + Point(0, h * k / ny)
                segs.append(f"{_p(y)} -- {_p(y + Point(w, 0))} -- {_p(y + Point(w, 0) + d)}")
            for k in range(1, nz):
                z = d * (k / nz)
                segs.append(f"{_p(dd + z)} -- {_p(cc + z)} -- {_p(b + z)}")
            out.append(f"\\draw[{edge}, line width=0.3pt] " + " ".join(segs) + ";")
        if self.label is not None:
            x0, y0, x1, _ = self.extent
            out.append(f"\\node[anchor=north, font=\\small, align=center] at ({fmt((x0 + x1) / 2, 3)},{fmt(y0 - 0.1, 3)}) {{{label_text(self.label)}}};")
        return out


class _Layer:
    def __init__(self, kind: str, width: float, height: float, depth: float, color: str, label: Any,
                 dims: tuple[Any, Any, Any], bands: int, gap_before: float | None, opacity: float | None):
        self.kind = kind
        self.width, self.height, self.depth = width, height, depth
        self.color = color
        self.label = label
        self.dims = dims
        self.bands = bands
        self.gap_before = gap_before
        self.opacity = opacity
        self.cuboid: Cuboid | None = None


def _default_channel_scale(c: float) -> float:
    return 0.045 * float(c) ** 0.48


def _default_size_scale(s: float) -> float:
    return 0.42 * math.log2(max(2.0, float(s))) - 0.15


class ConvNet(Element):
    """A left-to-right sequence of 3D feature maps joined by arrows.

    Spatial size sets a block's height and depth, the number of channels its
    thickness (both through adjustable scale functions). Every method returns
    the network so calls can be chained, and blocks can be referenced by index
    for extra annotations (``net.block(2).top_center``).
    """

    def __init__(
        self,
        at: Any = (0, 0),
        *,
        spacing: float = 0.75,
        angle: float = 40.0,
        depth_scale: float = 0.55,
        arrows: bool = True,
        dims: bool = True,
        channel_scale: Callable[[float], float] = _default_channel_scale,
        size_scale: Callable[[float], float] = _default_size_scale,
        caption_font: str = r"\small",
        name: str = "cnn",
    ):
        self.origin = as_point(at)
        self.spacing = spacing
        self.angle = angle
        self.depth_scale = depth_scale
        self.arrows = arrows
        self.show_dims = dims
        self.channel_scale = channel_scale
        self.size_scale = size_scale
        self.caption_font = caption_font
        self.name = name
        self.layers: list[_Layer] = []
        self._skips: list[tuple[int, int, Any, str]] = []
        self._laid_out = False

    # -- building -------------------------------------------------------------------
    def add_block(self, *, width: float, height: float, depth: float | None = None, color: str = "conv",
                  label: Any = None, dims: tuple[Any, Any, Any] = (None, None, None), bands: int = 1,
                  gap: float | None = None, opacity: float | None = None, kind: str = "block") -> "ConvNet":
        """Add a block with explicit sizes in cm (``depth`` defaults to ``height``)."""
        self.layers.append(_Layer(kind, width, height, height if depth is None else depth, color, label, dims,
                                  bands, gap, opacity))
        self._laid_out = False
        return self

    def input(self, size: int | tuple[int, int], channels: int = 3, *, label: Any = "Input",
              color: str = "Gray", **kw: Any) -> "ConvNet":
        """The input tensor (an image of ``size`` x ``size`` x ``channels``)."""
        h, w = (size, size) if isinstance(size, int) else size
        return self.add_block(width=self.channel_scale(channels), height=self.size_scale(h),
                              depth=self.size_scale(w), color=color, label=label,
                              dims=(h, w, channels), kind="input", **kw)

    def conv(self, channels: int, size: int | tuple[int, int], *, label: Any = None, color: str = "conv",
             bands: int = 1, **kw: Any) -> "ConvNet":
        """A convolution output with ``channels`` feature maps of ``size`` x ``size``.

        ``bands=k`` draws ``k`` stacked slices (e.g. conv-conv-relu).
        """
        h, w = (size, size) if isinstance(size, int) else size
        return self.add_block(width=self.channel_scale(channels) * bands, height=self.size_scale(h),
                              depth=self.size_scale(w), color=color, label=label, dims=(h, w, channels),
                              bands=bands, kind="conv", **kw)

    def pool(self, size: int | tuple[int, int], channels: int | None = None, *, label: Any = None,
             color: str = "pool", **kw: Any) -> "ConvNet":
        """A pooling output; ``channels`` defaults to the previous block's."""
        h, w = (size, size) if isinstance(size, int) else size
        if channels is None:
            channels = next((ly.dims[2] for ly in reversed(self.layers) if ly.dims[2] is not None), 16)
        return self.add_block(width=max(0.12, self.channel_scale(channels) * 0.55), height=self.size_scale(h),
                              depth=self.size_scale(w), color=color, label=label, dims=(h, w, None),
                              kind="pool", **kw)

    def fc(self, units: int, *, label: Any = None, color: str = "fc", **kw: Any) -> "ConvNet":
        """A fully connected layer, drawn as a thin tall bar."""
        height = 0.55 * math.log2(max(2, units)) - 0.6
        return self.add_block(width=0.22, height=max(0.6, height), depth=0.22, color=color, label=label,
                              dims=(units, None, None), kind="fc", **kw)

    def skip(self, start: int, end: int, label: Any = None, color: str = "Gray") -> "ConvNet":
        """A skip connection drawn above the blocks from ``start`` to ``end``."""
        self._skips.append((start, end, label, color))
        return self

    # -- layout -------------------------------------------------------------------------
    def _layout(self) -> None:
        x = self.origin.x
        for k, layer in enumerate(self.layers):
            if k > 0:
                x += layer.gap_before if layer.gap_before is not None else self.spacing
            d = Point.polar(self.angle, layer.depth * self.depth_scale)
            y = self.origin.y - (layer.height + d.y) / 2
            layer.cuboid = Cuboid((x, y), layer.width, layer.height, layer.depth, color=layer.color,
                                  angle=self.angle, depth_scale=self.depth_scale, opacity=layer.opacity)
            x += layer.width + max(0.0, d.x)
        self._laid_out = True

    def block(self, index: int) -> Cuboid:
        """The :class:`Cuboid` drawn for block ``index`` (positions are final)."""
        if not self._laid_out:
            self._layout()
        cub = self.layers[index].cuboid
        assert cub is not None
        return cub

    @property
    def extent(self) -> tuple[float, float, float, float]:
        if not self._laid_out:
            self._layout()
        ext = [ly.cuboid.extent for ly in self.layers if ly.cuboid is not None]
        return min(e[0] for e in ext), min(e[1] for e in ext), max(e[2] for e in ext), max(e[3] for e in ext)

    # -- rendering ------------------------------------------------------------------------
    def render(self, ctx: "RenderContext") -> list[str]:
        self._layout()
        theme = ctx.theme
        out = [f"% conv net {self.name}"]
        if not self.layers:
            return out
        _, ymin, _, ymax = self.extent
        dim_style = f"font=\\scriptsize, text={theme.ink}!70, inner sep=1.5pt"
        for k, layer in enumerate(self.layers):
            cub = layer.cuboid
            assert cub is not None
            out.append(f"% block {k}" + (f": {layer.label}" if layer.label else ""))
            if layer.bands > 1:
                slice_w = layer.width / layer.bands
                for b in range(layer.bands):
                    shade = layer.color if b < layer.bands - 1 else layer.color
                    part = Cuboid(cub.origin + Point(b * slice_w, 0), slice_w, layer.height, layer.depth,
                                  color=shade, angle=self.angle, depth_scale=self.depth_scale,
                                  opacity=layer.opacity)
                    out += part.render(ctx)
            else:
                out += cub.render(ctx)
            if self.show_dims:
                h_lab, w_lab, c_lab = layer.dims
                o = cub.origin
                d = cub.depth_vector
                if layer.kind == "fc":
                    if h_lab is not None:
                        out.append(f"\\node[anchor=south, {dim_style}] at {_p(cub.top_center + d / 2)} {{{h_lab}}};")
                else:
                    if h_lab is not None:
                        out.append(f"\\node[anchor=south, rotate=90, {dim_style}] at {_p(o + Point(0, layer.height / 2))} {{{h_lab}}};")
                    if w_lab is not None:
                        mid = o + Point(layer.width, 0) + d / 2
                        out.append(f"\\node[anchor=north west, rotate={fmt(self.angle)}, {dim_style}] at {_p(mid)} {{{w_lab}}};")
                    if c_lab is not None:
                        out.append(f"\\node[anchor=north, {dim_style}] at {_p(o + Point(layer.width / 2, 0))} {{{c_lab}}};")
        # arrows between consecutive blocks
        if self.arrows:
            for k in range(len(self.layers) - 1):
                a = self.layers[k].cuboid
                b = self.layers[k + 1].cuboid
                assert a is not None and b is not None
                start = Point(a.extent[2], self.origin.y)
                end = Point(b.origin.x, self.origin.y)
                if end.x - start.x > 0.15:
                    out.append(f"\\draw[->, {theme.ink}!75, line width=0.7pt] {_p(start)} -- {_p(end)};")
        # captions, aligned on one baseline
        base = ymin - (0.45 if self.show_dims else 0.15)
        for layer in self.layers:
            if layer.label is None:
                continue
            cub = layer.cuboid
            assert cub is not None
            x0, _, x1, _ = cub.extent
            out.append(
                f"\\node[anchor=north, font={self.caption_font}, align=center] at "
                f"({fmt((x0 + x1) / 2, 3)},{fmt(base, 3)}) {{{label_text(layer.label)}}};"
            )
        # skip connections
        for n, (s, e, label, color) in enumerate(self._skips):
            a, b = self.layers[s].cuboid, self.layers[e].cuboid
            assert a is not None and b is not None
            top = ymax + 0.35 + 0.3 * n
            p, q = a.top_center, b.top_center
            lab = ""
            if label is not None:
                lab = f" node[midway, above, font=\\scriptsize, align=center] {{{label_text(label)}}}"
            out.append(
                f"\\draw[->, {color}, line width=0.7pt, rounded corners=4pt] {_p(p)} -- ({fmt(p.x, 3)},{fmt(top, 3)})"
                f" --{lab} ({fmt(q.x, 3)},{fmt(top, 3)}) -- {_p(q)};"
            )
        return out
