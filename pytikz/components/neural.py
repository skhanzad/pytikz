"""Fully connected neural network diagrams."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from .._fmt import fmt, label_text
from ..coords import NodeRef, Point, as_point
from ..elements import Element
from ..color import color_text
from ..render import LazyName
from ..theme import ARG

if TYPE_CHECKING:  # pragma: no cover
    from ..render import RenderContext

__all__ = ["NeuralNetwork"]

Labeler = Callable[[int, int, int], Any]


def _fill_index(template: str, i: int, n: int, layer: int) -> str:
    return (
        template.replace("{i}", "{" + str(i) + "}")
        .replace("{n}", "{" + str(n) + "}")
        .replace("{l}", "{" + str(layer) + "}")
    )


class NeuralNetwork(Element):
    """A multilayer perceptron: circles for neurons, lines for weights.

    >>> net = NeuralNetwork([3, 5, 5, 2],
    ...                     labels=["$x_{i}$", None, None, r"$\\hat y_{i}$"],
    ...                     titles=["Input", "Hidden", "Hidden", "Output"])

    Parameters
    ----------
    layers:
        Number of neurons per layer.
    at:
        Position of the centre of the first layer.
    layer_spacing, neuron_spacing:
        Distances (cm) between layers and between neurons in a layer.
    neuron_size:
        Neuron diameter (cm).
    colors:
        Palette color per layer; default green input, blue hidden, red output.
    labels:
        Text inside the neurons: one entry per layer (a template where
        ``{i}`` is the 1-based neuron index and ``{n}`` the layer size, or a
        list of strings, or ``None``), or a function ``(layer, i, n) -> text``.
    inputs, outputs:
        Templates for labels outside the network with arrows, e.g.
        ``inputs="$x_{i}$"``.
    titles:
        Headings above the layers (``True`` for "Input layer" etc.).
    max_neurons:
        Larger layers are drawn as the first neurons, a vertical ellipsis and
        the last neuron.
    bias:
        Add a bias unit on top of every layer but the last.
    arrows:
        Arrow tips on the connections.
    direction:
        ``"right"`` (layers left to right) or ``"up"`` (bottom to top).
    """

    def __init__(
        self,
        layers: Sequence[int],
        *,
        at: Any = (0, 0),
        layer_spacing: float = 2.4,
        neuron_spacing: float = 1.0,
        neuron_size: float = 0.64,
        colors: Sequence[str] | None = None,
        labels: Sequence[Any] | Labeler | None = None,
        inputs: str | Sequence[str] | None = None,
        outputs: str | Sequence[str] | None = None,
        titles: Sequence[Any] | bool | None = None,
        max_neurons: int | None = None,
        bias: bool = False,
        arrows: bool = False,
        connection_color: str | None = None,
        connection_width: float = 0.45,
        direction: str = "right",
        shaded: bool = False,
        name: str | None = None,
        font: str = r"\small",
    ):
        if len(layers) < 1:
            raise ValueError("a network needs at least one layer")
        if direction not in ("right", "up"):
            raise ValueError("direction must be 'right' or 'up'")
        self.sizes = [int(n) for n in layers]
        self.origin = as_point(at)
        self.layer_spacing = layer_spacing
        self.neuron_spacing = neuron_spacing
        self.neuron_size = neuron_size
        self.colors = list(colors) if colors is not None else None
        self.labels = labels
        self.inputs = inputs
        self.outputs = outputs
        self.titles = titles
        self.max_neurons = max_neurons
        self.bias = bias
        self.arrows = arrows
        self.connection_color = connection_color
        self.connection_width = connection_width
        self.direction = direction
        self.shaded = shaded
        self._lname = LazyName(self, "nn", name)
        self.font = font
        self._highlight_neurons: dict[tuple[int, int], str] = {}
        self._highlight_edges: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._edge_labels: list[tuple[int, int, int, Any, float]] = []
        self._skip_edges: set[tuple[int, int, int]] = set()

    @property
    def name(self) -> str:
        """Prefix of the node names (``nn``, ``nn2``, ... unless given)."""
        return self._lname.resolve()

    # -- layout -----------------------------------------------------------------
    def _visible(self, layer: int) -> list[int | None]:
        """Neuron indices drawn in ``layer`` (``None`` marks the ellipsis)."""
        n = self.sizes[layer]
        m = self.max_neurons
        if m is None or n <= m:
            return list(range(n))
        head = max(1, m - 2)
        return list(range(head)) + [None, n - 1]

    def _slots(self, layer: int) -> int:
        return len(self._visible(layer)) + (1 if self.bias and layer < len(self.sizes) - 1 else 0)

    def _pos(self, layer: int, slot: int) -> Point:
        """Centre of slot ``slot`` (0 = top) in ``layer``."""
        m = self._slots(layer)
        along = layer * self.layer_spacing
        across = ((m - 1) / 2.0 - slot) * self.neuron_spacing
        if self.direction == "right":
            return self.origin + Point(along, across)
        return self.origin + Point(-across, along)

    def _slot_of(self, layer: int, i: int) -> int:
        vis = self._visible(layer)
        if i not in vis:
            raise IndexError(f"neuron {i} of layer {layer} is not drawn (hidden by max_neurons)")
        offset = 1 if self.bias and layer < len(self.sizes) - 1 else 0
        return vis.index(i) + offset

    def position(self, layer: int, i: int) -> Point:
        """Centre of neuron ``i`` (0-based) of ``layer`` in picture coordinates."""
        return self._pos(layer, self._slot_of(layer, i))

    # -- references -----------------------------------------------------------------
    def neuron(self, layer: int, i: int) -> NodeRef:
        """Reference to neuron ``i`` (0-based; negative counts from the end)."""
        if layer < 0:
            layer += len(self.sizes)
        if i < 0:
            i += self.sizes[layer]
        self._slot_of(layer, i)  # validates
        return NodeRef(self._lname.sub(f"-{layer}-{i}"))

    def bias_unit(self, layer: int) -> NodeRef:
        if not self.bias or layer >= len(self.sizes) - 1:
            raise IndexError(f"layer {layer} has no bias unit")
        return NodeRef(self._lname.sub(f"-{layer}-b"))

    def layer(self, layer: int) -> list[NodeRef]:
        """References to the visible neurons of ``layer``."""
        return [self.neuron(layer, i) for i in self._visible(layer) if i is not None]

    def title(self, layer: int) -> NodeRef:
        """Reference to the title node of ``layer`` (if titles are shown)."""
        return NodeRef(self._lname.sub(f"-title-{layer}"))

    # -- styling hooks ------------------------------------------------------------------
    def highlight(self, layer: int, i: int, color: str = "Orange") -> "NeuralNetwork":
        """Fill one neuron with another palette color."""
        self._highlight_neurons[(layer, i)] = color
        return self

    def highlight_connection(self, layer: int, i: int, j: int, color: str = "Orange", width: float = 1.4,
                             label: Any = None) -> "NeuralNetwork":
        """Emphasise the weight from neuron ``i`` of ``layer`` to ``j`` of the next layer."""
        self._highlight_edges[(layer, i, j)] = {"color": color, "width": width, "label": label}
        return self

    def label_connection(self, layer: int, i: int, j: int, text: Any, pos: float = 0.3) -> "NeuralNetwork":
        """Write ``text`` on a connection, near its start by default."""
        self._edge_labels.append((layer, i, j, text, pos))
        return self

    def remove_connection(self, layer: int, i: int, j: int) -> "NeuralNetwork":
        self._skip_edges.add((layer, i, j))
        return self

    # -- rendering ----------------------------------------------------------------------
    def _layer_color(self, layer: int, ctx: "RenderContext") -> str:
        if self.colors is not None:
            return color_text(ctx.theme.role(self.colors[layer % len(self.colors)]))
        if layer == 0:
            return ctx.theme.role("input")
        if layer == len(self.sizes) - 1:
            return ctx.theme.role("output")
        return ctx.theme.role("hidden")

    def _label(self, layer: int, i: int) -> str:
        n = self.sizes[layer]
        spec = self.labels
        if spec is None:
            return ""
        if callable(spec):
            text = spec(layer, i + 1, n)
            return "" if text is None else label_text(text)
        entry = spec[layer] if layer < len(spec) else None
        if entry is None:
            return ""
        if isinstance(entry, (list, tuple)):
            return label_text(entry[i]) if i < len(entry) else ""
        return label_text(_fill_index(str(entry), i + 1, n, layer))

    def _titles(self) -> list[Any]:
        if self.titles is True:
            n = len(self.sizes)
            out = []
            for k in range(n):
                if k == 0:
                    out.append("Input\\\\layer")
                elif k == n - 1:
                    out.append("Output\\\\layer")
                else:
                    out.append("Hidden\\\\layer" if n == 3 else f"Hidden\\\\layer {k}")
            return out
        if not self.titles:
            return []
        return list(self.titles)

    def render(self, ctx: "RenderContext") -> list[str]:
        theme = ctx.theme
        d = fmt(self.neuron_size * 10, 2) + "mm"
        pre = self.name
        if self.shaded:
            neuron = ctx.define_style(
                "neuron",
                f"{theme.arg_color()}, circle, draw={theme.stroke(ARG)}, ball color={ARG}!45, minimum size={d}, "
                f"inner sep=0pt, line width=0.6pt, font={self.font}",
                default="Blue",
                prefix=pre,
            )
        else:
            neuron = ctx.define_style(
                "neuron",
                f"{theme.arg_color()}, circle, draw={theme.stroke(ARG)}, fill={theme.fill(ARG, 22)}, "
                f"minimum size={d}, inner sep=0pt, line width=0.7pt, font={self.font}",
                default="Blue",
                prefix=pre,
            )
        bias = ctx.define_style("bias", f"{neuron}=Gray, dashed", prefix=pre)
        cc = self.connection_color or f"{theme.ink}!38"
        tip = ", -{Stealth[length=3.2pt, width=2.6pt]}, shorten >=1pt" if self.arrows else ""
        connection = ctx.define_style("connection", f"{cc}, line width={self.connection_width:g}pt{tip}", prefix=pre)
        title_style = ctx.define_style("layer title", "font=\\small\\bfseries, align=center", prefix=pre)
        nm = self.name
        out = [f"% neural network {nm}: " + "-".join(str(n) for n in self.sizes)]

        # neurons
        for layer in range(len(self.sizes)):
            color = self._layer_color(layer, ctx)
            if self.bias and layer < len(self.sizes) - 1:
                p = self._pos(layer, 0)
                out.append(f"\\node[{bias}] ({nm}-{layer}-b) at {p.tikz()} {{$1$}};")
            for i in self._visible(layer):
                if i is None:
                    continue
                p = self.position(layer, i)
                c = color_text(self._highlight_neurons.get((layer, i), color))
                out.append(f"\\node[{neuron}={c}] ({nm}-{layer}-{i}) at {p.tikz()} {{{self._label(layer, i)}}};")
            for slot, i in enumerate(self._visible(layer)):
                if i is None:
                    offset = 1 if self.bias and layer < len(self.sizes) - 1 else 0
                    p = self._pos(layer, slot + offset)
                    dots = r"$\vdots$" if self.direction == "right" else r"$\cdots$"
                    out.append(f"\\node[text={theme.stroke(color)}] at {p.tikz()} {{{dots}}};")

        # connections
        lines = []
        highlighted = []
        for layer in range(len(self.sizes) - 1):
            sources = [i for i in self._visible(layer) if i is not None]
            targets = [j for j in self._visible(layer + 1) if j is not None]
            if self.bias:
                for j in targets:
                    lines.append(f"({nm}-{layer}-b) -- ({nm}-{layer + 1}-{j})")
            for i in sources:
                for j in targets:
                    key = (layer, i, j)
                    if key in self._skip_edges:
                        continue
                    if key in self._highlight_edges:
                        highlighted.append(key)
                        continue
                    lines.append(f"({nm}-{layer}-{i}) -- ({nm}-{layer + 1}-{j})")
        if self.arrows:
            # arrow tips need one path per connection
            out += [f"\\draw[{connection}] {seg};" for seg in lines]
        elif lines:
            from ..elements import wrap_tokens

            out += wrap_tokens(f"\\draw[{connection}]", " ".join(lines).split(" "))
        for layer, i, j in highlighted:
            h = self._highlight_edges[(layer, i, j)]
            lab = ""
            if h["label"] is not None:
                lab = f" node[pos=0.35, sloped, above, font=\\scriptsize, text={h['color']}] {{{label_text(h['label'])}}}"
            out.append(
                f"\\draw[{connection}, {h['color']}, line width={h['width']:g}pt] "
                f"({nm}-{layer}-{i}) --{lab} ({nm}-{layer + 1}-{j});"
            )
        for layer, i, j, text, pos in self._edge_labels:
            out.append(
                f"\\path ({nm}-{layer}-{i}) -- node[pos={fmt(pos)}, sloped, above, font=\\scriptsize, "
                f"inner sep=1pt] {{{label_text(text)}}} ({nm}-{layer + 1}-{j});"
            )

        # external inputs / outputs
        gap = 0.85
        for which, spec in (("in", self.inputs), ("out", self.outputs)):
            if spec is None:
                continue
            layer = 0 if which == "in" else len(self.sizes) - 1
            n = self.sizes[layer]
            for i in self._visible(layer):
                if i is None:
                    continue
                text = spec[i] if isinstance(spec, (list, tuple)) else _fill_index(spec, i + 1, n, layer)
                p = self.position(layer, i)
                if self.direction == "right":
                    q = p + Point(-gap - self.neuron_size / 2, 0) if which == "in" else p + Point(gap + self.neuron_size / 2, 0)
                    anchor = "east" if which == "in" else "west"
                else:
                    q = p + Point(0, -gap - self.neuron_size / 2) if which == "in" else p + Point(0, gap + self.neuron_size / 2)
                    anchor = "north" if which == "in" else "south"
                label_name = f"{nm}-{which}-{i}"
                out.append(f"\\node[anchor={anchor}, inner sep=2pt, align=center] ({label_name}) at {q.tikz()} {{{label_text(text)}}};")
                if which == "in":
                    out.append(f"\\draw[->, shorten >=1pt] ({label_name}) -- ({nm}-{layer}-{i});")
                else:
                    out.append(f"\\draw[->, shorten <=1pt] ({nm}-{layer}-{i}) -- ({label_name});")

        # titles
        titles = self._titles()
        if titles:
            if self.direction == "right":
                top = max(self._pos(k, 0).y for k in range(len(self.sizes))) + self.neuron_size / 2 + 0.25
                for layer, title in enumerate(titles[: len(self.sizes)]):
                    if title is None:
                        continue
                    x = self._pos(layer, 0).x
                    color = theme.stroke(self._layer_color(layer, ctx))
                    out.append(
                        f"\\node[{title_style}, anchor=south, text={color}] ({nm}-title-{layer}) "
                        f"at ({fmt(x)},{fmt(top)}) {{{label_text(title)}}};"
                    )
            else:
                left = min(self._pos(k, 0).x for k in range(len(self.sizes))) - self.neuron_size / 2 - 0.3
                for layer, title in enumerate(titles[: len(self.sizes)]):
                    if title is None:
                        continue
                    y = self._pos(layer, 0).y
                    color = theme.stroke(self._layer_color(layer, ctx))
                    out.append(
                        f"\\node[{title_style}, anchor=east, text={color}] ({nm}-title-{layer}) "
                        f"at ({fmt(left)},{fmt(y)}) {{{label_text(title)}}};"
                    )
        return out
