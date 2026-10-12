"""Trees (decision trees, parse trees, hierarchies) with a tidy layout."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from .._fmt import fmt, label_text, node_text
from ..coords import NodeRef, Point, as_point
from ..elements import Element
from ..layout import tidy_tree_layout
from ..color import color_text
from ..render import LazyName
from ..theme import ARG

if TYPE_CHECKING:  # pragma: no cover
    from ..render import RenderContext

__all__ = ["Tree", "TreeNode"]


class TreeNode:
    """A tree node: ``TreeNode("x < 3?", [child1, child2], edge_label="yes")``.

    ``color`` picks a palette color, ``style`` adds TikZ options and
    ``name`` lets you reference the node later via :meth:`Tree.node`.
    """

    def __init__(
        self,
        text: Any,
        children: Sequence[Any] = (),
        *,
        edge_label: Any = None,
        color: str | None = None,
        style: str | None = None,
        name: str | None = None,
        width: float | None = None,
    ):
        self.text = text
        self.children = [as_tree(c) for c in children]
        self.edge_label = edge_label
        self.color = color
        self.style = style
        self.name = name
        self.width = width

    def walk(self) -> Iterator["TreeNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def __repr__(self) -> str:
        return f"TreeNode({self.text!r}, {len(self.children)} children)"


def as_tree(spec: Any) -> TreeNode:
    """Build a :class:`TreeNode` from nested tuples: ``("root", child, ...)``.

    A plain value is a leaf; a tuple's first item is the label and the rest
    are its children.
    """
    if isinstance(spec, TreeNode):
        return spec
    if isinstance(spec, (tuple, list)):
        if not spec:
            raise ValueError("empty tree specification")
        return TreeNode(spec[0], [as_tree(c) for c in spec[1:]])
    return TreeNode(spec)


_MACRO = re.compile(r"\\[A-Za-z]+")


def estimate_width(text: Any, char_width: float = 0.19, padding: float = 0.35) -> float:
    """Rough width (cm) of a node label: TikZ measures text, Python cannot."""
    tex, _ = node_text(text)
    lines = tex.split(r"\\")
    best = 0.0
    for line in lines:
        is_math = "$" in line
        stripped = _MACRO.sub("x", line)
        stripped = re.sub(r"[{}$^_]", "", stripped)
        width = len(stripped) * (char_width * (0.85 if is_math else 1.0))
        best = max(best, width)
    return best + padding


class Tree(Element):
    """A tree laid out automatically.

    >>> t = Tree(("root", ("a", "a1", "a2"), "b"), grow="down")

    Parameters
    ----------
    root:
        A :class:`TreeNode` or nested tuples.
    grow:
        ``"down"`` (root at the top), ``"up"``, ``"right"`` or ``"left"``.
    level_distance:
        Distance between levels (cm).
    gap:
        Minimum gap between neighbouring nodes (cm).
    node_width:
        Fixed node width for the layout; estimated from the labels if omitted.
    style:
        Node style: ``"box"`` (rounded rectangles, default), ``"circle"`` or
        ``"plain"`` (text only).
    """

    def __init__(
        self,
        root: Any,
        *,
        at: Any = (0, 0),
        grow: str = "down",
        level_distance: float = 1.5,
        gap: float = 0.35,
        node_width: float | None = None,
        style: str = "box",
        color: str = "Blue",
        leaf_color: str | None = None,
        edge_color: Any = None,
        edge_width: float = 0.7,
        arrows: bool = False,
        font: str = r"\small",
        edge_label_font: str = r"\footnotesize",
        name: str | None = None,
    ):
        self.root = as_tree(root)
        self.origin = as_point(at)
        if grow not in ("down", "up", "right", "left"):
            raise ValueError("grow must be 'down', 'up', 'right' or 'left'")
        self.grow = grow
        self.level_distance = level_distance
        self.gap = gap
        self.node_width = node_width
        self.style = style
        self.color = color
        self.leaf_color = leaf_color
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.arrows = arrows
        self.font = font
        self.edge_label_font = edge_label_font
        self._lname = LazyName(self, "t", name)
        self._suffix: dict[int, str] = {}
        for k, node in enumerate(self.root.walk()):
            self._suffix[id(node)] = f"-{node.name or k}"

    @property
    def name(self) -> str:
        """Prefix of the node names (``t``, ``t2``, ... unless given)."""
        return self._lname.resolve()

    def _node_name(self, node: TreeNode) -> str:
        return self.name + self._suffix[id(node)]

    def node(self, key: Any) -> NodeRef:
        """Reference a node by its ``name`` or by its pre-order index."""
        for k, node in enumerate(self.root.walk()):
            if node.name == key or (isinstance(key, int) and k == key) or node is key:
                return NodeRef(self._lname.sub(self._suffix[id(node)]))
        raise KeyError(key)

    def positions(self) -> dict[int, Point]:
        horizontal = self.grow in ("right", "left")

        def width(node: TreeNode) -> float:
            if node.width is not None:
                return node.width
            if horizontal:
                return 0.75 * max(1, len(node_text(node.text)[0].split(r"\\")))
            if self.node_width is not None:
                return self.node_width
            return max(0.65, estimate_width(node.text))

        raw = tidy_tree_layout(self.root, lambda n: n.children, width=width, gap=self.gap,
                               level_distance=self.level_distance)
        out = {}
        for key, p in raw.items():
            if self.grow == "down":
                q = Point(p.x, p.y)
            elif self.grow == "up":
                q = Point(p.x, -p.y)
            elif self.grow == "right":
                q = Point(-p.y, -p.x)
            else:
                q = Point(p.y, -p.x)
            out[key] = self.origin + q
        return out

    def render(self, ctx: "RenderContext") -> list[str]:
        theme = ctx.theme
        if self.style == "circle":
            body = "circle, minimum size=7mm, inner sep=1pt"
        elif self.style == "plain":
            body = "inner sep=2pt"
        else:
            body = f"rounded corners={fmt(theme.corner_radius, 2)}pt, minimum height=6.5mm, inner sep=4pt, align=center"
        pre = self.name
        if self.style == "plain":
            node_style = ctx.define_style("tree node", f"{theme.arg_color()}, {body}, font={self.font}, text={theme.stroke(ARG)}",
                                          default=self.color, prefix=pre)
        else:
            node_style = ctx.define_style(
                "tree node",
                f"{theme.arg_color()}, {body}, draw={theme.stroke(ARG)}, fill={theme.fill(ARG, 18)}, line width=0.7pt, "
                f"font={self.font}",
                default=self.color,
                prefix=pre,
            )
        ec = self.edge_color or f"{theme.ink}!60"
        tip = ", ->, shorten >=1pt" if self.arrows else ""
        edge_style = ctx.define_style("tree edge", f"{ec}, line width={self.edge_width:g}pt{tip}", prefix=pre)
        label_style = ctx.define_style(
            "tree label", f"font={self.edge_label_font}, fill={theme.paper or 'white'}, inner sep=1.5pt", prefix=pre
        )
        pos = self.positions()
        out = [f"% tree {self.name}"]
        for node in self.root.walk():
            p = pos[id(node)]
            color = color_text(node.color or (self.leaf_color if (not node.children and self.leaf_color) else self.color))
            extra = f", {node.style}" if node.style else ""
            text, multi = node_text(node.text)
            if multi and self.style == "circle":
                extra += ", align=center"
            out.append(f"\\node[{node_style}={color}{extra}] ({self._node_name(node)}) at ({fmt(p.x)},{fmt(p.y)}) {{{text}}};")
        for node in self.root.walk():
            for child in node.children:
                lab = ""
                if child.edge_label is not None:
                    lab = f" node[{label_style}, midway, align=center] {{{label_text(child.edge_label)}}}"
                out.append(f"\\draw[{edge_style}] ({self._node_name(node)}) --{lab} ({self._node_name(child)});")
        return out
