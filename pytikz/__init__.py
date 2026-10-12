"""pytikz -- write Python, get beautiful TikZ.

Build diagrams with Python code and compile them to clean, readable TikZ (and
PDF/PNG/SVG)::

    import pytikz as tz

    pic = tz.Picture()
    x = pic.node("$x$", at=(0, 0), style="block=Green")
    h = pic.node("Hidden", right_of=x, distance=1.2, style="block")
    y = pic.node(r"$\\hat y$", right_of=h, distance=1.2, style="block=Red")
    pic.arrow(x, h)
    pic.arrow(h, y)
    pic.render("tiny.pdf")
"""

from .color import Color, blend, hex_color, rgb
from .colormaps import Colormap, get_colormap
from .compiler import LatexNotFoundError, TikzCompileError, latex_available
from .container import Scope
from .coords import (
    ORIGIN,
    Coord,
    LazyCoord,
    NodeRef,
    Point,
    Point3,
    Polar,
    as_coord,
    as_point,
    between,
    midpoint,
    project,
    towards,
    xy,
)
from .elements import (
    Comment,
    Coordinate,
    Element,
    Node,
    Path,
    Raw,
    arc,
    circle,
    curve,
    ellipse,
    label,
    raw_op,
    rel,
    to,
)
from .options import Options
from .picture import Picture
from .style import Style
from .theme import THEMES, Theme, get_theme, register_theme
from .units import Length, cm, em, ex, inch, mm, pt
from ._fmt import tex_escape

from .components import *  # noqa: F401,F403  (Axes, NeuralNetwork, ...)
from .components import __all__ as _component_names

__version__ = "0.1.0"

__all__ = [
    "Picture",
    "Scope",
    "Node",
    "Coordinate",
    "Path",
    "Raw",
    "Comment",
    "Element",
    "Options",
    "Style",
    "Theme",
    "THEMES",
    "get_theme",
    "register_theme",
    "Color",
    "rgb",
    "hex_color",
    "blend",
    "Colormap",
    "get_colormap",
    "Coord",
    "Point",
    "Point3",
    "Polar",
    "NodeRef",
    "LazyCoord",
    "ORIGIN",
    "as_coord",
    "as_point",
    "between",
    "midpoint",
    "towards",
    "project",
    "xy",
    "to",
    "curve",
    "arc",
    "circle",
    "ellipse",
    "rel",
    "label",
    "raw_op",
    "Length",
    "cm",
    "mm",
    "pt",
    "em",
    "ex",
    "inch",
    "tex_escape",
    "TikzCompileError",
    "LatexNotFoundError",
    "latex_available",
    *_component_names,
]
