"""The :class:`Picture`: root container, TikZ/LaTeX generation and output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path as FilePath
from typing import Any, Iterable

from .color import Color, registered_colors
from .container import Container, _indent
from .options import Options
from .render import RenderContext, rendering
from .style import Style
from .theme import Theme, get_theme

__all__ = ["Picture", "detect_libraries"]

#: Packages loaded in every standalone document.
BASE_PACKAGES = (
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{lmodern}",
    r"\usepackage{amsmath,amssymb,bm}",
)

def _words(*words: str) -> str:
    """Regex for option words standing on their own in an option list,
    a style definition or a ``shape=`` value."""
    alternatives = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in words)
    return r"(?:^|[\[,{=])\s*(?:" + alternatives + r")\s*(?=[,\]}=]|$)"


_GEOMETRIC = ("diamond", "ellipse", "trapezium", "semicircle", "regular polygon", "star", "isosceles triangle",
              "kite", "dart", "circular sector", "cylinder")
_MISC = ("cross out", "strike out", "rounded rectangle", "chamfered rectangle")
_SYMBOLS = ("correct forbidden sign", "forbidden sign", "magnifying glass", "cloud", "starburst", "signal",
            "tape", "magnetic tape")

_LIBRARY_PATTERNS: list[tuple[str, str]] = [
    ("arrows.meta", r"Stealth|Latex|Triangle|Kite|Hooks|Straight Barb|Rays|Classical TikZ|Computer Modern|\{Bar|Bar\}"
                    r"|Implies|Arc Barb|Tee Barb|Square\[|Circle\[|Diamond\["),
    ("bending", r"[\[,]\s*(?:bend|flex'?|quick)\s*[,\]]"),
    ("calc", r"\(\$"),
    ("positioning", r"\b(?:above|below|left|right)(?: (?:left|right))?\s*=\s*\{?[^,\]\[]*?\bof\b"),
    ("fit", r"\bfit\s*="),
    ("backgrounds", r"on background layer|background rectangle|background grid|background top"),
    ("decorations.pathreplacing", r"\bbrace\b|show path construction|\bticks\s*[,\]}]|\bexpanding waves\b"),
    ("decorations.pathmorphing", r"\b(?:zigzag|snake|coil|bumps|random steps|saw)\b"),
    ("decorations.markings", r"\bmarkings\b"),
    ("decorations.text", r"text along path"),
    ("shapes.geometric", _words(*_GEOMETRIC) + r"|regular polygon sides|trapezium (?:left |right )?angle"
                         r"|star points|isosceles triangle apex angle|cylinder uses custom fill"),
    ("shapes.misc", _words(*_MISC)),
    ("shapes.symbols", _words(*_SYMBOLS)),
    ("shapes.arrows", r"single arrow|double arrow|arrow box"),
    ("shapes.multipart", r"circle split|rectangle split|ellipse split|circle solidus"),
    ("shapes.callouts", r"rectangle callout|ellipse callout|cloud callout|callout absolute pointer|callout relative pointer"),
    ("patterns.meta", r"pattern\s*=\s*\{\s*(?:Lines|Hatch|Dots|Stars)"),
    ("patterns", r"pattern\s*=\s*(?!\{\s*(?:Lines|Hatch|Dots|Stars))"),
    ("shadows", r"drop shadow|copy shadow|circular glow"),
    ("automata", _words("state", "initial", "accepting", "state with output", "state without output")
                 + r"|initial (?:where|text|distance|by arrow|by diamond)|accepting (?:by arrow|by double|where|text|distance)"),
    ("matrix", r"matrix of"),
    ("fadings", r"path fading|scope fading|fit fading"),
    ("intersections", r"name path|name intersections"),
    ("through", r"circle through"),
    ("angles", r"pic\s*(?:\[[^\]]*\])?\s*\{\s*(?:right )?angle"),
    ("quotes", r"[\[,]\s*\"[^\"]*\""),
    ("3d", r"canvas is (?:xy|yz|zx|xz) plane"),
]

_ORDER = [name for name, _ in _LIBRARY_PATTERNS]


def detect_libraries(text: str) -> list[str]:
    """TikZ libraries that ``text`` appears to need (best effort)."""
    return [name for name, pattern in _LIBRARY_PATTERNS if re.search(pattern, text, re.M)]


def _style_tokens(text: str) -> set[str]:
    """Every word that could name a style: keys and values of option lists."""
    tokens: set[str] = set()
    for piece in re.split(r"[\[\]{},=]", text):
        piece = piece.strip()
        if not piece:
            continue
        if "/." in piece:
            piece = piece.split("/.", 1)[0]
        tokens.add(piece)
    return tokens


#: Layer under everything (incl. ``on background layer``) for page colors.
PAGE_LAYER = "pytikzpage"
#: Margin (pt) of the page color around the picture.
PAGE_MARGIN = 8


_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*")


@dataclass
class _Built:
    body: list[str]
    picture_options: str
    styles: list[Style]
    colors: list[tuple[str, str, str]]
    libraries: list[str]
    packages: list[tuple[str, str | None]]


class Picture(Container):
    """A TikZ picture.

    Build it with the drawing methods (:meth:`node`, :meth:`draw`,
    :meth:`arrow`, ...) or by adding components, then write it out::

        pic = Picture()
        pic.node("Hello", style="block")
        pic.save("hello.tex")      # standalone LaTeX document
        pic.save("hello.tikz")     # just the picture, to \\input in a paper
        pic.render("hello.pdf")    # also .png and .svg

    ``theme`` selects colors/fonts (see :mod:`pytikz.theme`), ``scale``
    scales all coordinates, ``border`` is the margin of the standalone page in
    points and any extra keyword arguments become tikzpicture options.
    """

    def __init__(
        self,
        *,
        theme: str | Theme | None = None,
        scale: float | None = None,
        font: str | None = None,
        border: float = 6.0,
        background: Any = None,
        libraries: Iterable[str] = (),
        packages: Iterable[str | tuple[str, str]] = (),
        preamble: str | None = None,
        engine: str = "pdflatex",
        **options: Any,
    ):
        super().__init__()
        self.theme = get_theme(theme)
        self.scale = scale
        self.font = font
        self.border = border
        self.page_color = background
        self.engine = engine
        self.options = Options(**options)
        self.styles: dict[str, Style] = {}
        self._libraries: list[str] = list(libraries)
        self._packages: list[tuple[str, str | None]] = []
        for pkg in packages:
            self.use_package(*(pkg if isinstance(pkg, tuple) else (pkg,)))
        self._preamble: list[str] = [preamble] if preamble else []
        self._colors: dict[str, str] = {}

    # -- configuration ------------------------------------------------------------
    def style(self, name: str, *args: Any, default: str | None = None, **kwargs: Any) -> Style:
        """Define (or override) a named style, usable as ``style=name``."""
        style = Style(name, *args, default=default, **kwargs)
        self.styles[name] = style
        return style

    def define_color(self, name: str, value: Any) -> Color:
        """Define a named color for this picture (overrides palette colors)."""
        color = Color(value) if not isinstance(value, Color) else value
        if color.rgb is None:
            raise ValueError(f"cannot define {name!r}: unknown RGB value for {value!r}")
        hex_code = color.hex.lstrip("#")  # type: ignore[union-attr]
        self._colors[name] = hex_code
        return Color(name) if name in self.theme.palette else Color("#" + hex_code, name=name)

    def use_library(self, *names: str) -> None:
        """Load TikZ libraries explicitly (most are detected automatically)."""
        for name in names:
            if name not in self._libraries:
                self._libraries.append(name)

    def use_package(self, name: str, options: str | None = None) -> None:
        """Load a LaTeX package in the standalone document."""
        if all(p[0] != name for p in self._packages):
            self._packages.append((name, options))

    def add_preamble(self, tex: str) -> None:
        """Extra preamble code for the standalone document."""
        self._preamble.append(tex)

    # -- rendering ------------------------------------------------------------------
    def _picture_options(self) -> Options:
        opts = self.theme.picture_options()
        if self.font is not None:
            if self.font:
                opts.set("execute at begin picture", "{" + self.font + "}")
            else:
                opts.remove("execute at begin picture")
        if self.scale is not None:
            opts.set("scale", self.scale)
        opts.add(self.options)
        return opts

    def _build(self) -> _Built:
        # Discovery pass: rendering marks every node that something refers
        # to, so the real pass can give exactly those nodes a name -- even
        # when the reference is only resolved lazily (inline nodes, components).
        probe = RenderContext(self)
        probe.probe = True
        with rendering(probe):
            self.render_children(probe)
        ctx = RenderContext(self)
        ctx.reserve(probe.claimed)  # automatic names never take a name chosen later on
        with rendering(ctx):
            body = self.render_children(ctx)
            pic_opts = self._picture_options().render()
            styles = self._used_styles(ctx, "\n".join(body) + pic_opts)
            style_text = "\n".join(line for s in styles for line in s.definitions())
        everything = "\n".join(body) + pic_opts + style_text + " " + (self._page() or "")
        colors = self._used_colors(everything)
        libraries: list[str] = []
        for lib in [*self._libraries, *ctx.libraries, *detect_libraries(everything)]:
            if lib not in libraries:
                libraries.append(lib)
        libraries.sort(key=lambda n: _ORDER.index(n) if n in _ORDER else len(_ORDER))
        packages = list(self._packages)
        for pkg in ctx.packages:
            if all(p[0] != pkg[0] for p in packages):
                packages.append(pkg)
        return _Built(body, pic_opts, styles, colors, libraries, packages)

    def _used_styles(self, ctx: RenderContext, text: str) -> list[Style]:
        candidates: dict[str, Style] = {s.name: s for s in self.theme.styles()}
        candidates.update(ctx.styles)
        candidates.update(self.styles)
        used: list[str] = []
        pending = text
        while True:
            tokens = _style_tokens(pending)
            new = [name for name in candidates if name in tokens and name not in used]
            if not new:
                break
            used.extend(new)
            pending = "\n".join(line for n in new for line in candidates[n].definitions())
        return [candidates[name] for name in candidates if name in used]

    def _used_colors(self, text: str) -> list[tuple[str, str, str]]:
        """``(name, model, spec)`` for every color the text refers to.

        Semantic role names of the theme (``input``, ``conv``, ...) become
        ``\\colorlet`` aliases of palette colors, so ``block=input`` follows
        the theme.
        """
        tokens = set(_NAME_TOKEN.findall(text))
        sources: list[dict[str, str]] = [
            self._colors,
            self.theme.palette,
            {name: spec for name, (_, spec, _) in registered_colors().items()},
        ]
        defined = {name for source in sources for name in source}
        aliases = []
        for role, target in self.theme.roles.items():
            if role in tokens and role not in defined:
                base = target.split("!")[0]
                tokens.add(base)
                aliases.append((role, "colorlet", target))
        out: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for source in sources:
            for name, code in source.items():
                if name in tokens and name not in seen:
                    seen.add(name)
                    out.append((name, "HTML", code.upper()))
        return out + aliases

    def tikz(self) -> str:
        """The picture as embeddable code: color/style definitions followed by
        the ``tikzpicture`` environment. ``\\input`` it in a document that loads
        TikZ and the libraries listed in the header comment."""
        built = self._build()
        lines = ["% Generated by pytikz."]
        requirements = [r"\usepackage{tikz}"]
        if built.libraries:
            requirements.append(r"\usetikzlibrary{" + ", ".join(built.libraries) + "}")
        lines.append("% Requires: " + " ".join(requirements))
        lines.extend(self._definitions(built, trailing="%"))
        lines.extend(self._environment(built))
        lines[-1] += "%"
        return "\n".join(lines) + "\n"

    def _definitions(self, built: _Built, trailing: str = "") -> list[str]:
        lines = [
            f"\\colorlet{{{n}}}{{{c}}}{trailing}" if m == "colorlet" else f"\\definecolor{{{n}}}{{{m}}}{{{c}}}{trailing}"
            for n, m, c in built.colors
        ]
        if built.styles:
            lines.append(r"\tikzset{" + trailing)
            for style in built.styles:
                for definition in style.definitions():
                    lines.append(f"  {definition},")
            lines.append("}" + trailing)
        return lines

    def _page(self) -> str | None:
        """The page color: ``Picture(background=...)`` or the theme's paper."""
        if self.page_color is not None:
            return str(self.page_color)
        return self.theme.paper

    def _environment(self, built: _Built) -> list[str]:
        body = list(built.body)
        page = self._page()
        if page:
            # Paint the page color on a layer below everything, including
            # ``on background layer`` content; the layer list is set inside
            # this picture only.
            layers = [PAGE_LAYER] + (["background"] if "backgrounds" in built.libraries else []) + ["main"]
            m = PAGE_MARGIN
            body = [
                rf"\ifcsname pgf@layerbox@{PAGE_LAYER}\endcsname\else\pgfdeclarelayer{{{PAGE_LAYER}}}\fi",
                rf"\pgfsetlayers{{{','.join(layers)}}}",
                *body,
                rf"\begin{{pgfonlayer}}{{{PAGE_LAYER}}}",
                rf"  \fill[{page}] ([shift={{(-{m}pt,-{m}pt)}}]current bounding box.south west) rectangle "
                rf"([shift={{({m}pt,{m}pt)}}]current bounding box.north east);",
                r"\end{pgfonlayer}",
            ]
        return [
            rf"\begin{{tikzpicture}}{built.picture_options}",
            *_indent(body),
            r"\end{tikzpicture}",
        ]

    def document(self) -> str:
        """A complete standalone LaTeX document containing the picture."""
        built = self._build()
        border = f"{self.border:g}pt"
        lines = [rf"\documentclass[tikz, border={border}]{{standalone}}"]
        unicode_engine = self.engine in ("lualatex", "xelatex")
        for line in BASE_PACKAGES:
            if unicode_engine and "fontenc" in line:
                continue
            lines.append(line)
        lines.extend(self.theme.preamble)
        for name, opts in built.packages:
            lines.append(rf"\usepackage[{opts}]{{{name}}}" if opts else rf"\usepackage{{{name}}}")
        if built.libraries:
            lines.append(r"\usetikzlibrary{" + ", ".join(built.libraries) + "}")
        lines.extend(self._preamble)
        lines.extend(self._definitions(built))
        lines.append(r"\begin{document}")
        lines.extend(self._environment(built))
        lines.append(r"\end{document}")
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.tikz()

    def __repr__(self) -> str:
        return f"<Picture: {len(self.children)} elements, theme={self.theme.name!r}>"

    # -- output -----------------------------------------------------------------------
    def save(self, path: str | FilePath) -> FilePath:
        """Write ``.tex`` (standalone document) or ``.tikz`` (embeddable code)."""
        target = FilePath(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".tikz":
            target.write_text(self.tikz())
        else:
            target.write_text(self.document())
        return target

    def render(
        self,
        path: str | FilePath,
        *,
        dpi: int = 300,
        transparent: bool = False,
        engine: str | None = None,
        cache: bool = True,
    ) -> FilePath:
        """Compile with LaTeX and write ``.pdf``, ``.png`` or ``.svg``.

        ``.tex``/``.tikz`` paths are written with :meth:`save` instead.
        Raises :class:`~pytikz.TikzCompileError` with the relevant part of the
        LaTeX log if compilation fails.
        """
        from .compiler import render_document

        target = FilePath(path)
        if target.suffix in (".tex", ".tikz"):
            return self.save(target)
        return render_document(
            self.document(),
            target,
            engine=engine or self.engine,
            dpi=dpi,
            transparent=transparent,
            cache=cache,
        )

    def to_svg(self) -> str:
        """The rendered picture as SVG text."""
        from .compiler import compile_to_bytes

        return compile_to_bytes(self.document(), "svg", engine=self.engine).decode("utf-8")

    def to_png(self, dpi: int = 200) -> bytes:
        """The rendered picture as PNG bytes."""
        from .compiler import compile_to_bytes

        return compile_to_bytes(self.document(), "png", engine=self.engine, dpi=dpi)

    def _repr_svg_(self) -> str:  # Jupyter
        return self.to_svg()
