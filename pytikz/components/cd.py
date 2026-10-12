"""Commutative diagrams."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from .._fmt import fmt
from ..coords import NodeRef, Point, as_point
from ..elements import Element
from ..render import LazyName

if TYPE_CHECKING:  # pragma: no cover
    from ..render import RenderContext

__all__ = ["CommutativeDiagram"]

#: Arrow kinds -> TikZ options.
ARROW_KINDS = {
    "->": "{arrow}",
    "to": "{arrow}",
    "hook": "{arrow}, {{Hooks[right,length=2.2pt]}}->",
    "hook'": "{arrow}, {{Hooks[left,length=2.2pt]}}->",
    "epi": "{arrow}, ->>",
    "mono": "{arrow}, >->",
    "mapsto": "{arrow}, |->",
    "dashed": "{arrow}, dashed",
    "dotted": "{arrow}, dotted",
    "equal": "{line}, double, double distance=1.4pt",
    "line": "{line}",
    "<-": "{arrow}, <-",
    "<->": "{arrow}, <->",
    "iso": "{arrow}",
    "implies": "{line}, double, double distance=1.6pt, -{{Implies}}",
}


class CommutativeDiagram(Element):
    """Objects on a grid joined by labelled arrows.

    >>> cd = CommutativeDiagram([["A", "B"], ["C", "D"]])
    >>> cd.arrow((0, 0), (0, 1), "f")          # by grid position
    >>> cd.arrow("A", "C", "g", swap=True)    # or by (unique) label

    Entries are typeset in math mode. Arrow kinds: ``"->"``, ``"hook"``,
    ``"epi"``, ``"mono"``, ``"mapsto"``, ``"dashed"``, ``"equal"``,
    ``"iso"``, ``"<->"``, ``"implies"``. Labels go on the left of the
    direction of travel (above a rightward arrow); ``swap=True`` flips them.
    """

    def __init__(
        self,
        entries: Sequence[Sequence[Any]],
        *,
        at: Any = (0, 0),
        col_sep: float = 2.4,
        row_sep: float = 2.0,
        name: str | None = None,
        math: bool = True,
        font: str = "",
        arrow_tip: str = "Computer Modern Rightarrow[length=3.2pt, width=5.6pt]",
    ):
        self.entries = [list(r) for r in entries]
        self.origin = as_point(at)
        self.col_sep = col_sep
        self.row_sep = row_sep
        self._lname = LazyName(self, "cd", name)
        self.math = math
        self.font = font
        self.arrow_tip = arrow_tip
        self._arrows: list[dict[str, Any]] = []
        self._extra: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Prefix of the entry names (``cd``, ``cd2``, ... unless given)."""
        return self._lname.resolve()

    # -- addressing ------------------------------------------------------------------
    def _find(self, key: Any) -> tuple[int, int]:
        if isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, int) for k in key):
            return key
        hits = [(i, j) for i, row in enumerate(self.entries) for j, e in enumerate(row) if e is not None and str(e) == str(key)]
        if len(hits) != 1:
            raise KeyError(f"{key!r} matches {len(hits)} entries; use a (row, column) position")
        return hits[0]

    def node(self, key: Any) -> NodeRef:
        """Reference to an entry (by position or unique label)."""
        i, j = self._find(key)
        return NodeRef(self._lname.sub(f"-{i}-{j}"))

    def position(self, key: Any) -> Point:
        i, j = self._find(key)
        return self.origin + Point(j * self.col_sep, -i * self.row_sep)

    # -- arrows ----------------------------------------------------------------------------
    def arrow(
        self,
        source: Any,
        target: Any,
        label: Any = None,
        *,
        kind: str = "->",
        swap: bool = False,
        bend: float | None = None,
        color: str | None = None,
        label_pos: float = 0.5,
        other_label: Any = None,
        shift: float | None = None,
    ) -> "CommutativeDiagram":
        """Add an arrow between two entries.

        ``other_label`` is written on the opposite side; ``shift`` (cm)
        offsets the arrow sideways (for parallel arrows).
        """
        if kind not in ARROW_KINDS:
            raise ValueError(f"unknown arrow kind {kind!r}; choose from {sorted(ARROW_KINDS)}")
        for end in (source, target):
            i, j = self._find(end)
            if i >= len(self.entries) or j >= len(self.entries[i]) or self.entries[i][j] is None:
                raise ValueError(f"no object at {end!r}: arrows must join non-empty cells")
        if kind == "iso" and other_label is None:
            other_label = r"\sim"
        self._arrows.append(dict(source=self._find(source), target=self._find(target), label=label, kind=kind,
                                 swap=swap, bend=bend, color=color, pos=label_pos, other=other_label, shift=shift))
        return self

    # -- rendering ---------------------------------------------------------------------------
    def _math(self, text: Any) -> str:
        if text is None:
            return ""
        text = str(text)
        if self.math and not text.startswith("$"):
            return f"${text}$"
        return text

    def render(self, ctx: "RenderContext") -> list[str]:
        arrow = ctx.define_style("cd arrow", f"line width=0.5pt, >={{{self.arrow_tip}}}, ->, shorten <=2pt, "
                                 "shorten >=2pt", prefix=self.name)
        line = ctx.define_style("cd line", "line width=0.5pt, shorten <=2pt, shorten >=2pt", prefix=self.name)
        label = ctx.define_style("cd label", "font=\\small, inner sep=2.5pt", prefix=self.name)
        out = [f"% commutative diagram {self.name}"]
        font = f"font={self.font}, " if self.font else ""
        for i, row in enumerate(self.entries):
            for j, e in enumerate(row):
                if e is None:
                    continue
                p = self.origin + Point(j * self.col_sep, -i * self.row_sep)
                out.append(f"\\node[{font}inner sep=3pt] ({self.name}-{i}-{j}) at ({fmt(p.x)},{fmt(p.y)}) {{{self._math(e)}}};")
        for a in self._arrows:
            (i0, j0), (i1, j1) = a["source"], a["target"]
            src, dst = f"{self.name}-{i0}-{j0}", f"{self.name}-{i1}-{j1}"
            opts = ARROW_KINDS[a["kind"]].format(arrow=arrow, line=line)
            if a["color"]:
                opts += f", {a['color']}"
            if a["shift"]:
                p0 = self.origin + Point(j0 * self.col_sep, -i0 * self.row_sep)
                p1 = self.origin + Point(j1 * self.col_sep, -i1 * self.row_sep)
                n = (p1 - p0).unit().perp() * a["shift"]
                opts += f", transform canvas={{shift={{({fmt(n.x)},{fmt(n.y)})}}}}"
            conn = "--"
            if a["bend"]:
                side = "left" if a["bend"] > 0 else "right"
                conn = f"to[bend {side}={fmt(abs(a['bend']))}]"
            labels = ""
            if a["label"] is not None:
                side = "auto, swap" if a["swap"] else "auto"
                text_color = f", text={a['color']}" if a["color"] else ""
                labels += f" node[{label}, {side}, pos={fmt(a['pos'])}{text_color}] {{{self._math(a['label'])}}}"
            if a["other"] is not None:
                side = "auto" if a["swap"] else "auto, swap"
                labels += f" node[{label}, {side}, pos={fmt(a['pos'])}] {{{self._math(a['other'])}}}"
            out.append(f"\\draw[{opts}] ({src}) {conn}{labels} ({dst});")
        return out
