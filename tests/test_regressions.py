"""Regression tests for problems found in review (naming, colors, escaping,
log axes, backgrounds, library detection, ...)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

import pytikz as tz
from pytikz.picture import detect_libraries

needs_latex = pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")


def lines(pic):
    return [ln.strip() for ln in pic.tikz().splitlines()]


def compiles(pic, tmp_path, name="x"):
    out = pic.render(tmp_path / f"{name}.pdf", cache=False)
    assert out.stat().st_size > 500


# ---------------------------------------------------------------- naming
def test_auto_names_avoid_names_claimed_later():
    pic = tz.Picture()
    a = pic.node("A")
    b = pic.node("B", at=(3, 0), name="n1")
    pic.arrow(a, b)
    code = lines(pic)
    assert r"\node (n2) {A};" in code
    assert r"\draw[->] (n2) -- (n1);" in code


def test_duplicate_explicit_names_warn():
    pic = tz.Picture()
    pic.node("x", name="dup")
    pic.node("y", name="dup")
    with pytest.warns(UserWarning, match="dup"):
        pic.tikz()


def test_components_get_unique_default_names():
    pic = tz.Picture()
    A = pic.add(tz.Matrix([[1, 2], [3, 4]]))
    B = pic.add(tz.Matrix([[1, 2], [3, 4]], at=(4, 0)))
    pic.arrow(A.entry(0, 0), B.entry(1, 1))
    g1 = pic.add(tz.Graph(edges=[(0, 1)], layout="circular"))
    g2 = pic.add(tz.Graph(edges=[(0, 1)], layout="circular", at=(6, 0)))
    pic.arrow(g1.node(0), g2.node(1))
    code = lines(pic)
    assert r"\draw[->] (M-1-1) -- (M2-2-2);" in code
    assert r"\draw[->] (g-0) -- (g2-1);" in code


def test_graph_node_names_are_unique():
    g = tz.Graph(["a b", "a-b", "a_b", 1, "1"], [("a b", "a-b"), (1, "1")], layout="circular", name="G")
    pic = tz.Picture()
    pic.add(g)
    code = pic.tikz()
    for name in ("G-a-b", "G-a-b-2", "G-a-b-3", "G-1", "G-1-2"):
        assert f"({name})" in code
    assert "(G-a-b) -- (G-a-b-2)" in code and "(G-1) -- (G-1-2)" in code


# ---------------------------------------------------------------- picture API
def test_background_method_and_page_color():
    pic = tz.Picture(background="Yellow!20")
    pic.background().circle((0, 0), 1, fill="Red")
    code = pic.tikz()
    assert r"\pgfsetlayers{pytikzpage,background,main}" in code
    assert r"\fill[Yellow!20]" in code


def test_dark_theme_page_color_is_below_background_layer():
    pic = tz.Picture(theme="dark")
    a = pic.node("A")
    pic.fit(a, title="G")
    code = pic.tikz()
    assert code.index("on background layer") < code.index(r"\begin{pgfonlayer}{pytikzpage}")
    assert "show background rectangle" not in code
    assert r"\definecolor{Paper}" in code


# ---------------------------------------------------------------- colors
def test_color_expressions_stay_valid():
    assert str(tz.Color("Blue!40").lighten(0.5)) == "Blue!40!white!50"
    assert str(tz.Color("Blue!40").lighten(0.5).darken(0.5)) == "Blue!40!white!50!white!50!black"
    mixed = tz.Color("Blue").mix("Red!50")
    assert "!" not in mixed.expr.split("!")[-1][:1] and mixed.expr.startswith("Blue!50!c")


def test_hex_and_tuple_colors_in_options():
    pic = tz.Picture()
    pic.node("x", fill="#E4572E", draw=(0.2, 0.4, 0.8), dashed=np.bool_(True))
    code = pic.tikz()
    assert r"\definecolor{cE4572E}{HTML}{E4572E}" in code
    assert "fill=cE4572E, draw=c3366CC, dashed" in code


def test_redefining_a_named_color_warns():
    tz.Color("#010203", name="RegressionColor")
    with pytest.warns(UserWarning, match="redefined"):
        tz.Color("#040506", name="RegressionColor")


def test_theme_with_copies_palette():
    base = tz.get_theme("modern")
    custom = base.with_(name="custom")
    custom.palette["Blue"] = "000000"
    assert base.palette["Blue"] != "000000"
    assert tz.get_theme("paper").palette is not base.palette


# ---------------------------------------------------------------- styles and libraries
def test_styles_used_as_values_are_emitted():
    pic = tz.Picture()
    with pic.scope(nodes="block") as s:
        s.node("x")
    assert "block/.style" in pic.tikz()


@pytest.mark.parametrize(
    "text, library",
    [
        ("diamond/.style={diamond, draw}", "shapes.geometric"),
        ("nodes={ellipse, draw}", "shapes.geometric"),
        ("shape=star", "shapes.geometric"),
        ("q/.style={state, fill=red}", "automata"),
        ('\\draw (a) to["x"] (b);', "quotes"),
        ("-{Stealth[bend]}", "bending"),
        ("rectangle callout", "shapes.callouts"),
    ],
)
def test_library_detection(text, library):
    assert library in detect_libraries(text)


def test_ellipse_path_operation_needs_no_shape_library():
    assert "shapes.geometric" not in detect_libraries(r"\draw (0,0) ellipse[x radius=1, y radius=2];")


# ---------------------------------------------------------------- text
def test_text_is_escaped_everywhere():
    pic = tz.Picture()
    pic.dot((0, 0), "conv_1")
    pic.fit((0, 0), (1, 1), title="a & b")
    ax = pic.add(tz.Axes(xlim=(0, 1), ylim=(0, 1), xticks={0: "cats & dogs", 1: "R^2"}))
    ax.plot([0, 1], [0, 1])
    hm = tz.Heatmap([[0.5, 0.25]], values=True, number_format="{:.0%}", at=(0, -3))
    pic.add(hm)
    code = pic.tikz()
    assert r"conv\_1" in code and r"a \& b" in code and r"cats \& dogs" in code
    assert r"R\textasciicircum{}2" in code and r"50\%" in code


def test_numbers_as_text():
    pic = tz.Picture()
    pic.node(1e-5)
    pic.node(0.1 + 0.2)
    pic.add(tz.Matrix([[0.1 + 0.2, 1.0]], name="N"))
    code = pic.tikz()
    assert "{1e-05}" in code and "{0.3}" in code and "0.3 & 1" in code


def test_relative_placement_to_a_point():
    pic = tz.Picture()
    pic.node("x", right_of=(1, 2))
    assert r"\node[right={of 1,2}] {x};" in lines(pic)


def test_auto_align_comes_first():
    pic = tz.Picture()
    pic.style("left text", "align=left")
    pic.node("a\nb", style="left text")
    assert r"\node[align=center, left text] {a\\b};" in lines(pic)


# ---------------------------------------------------------------- geometry helpers
def test_sector_around_a_node_uses_its_centre():
    pic = tz.Picture()
    n = pic.node("x", name="c")
    pic.sector(n, 1, 0, 90, fill="Red")
    assert r"\draw[fill=Red] (c.center) -- ($(c.center)+(1,0)$) arc" in pic.tikz()


def test_function_breaks_at_undefined_points():
    pic = tz.Picture()
    pic.function(lambda x: 1 / x, (-1, 1), samples=5)
    code = pic.tikz()
    assert "(-1,-1) -- (-0.5,-2) (0.5,2) -- (1,1);" in code
    with pytest.raises(ValueError):
        pic.function(lambda x: x, (0, 1), samples=1)


def test_line_accepts_arrays():
    pic = tz.Picture()
    pic.line(np.array([[0, 0], [1, 1], [2, 0]]))
    assert r"\draw (0,0) -- (1,1) -- (2,0);" in lines(pic)


def test_cd_arrow_to_empty_cell_fails_early():
    cd = tz.CommutativeDiagram([["A", None]])
    with pytest.raises(ValueError):
        cd.arrow((0, 0), (0, 1), "f")


# ---------------------------------------------------------------- axes
def test_log_axes_ignore_non_positive_values():
    ax = tz.Axes(yscale="log")
    ax.plot([1, 2, 3], [0, 10, 100])
    ax.bar([1, 2, 3], [0, 10, 100])
    ax.fill_between([1, 3], [1, 100])
    pic = tz.Picture()
    pic.add(ax)
    pic.tikz()
    assert ax.ylim[0] > 0
    ax2 = tz.Axes(xscale="log")
    ax2.function(np.sqrt, (0, 10))
    tz.Picture().add(ax2)


def test_poles_do_not_draw_vertical_lines():
    ax = tz.Axes(xlim=(-1, 1), ylim=(-3, 3), width=4, height=4)
    ax.function(lambda x: 1 / x, samples=200)
    pic = tz.Picture()
    pic.add(ax)
    for ln in lines(pic):
        if ln.startswith(r"\draw[color=Blue"):
            xs = [float(v) for v in __import__("re").findall(r"\((-?[\d.]+),", ln)]
            assert max(xs) - min(xs) < 2.01 or min(xs) >= 1.99  # one branch per path


def test_ticks_from_ranges_and_numpy_numbers():
    pic = tz.Picture()
    ax = pic.add(tz.Axes(xlim=(0, 10), ylim=(0, 10), xticks=range(0, 11, 5), yticks=np.int64(5)))
    ax.plot([0, 10], [0, 10])
    code = pic.tikz()
    assert "(ax1-xt2) at (8," in code and code.count("ax1-yt") >= 3


def test_array_contours_set_limits_and_nan_regions_stay_empty():
    Z = np.add.outer(np.arange(5.0), np.arange(6.0))
    Z[0, 0] = np.nan
    ax = tz.Axes()
    ax.contourf(Z, levels=[2, 4, 6])
    ax.colorbar()
    pic = tz.Picture()
    pic.add(ax)
    pic.tikz()
    assert ax.xlim[0] < 0.5 and ax.xlim[1] > 4.5


def test_legend_avoids_bars():
    ax = tz.Axes(width=6, height=4)
    ax.bar([0, 1, 2], [1, 2, 5], label="bars")
    ax.legend()
    pic = tz.Picture()
    pic.add(ax)
    pic.tikz()
    ax._autoscale()
    assert ax._best_location() == "upper left"


# ---------------------------------------------------------------- compile checks
@needs_latex
def test_previously_failing_inputs_compile(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTIKZ_CACHE", str(tmp_path / "cache"))
    pic = tz.Picture(theme="mono")
    pic.node("a", style="block=Blue!50", fill="#E4572E")
    pic.add(tz.NeuralNetwork([2, 2], colors=["Blue!50", "#E4572E"], at=(3, 0)))
    ax = pic.add(tz.Axes(at=(0, -5), xlim=(0, 4), yscale="log"))
    for k in range(4):
        ax.bar([k + 0.5], [10 ** k], width=0.5)   # mono cycle has expressions like Ink!60
    ax.plot([0.5, 3.5], [1, 1000], color="#123456")
    pic.dot((7, 0), "R^2 & co")
    pic.node("pt", right_of=(8, 0))
    pic.style("d", "diamond, draw")
    pic.node("x", at=(10, 0), style="d")
    pic.node("y", at=(11, 0), shape="ellipse", draw=True)
    pic.add(tz.Heatmap([[0.1, 0.9]], values=True, number_format="{:.0%}", at=(7, -3)))
    compiles(pic, tmp_path, "mixed")
    dark = tz.Picture(theme="dark")
    a = dark.node("A", style="block")
    dark.fit(a, title="Group")
    dark.arrow(a.east, (3, 0), label="on", label_side=None)
    compiles(dark, tmp_path, "dark")
