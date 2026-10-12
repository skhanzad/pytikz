"""Tests of the core: formatting, options, coordinates, colors, elements and pictures."""

import math

import pytest

import pytikz as tz
from pytikz._fmt import brace, fmt, node_text, smart_escape, split_top_level
from pytikz.color import Color
from pytikz.coords import NodeRef, Point, Polar, as_coord
from pytikz.elements import Path, path_tokens
from pytikz.options import Options
from pytikz.picture import detect_libraries


def body(pic):
    """The lines between \\begin{tikzpicture} and \\end{tikzpicture}."""
    lines = pic.tikz().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(r"\begin{tikzpicture}"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith(r"\end{tikzpicture}"))
    return [ln.strip() for ln in lines[start + 1 : end]]


# ---------------------------------------------------------------- formatting
@pytest.mark.parametrize(
    "value, expected",
    [(1, "1"), (1.0, "1"), (2.5, "2.5"), (1 / 3, "0.3333"), (-0.00001, "0"), (-1.23456, "-1.2346"), (1e-9, "0")],
)
def test_fmt(value, expected):
    assert fmt(value) == expected


def test_fmt_rejects_nan_and_bool():
    with pytest.raises(ValueError):
        fmt(float("nan"))
    with pytest.raises(TypeError):
        fmt(True)


def test_brace_only_when_needed():
    assert brace("red!20") == "red!20"
    assert brace("a,b") == "{a,b}"
    assert brace("(a) (1,2)") == "{(a) (1,2)}"  # parentheses do not protect commas
    assert brace("{already, braced}") == "{already, braced}"
    assert brace("[x]y") == "{[x]y}"
    assert brace("") == "{}"


def test_split_top_level():
    assert split_top_level("draw, fill={red!20}, label={[x]a,b}") == ["draw", "fill={red!20}", "label={[x]a,b}"]
    assert split_top_level("shift=(1,2), thick") == ["shift=(1,2)", "thick"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Add & Norm", r"Add \& Norm"),
        ("conv_1", r"conv\_1"),
        ("50%", r"50\%"),
        (r"50\%", r"50\%"),
        ("$x_1$ & co", r"$x_1$ \& co"),
        (r"\begin{tabular}{l} a & b \end{tabular}", r"\begin{tabular}{l} a & b \end{tabular}"),
    ],
)
def test_smart_escape(text, expected):
    assert smart_escape(text) == expected


def test_node_text_newlines():
    assert node_text("Multi-Head\nAttention") == (r"Multi-Head\\Attention", True)
    text, multi = node_text("$\\begin{aligned}a&=1\\\\\nb&=2\\end{aligned}$")
    assert r"\\" in text and not multi and text.count(r"\\") == 1
    assert node_text(3.0) == ("3", False)


# ---------------------------------------------------------------- options
def test_options_keywords_units_and_flags():
    o = Options(draw=True, fill="Blue!20", rounded_corners=3, minimum_width=2, line_width=1.5, opacity=0.5)
    assert o.inner() == ("draw, fill=Blue!20, rounded corners=3pt, minimum width=2cm, line width=1.5pt, "
                         "opacity=0.5")


def test_options_raw_strings_merge_and_override():
    o = Options("thick, fill=red", fill="blue", dashed=True)
    assert o.inner() == "thick, fill=blue, dashed"
    o.set("thick", False)
    assert o.inner() == "fill=blue, dashed"


def test_options_special_keys():
    assert Options(arrow=True).inner() == "->"
    assert Options(arrow="<->").inner() == "<->"
    assert Options(style="block=Red, thick").inner() == "block=Red, thick"
    assert Options(in_=90, out=0).inner() == "in=90, out=0"
    assert Options(shorten_end="2pt").inner() == "shorten >=2pt"
    assert Options(text_color="red").inner() == "text=red"
    assert Options(minimum_width=tz.mm(5)).inner() == "minimum width=5mm"
    assert Options({"shorten <": "1pt"}).inner() == "shorten <=1pt"


def test_options_values_get_braces():
    assert Options(label="above:$x,y$").inner() == "label={above:$x,y$}"
    assert Options(shift=(1, 2)).inner() == "shift={(1,2)}"


# ---------------------------------------------------------------- coordinates
def test_point_arithmetic():
    p = Point(1, 2) + (3, 4)
    assert p == Point(4, 6)
    assert (3, 4) + Point(1, 2) == Point(4, 6)
    assert Point(1, 0).rotate(90).round(9) == Point(0, 1)
    assert Point(3, 4).norm() == 5
    assert Point(2, 0).lerp((4, 2), 0.5) == Point(3, 1)
    assert Point(1, 1).tikz() == "(1,1)"
    x, y = Point(5, 6)
    assert (x, y) == (5, 6)


def test_polar():
    p = Polar(90, 2)
    assert p.tikz() == "(90:2)"
    assert p.round(9) == Point(0, 2)
    assert (p + (1, 0)).round(9) == Point(1, 2)


def test_symbolic_coordinates_build_calc():
    a = NodeRef("a")
    assert (a.east + (0.5, 0)).tikz() == "($(a.east)+(0.5,0)$)"
    assert ((1, 0) + a.north).tikz() == "($(a.north)+(1,0)$)"
    assert (a - NodeRef("b")).tikz() == "($(a)-(b)$)"
    assert (0.5 * a).tikz() == "($0.5*(a)$)"
    assert tz.between(a, NodeRef("b"), 0.25).tikz() == "($(a)!0.25!(b)$)"
    assert tz.xy(a.west, "b.north").tikz() == "(a.west |- b.north)"
    assert a.anchor(30).tikz() == "(a.30)"


def test_numeric_helpers_stay_numeric():
    assert tz.midpoint((0, 0), (2, 2)) == Point(1, 1)
    assert tz.towards((0, 0), (10, 0), 3) == Point(3, 0)
    assert tz.project((1, 1), (0, 0), (2, 0)) == Point(1, 0)
    assert tz.xy((1, 2), (3, 4)) == Point(1, 4)


def test_as_coord_variants():
    assert as_coord("a.north").tikz() == "(a.north)"
    assert as_coord("(1,2)").tikz() == "(1,2)"
    assert as_coord(1 + 2j) == Point(1, 2)
    assert as_coord((1, 2, 3)).tikz() == "(1,2,3)"


# ---------------------------------------------------------------- colors
def test_color_definitions_and_mixing():
    c = Color("#3B82F6")
    assert c.expr == "c3B82F6"
    assert c.hex == "#3B82F6"
    assert str(c.lighten(0.8)) == "c3B82F6!20"
    assert str(c.darken(0.3)) == "c3B82F6!70!black"
    assert str(Color("Blue").mix("red", 0.25)) == "Blue!75!red"
    named = Color((10, 20, 30), name="Brand")
    assert named.expr == "Brand"
    assert named.rgb == pytest.approx((10 / 255, 20 / 255, 30 / 255))


def test_color_expressions_resolve_against_active_theme():
    modern = tz.get_theme("modern").palette["Blue"]
    assert Color("Blue").hex == "#" + modern
    blend = Color("Blue!50")
    assert blend.rgb is not None and blend.rgb[0] > Color("Blue").rgb[0]


def test_luminance_and_contrast():
    assert Color("#000000").is_dark()
    assert not Color("#FFFFFF").is_dark()


def test_colormaps():
    cmap = tz.get_colormap("viridis")
    assert cmap(0).hex == "#440154"
    assert cmap(1).hex == "#FDE725"
    assert tz.get_colormap("viridis_r")(0).hex == "#FDE725"
    custom = tz.get_colormap(["white", "black"])
    assert custom(0.5).hex in ("#7F7F7F", "#808080")
    with pytest.raises(ValueError):
        tz.get_colormap("no-such-map")


# ---------------------------------------------------------------- paths
def test_path_tokens_insert_connectors():
    tokens = path_tokens([(0, 0), (1, 0), (1, 1), "cycle"])
    assert " ".join(tokens) == "(0,0) -- (1,0) -- (1,1) -- cycle"


def test_path_labels_between_points():
    tokens = path_tokens([(0, 0), tz.label("$f$", "above"), (1, 0)])
    assert " ".join(tokens) == "(0,0) -- node[above] {$f$} (1,0)"


def test_path_arc_continues_from_current_point():
    tokens = path_tokens([(1, 0), tz.arc(0, 90, 1), (0, 0), "cycle"])
    assert " ".join(tokens) == "(1,0) arc[start angle=0, end angle=90, radius=1cm] -- (0,0) -- cycle"


def test_path_connectors_and_ops():
    assert " ".join(path_tokens([(0, 0), "-|", (1, 1)])) == "(0,0) -| (1,1)"
    assert " ".join(path_tokens([(0, 0), tz.to(bend_left=20), (1, 1)])) == "(0,0) to[bend left=20] (1,1)"
    assert " ".join(path_tokens([(0, 0), tz.rel(1, 0)])) == "(0,0) -- ++(1,0)"
    assert " ".join(path_tokens([(0, 0), tz.curve((1, 1), (2, 1)), (3, 0)])) == \
        "(0,0) .. controls (1,1) and (2,1) .. (3,0)"


def test_long_paths_wrap():
    path = Path([(i, i) for i in range(60)])
    lines = path.render(None)  # type: ignore[arg-type]
    assert len(lines) > 1
    assert all(len(ln) <= 110 for ln in lines)
    assert lines[-1].endswith(";")


# ---------------------------------------------------------------- pictures
def test_unreferenced_nodes_have_no_name_referenced_ones_do():
    pic = tz.Picture()
    pic.node("free", at=(0, 0))
    a = pic.node("A", at=(1, 0))
    b = pic.node("B", at=(2, 0), name="bee")
    pic.arrow(a, b)
    lines = body(pic)
    assert lines[0] == r"\node at (0,0) {free};"
    assert lines[1] == r"\node (n1) at (1,0) {A};"
    assert lines[2] == r"\node (bee) at (2,0) {B};"
    assert lines[3] == r"\draw[->] (n1) -- (bee);"


def test_nodes_referenced_only_lazily_still_get_names():
    pic = tz.Picture()
    target = pic.node("T", at=(0, 0))
    pic.arrow(pic.node("S", at=(2, 0)), target)  # the source is created inline
    lines = body(pic)
    assert lines[1].startswith(r"\node (n2) at (2,0)")
    assert lines[2] == r"\draw[->] (n2) -- (n1);"


def test_relative_placement_and_anchor():
    pic = tz.Picture()
    a = pic.node("A")
    pic.node("B", right_of=a, distance=1.5, anchor="west")
    pic.node("C", below_of=a)
    lines = body(pic)
    assert lines[1] == r"\node[anchor=west, right=1.5cm of n1] {B};"
    assert lines[2] == r"\node[below=of n1] {C};"


def test_multiline_nodes_get_align():
    pic = tz.Picture()
    pic.node("two\nlines")
    assert body(pic)[0] == r"\node[align=center] {two\\lines};"


def test_only_used_colors_and_styles_are_emitted():
    pic = tz.Picture()
    pic.node("x", style="block=Orange")
    code = pic.tikz()
    assert r"\definecolor{Orange}" in code
    assert r"\definecolor{Purple}" not in code
    assert "block/.style=" in code and "dot/.style" not in code
    assert r"\definecolor{Ink}" in code  # used by the picture options


def test_role_names_become_color_aliases():
    pic = tz.Picture()
    pic.node("x", style="block=input")
    code = pic.tikz()
    assert r"\colorlet{input}{Green}" in code
    assert r"\definecolor{Green}" in code


def test_custom_colors_and_styles():
    pic = tz.Picture()
    brand = tz.Color("#123456", name="TestBrand")
    pic.style("tag", "draw, rounded corners", fill=brand)
    pic.node("x", style="tag")
    code = pic.tikz()
    assert r"\definecolor{TestBrand}{HTML}{123456}" in code
    assert "tag/.style={draw, rounded corners, fill=TestBrand}" in code  # user style overrides the theme's


def test_style_dependencies_are_emitted():
    pic = tz.Picture()
    pic.style("outer", "inner2, thick")
    pic.style("inner2", fill="red")
    pic.node("x", style="outer")
    code = pic.tikz()
    assert "outer/.style" in code and "inner2/.style" in code


def test_themes_change_definitions():
    modern = tz.Picture()
    dark = tz.Picture(theme="dark")
    for pic in (modern, dark):
        pic.node("x", style="block")
    assert "3A76D8" in modern.tikz()
    assert "60A5FA" in dark.tikz() and r"\begin{pgfonlayer}{pytikzpage}" in dark.tikz()


def test_document_structure():
    pic = tz.Picture(border=3)
    pic.node("x", right_of="y")
    doc = pic.document()
    assert doc.startswith(r"\documentclass[tikz, border=3pt]{standalone}")
    assert r"\usetikzlibrary{arrows.meta, positioning}" in doc
    assert doc.rstrip().endswith(r"\end{document}")


def test_library_detection():
    libs = detect_libraries(r"\node[fit=(a) (b), trapezium] {}; \draw ($(a)+(1,0)$) -- (b);")
    assert {"fit", "shapes.geometric", "calc"} <= set(libs)
    assert "decorations.pathreplacing" in detect_libraries("decoration={brace}")
    assert detect_libraries(r"\draw (0,0) -- (1,1);") == []


def test_scope_and_raw():
    pic = tz.Picture()
    with pic.scope(xshift=1, rotate=30) as s:
        s.circle((0, 0), 1)
    pic.raw(r"\draw (0,0) -- (1,1);")
    pic.comment("done")
    lines = body(pic)
    assert lines[0] == r"\begin{scope}[xshift=1cm, rotate=30]"
    assert lines[1] == r"\draw (0,0) circle[radius=1cm];"
    assert lines[2] == r"\end{scope}"
    assert lines[3] == r"\draw (0,0) -- (1,1);"
    assert lines[4] == "% done"


def test_container_helpers():
    pic = tz.Picture()
    nodes = pic.chain(["A", ("B", {"style": "block=Red"}), "C"], direction="down")
    assert len(nodes) == 3
    pic.bypass(nodes[0].east, nodes[2].east, offset=1)
    pic.fit(*nodes, title="group")
    pic.brace((0, 0), (2, 0), "brace")
    pic.dot((1, 1), "$p$", position=45)
    pic.angle_mark((1, 0), (0, 0), (0, 1), r"$\theta$")
    pic.right_angle_mark((1, 0), (0, 0), (0, 1))
    pic.tick_marks((0, 0), (2, 0), 2)
    code = pic.tikz()
    assert "below=0.8cm of n1" in code
    assert "-- ++(1,0) |-" in code
    assert "fit=(n1) (n2) (n3)" in code
    assert "decoration={brace" in code
    assert "label={45:{$p$}}" in code


def test_angle_mark_draws_interior_angle():
    pic = tz.Picture()
    pic.angle_mark((0, 1), (0, 0), (1, 0), radius=1)  # given clockwise: still the 90 degree angle
    assert "start angle=0, end angle=90" in pic.tikz()


def test_geometry_helpers():
    from pytikz import geometry as g

    o, r = g.circumcircle((0, 0), (2, 0), (0, 2))
    assert o == Point(1, 1) and r == pytest.approx(math.sqrt(2))
    i, ri = g.incircle((0, 0), (3, 0), (0, 4))
    assert ri == pytest.approx(1.0) and i.round(9) == Point(1, 1)
    assert g.line_intersection((0, 0), (1, 1), (0, 1), (1, 0)) == Point(0.5, 0.5)
    assert g.polygon_area([(0, 0), (1, 0), (1, 1), (0, 1)]) == pytest.approx(1.0)
    assert len(g.circle_circle_intersections((0, 0), 1, (1, 0), 1)) == 2
    pieces = g.clip_polyline([-1, 0.5, 2], [0.5, 0.5, 0.5], (0, 0, 1, 1))
    assert pieces == [[(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)]]
    assert len(g.clip_polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)], (0, 0, 1, 1))) == 4


def test_circle_and_ellipse_path_ops():
    tokens = path_tokens([(0, 0), tz.circle(2), (3, 0), tz.ellipse(1, 0.5)])
    assert " ".join(tokens) == "(0,0) circle[radius=2cm] (3,0) ellipse[x radius=1cm, y radius=0.5cm]"


def test_picture_repr_and_text_contrast():
    pic = tz.Picture(theme="dark")
    pic.node("x")
    assert repr(pic) == "<Picture: 1 elements, theme='dark'>"
    dark, modern = tz.get_theme("dark"), tz.get_theme("modern")
    assert modern.text_on(True) == "white" and modern.text_on(False) == "Ink"
    assert dark.text_on(True) == "Ink" and dark.text_on(False) == "Paper"
