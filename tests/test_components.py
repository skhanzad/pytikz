"""Tests of the high-level components, layouts, ticks and contours."""

import math

import numpy as np
import pytest

import pytikz as tz
from pytikz.contour import contour_lines, superlevel_loops
from pytikz.coords import Point
from pytikz.layout import circular_layout, layered_layout, spring_layout, tidy_tree_layout
from pytikz.ticks import format_number, linear_ticks, nice_step, pi_ticks


def code_of(*elements, **picture_kwargs):
    pic = tz.Picture(**picture_kwargs)
    for e in elements:
        pic.add(e)
    return pic.tikz()


# ---------------------------------------------------------------- ticks
def test_nice_steps_and_ticks():
    assert nice_step(10, 5) == 2
    assert nice_step(1, 4) == 0.25
    assert linear_ticks(-0.3, 1.2, step=0.5) == [0.0, 0.5, 1.0]
    assert linear_ticks(0, 1, step=0.1)[3] == 0.3  # no 0.30000000000000004


def test_tick_labels():
    assert format_number(-1.5, 1) == "$-$1.5"
    assert format_number(12500, 0) == r"12\,500"
    labels = dict(pi_ticks(-math.pi, math.pi))
    assert labels[math.pi / 2] == r"$\frac{\pi}{2}$"
    assert labels[-math.pi] == r"$-\pi$"
    assert labels[0] == "0"


# ---------------------------------------------------------------- contours
def test_contour_of_a_circle():
    x = y = np.linspace(-2, 2, 81)
    X, Y = np.meshgrid(x, y)
    lines = contour_lines(x, y, X**2 + Y**2, 1.0)
    assert len(lines) == 1 and lines[0][0] == lines[0][-1]
    radii = [math.hypot(*p) for p in lines[0]]
    assert max(abs(r - 1) for r in radii) < 0.01


def test_superlevel_loops_are_closed():
    x = y = np.linspace(-2, 2, 41)
    X, Y = np.meshgrid(x, y)
    loops = superlevel_loops(x, y, X**2 + Y**2, 5.0)  # the four corners
    assert len(loops) == 4 and all(loop[0] == loop[-1] for loop in loops)


# ---------------------------------------------------------------- layouts
def test_circular_layout():
    pos = circular_layout(list("abcd"), radius=2)
    assert pos["a"].round(9) == Point(0, 2)
    assert all(abs(p.norm() - 2) < 1e-9 for p in pos.values())


def test_spring_layout_is_deterministic_and_bounded():
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    a = spring_layout(range(4), edges, seed=3)
    b = spring_layout(range(4), edges, seed=3)
    assert a == b
    assert max(max(abs(p.x), abs(p.y)) for p in a.values()) == pytest.approx(1.0)


def test_layered_layout_puts_sources_above_targets():
    pos = layered_layout(list("abcd"), [("a", "b"), ("b", "c"), ("d", "c")])
    assert pos["a"].y > pos["b"].y > pos["c"].y
    assert pos["d"].y == pos["b"].y  # sources sit just above their consumer


def test_tidy_tree_layout_separates_subtrees():
    tree = tz.TreeNode("r", [tz.TreeNode("a", ["a1", "a2", "a3"]), tz.TreeNode("b", ["b1", "b2"])])
    pos = tidy_tree_layout(tree, lambda n: n.children, width=lambda n: 1.0, gap=0.5)
    leaves = [pos[id(n)] for n in tree.walk() if not n.children]
    xs = sorted(p.x for p in leaves)
    assert all(b - a >= 1.5 - 1e-9 for a, b in zip(xs, xs[1:]))
    root, a, b = pos[id(tree)], pos[id(tree.children[0])], pos[id(tree.children[1])]
    assert root.x == pytest.approx((a.x + b.x) / 2)


# ---------------------------------------------------------------- components
def test_axes_render_ticks_labels_and_clipping():
    ax = tz.Axes(width=6, height=4, xlim=(-1, 1), ylim=(0, 2), xlabel="$x$", ylabel="$y$", title="T", grid=True)
    ax.function(lambda x: 1 / x, label="hyperbola")  # pole at 0: the curve must break and stay clipped
    ax.scatter([0, 0.5], [1, 1.5], label="pts")
    ax.legend()
    code = code_of(ax)
    assert r"\clip (0,0) rectangle (6,4);" in code
    assert "ax1-xt0" in code and "fit=(ax1-xt0)" in code
    assert r"\begin{tabular}" in code
    for line in code.splitlines():
        if line.strip().startswith(r"\draw[color=Blue"):
            coords = [float(v) for v in __import__("re").findall(r"\((-?[\d.]+),", line)]
            assert all(-0.01 <= c <= 6.01 for c in coords)


def test_axes_autoscale_and_point():
    ax = tz.Axes(width=5, height=5, margins=0)
    ax.plot([0, 10], [0, 20])
    code_of(ax)
    assert ax.xlim == (0, 10) and ax.ylim == (0, 20)
    assert ax.point(5, 10).tikz() == "(2.5,2.5)"


def test_axes_log_scale():
    ax = tz.Axes(width=4, height=4, xscale="log", xlim=(1, 1000), ylim=(0, 1))
    ax.plot([1, 10, 100, 1000], [0, 0.3, 0.6, 1])
    code = code_of(ax)
    assert "$10^{2}$" in code


def test_axes_fields_and_contours_render():
    ax = tz.Axes(xlim=(-2, 2), ylim=(-2, 2))
    ax.contourf(lambda x, y: x**2 + y**2, levels=5)
    ax.contour(lambda x, y: x * y, levels=[-1, 1])
    ax.vector_field(lambda x, y: (-y, x), n=6)
    ax.streamlines(lambda x, y: (-y, x), density=0.5)
    ax.regions(lambda x, y: (x > 0).astype(int))
    ax.imshow(np.eye(3), extent=(-2, 2, -2, 2))
    ax.colorbar()
    code = code_of(ax)
    assert "even odd rule" in code and "markings" in code and "Stealth" in code


def test_neural_network_counts_and_refs():
    net = tz.NeuralNetwork([3, 10, 2], max_neurons=5, labels=["$x_{i}$", None, None], titles=True, name="nn")
    code = code_of(net)
    assert code.count(r"\node[neuron=") == 3 + 4 + 2  # 10 neurons shown as 3 + dots + last
    assert r"$\vdots$" in code
    assert "$x_{1}$" in code and "$x_{3}$" in code
    assert net.neuron(1, 9).tikz() == "(nn-1-9)"
    with pytest.raises(IndexError):
        net.neuron(1, 5)


def test_two_networks_do_not_share_conflicting_styles():
    a = tz.NeuralNetwork([2, 2], name="a")
    b = tz.NeuralNetwork([2, 2], name="b", arrows=True, at=(5, 0))
    code = code_of(a, b)
    assert "connection/.style" in code and "b connection/.style" in code


def test_convnet_and_cuboid():
    cnn = tz.ConvNet()
    cnn.input(32, 3).conv(16, 32, bands=2, label="conv").pool(16).fc(10, label="fc")
    cnn.skip(1, 3)
    code = code_of(cnn, tz.Cuboid((0, -4), 1, 1, 1, grid=(2, 2, 2), label="t"))
    assert code.count(r"\filldraw") >= 3 * 5
    assert cnn.block(1).extent[2] > cnn.block(1).extent[0]


def test_heatmap_cellgrid_matrix():
    hm = tz.Heatmap([[0, 1], [1, 0]], row_labels=["a", "b"], col_labels=["c", "d"], values=True, colorbar=True)
    grid = tz.CellGrid(values=[[1, 2], [3, 4]], at=(5, 0))
    grid.outline(0, 0, 1, 1).color_cell(0, 1, "Red!20")
    m = tz.Matrix([[1, 2], [3, 4]], at=(9, 0), name="M")
    m.highlight_row(0)
    code = code_of(hm, grid, m)
    assert "matrix of math nodes" in code and "left delimiter={[}" in code
    assert m.entry(1, 1).tikz() == "(M-2-2)"
    assert grid.cell(0, 0) == Point(4.7, 0.3)


def test_graph_layouts_and_highlighting():
    for layout in ("spring", "circular", "layered", "grid"):
        g = tz.Graph(edges=[(1, 2), (2, 3), (3, 1)], layout=layout, directed=True)
        assert "->" in code_of(g)
    g = tz.Graph(edges=[("a", "b", 3), ("b", "c", {"label": 4, "dashed": True}), ("c", "c")])
    g.highlight_path(["a", "b"])
    code = code_of(g)
    assert "looseness" in code and "dashed" in code and "{3}" in code


def test_automaton():
    dfa = tz.Automaton(["p", "q"], [("p", "q", "a"), ("p", "q", "b"), ("q", "q", "a")], initial="p",
                       accepting=["q"])
    code = code_of(dfa)
    assert "initial" in code and "accepting" in code
    assert "{$a, b$}" in code  # parallel transitions are merged


def test_tree_from_tuples():
    t = tz.Tree(("root", ("a", "a1", "a2"), "b"), grow="right")
    code = code_of(t)
    assert code.count(r"\draw[tree edge]") == 4


def test_commutative_diagram():
    cd = tz.CommutativeDiagram([["A", "B"], ["C", "D"]])
    cd.arrow("A", "B", "f").arrow((0, 0), (1, 0), "g", kind="hook", swap=True).arrow("B", "D", kind="epi")
    code = code_of(cd)
    assert "{Hooks[right,length=2.2pt]}->" in code and "->>" in code
    with pytest.raises(ValueError):
        cd.arrow("A", "D", kind="nope")


def test_axes3d_depth_sorting():
    ax = tz.Axes3D(xlim=(-1, 1), ylim=(-1, 1))
    ax.surface(lambda x, y: x * y, resolution=6)
    ax.scatter3d([0], [0], [0])
    ax.plot3d([0, 1], [0, 1], [0, 1])
    ax.contour_floor(lambda x, y: x * y, levels=3, resolution=20)
    code = code_of(ax)
    assert code.count(r"\filldraw[fill=") >= 25


def test_streamlines_use_autoscaled_limits():
    ax = tz.Axes()
    ax.plot([-1, 1], [-1, 1])
    ax.streamlines(lambda x, y: (-y, x), density=0.4)
    assert "markings" in code_of(ax)
