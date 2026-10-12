# pytikz

**Write Python, get beautiful TikZ.** pytikz turns short Python scripts into clean, readable
TikZ code (and PDF/PNG/SVG) for machine-learning and mathematical diagrams: network
architectures, plots, 3D surfaces, graphs, trees, commutative diagrams and geometry.

<p align="center">
  <img src="examples/output/transformer.png" height="330">
  <img src="examples/output/loss_landscape.png" height="250">
  <img src="examples/output/attention.png" height="250">
</p>

* **Everything computed in Python, drawn with plain TikZ.** Layouts, plot transforms,
  contours, 3D projection and depth sorting all happen in Python, so the output needs
  no pgfplots and compiles anywhere.
* **Output you can read and edit.** Named styles (`block=Blue`, `neuron=Green`), palette
  colors and node names instead of walls of options; only the colors, styles and TikZ
  libraries a picture actually uses are emitted.
* **Batteries for ML and math.** `Axes`, `Axes3D`, `NeuralNetwork`, `ConvNet`, `Heatmap`,
  `CellGrid`, `Matrix`, `Graph`, `Automaton`, `Tree`, `CommutativeDiagram`, plus
  geometry helpers.
* **Themes.** `modern` (default), `paper`, `dark` and `mono` restyle a whole figure.
* **A real workflow.** A CLI (`pytikz render`, `pytikz tex`, `pytikz watch`), readable
  LaTeX error messages pointing at the offending line, a compile cache and Jupyter display.

## Installation

```bash
pip install -e .            # from this repository (needs Python >= 3.9 and numpy)
```

Compiling needs a TeX distribution with TikZ (TeX Live, MiKTeX or MacTeX) providing
`pdflatex`; PNG/SVG output uses poppler's `pdftocairo` (or `pdftoppm`/ImageMagick). Without
LaTeX you can still generate `.tex`/`.tikz` files and compile them elsewhere (e.g. Overleaf).

## Quick start

```python
import pytikz as tz

pic = tz.Picture()
data = pic.node("Data", style="block=Green")
model = pic.node("Model", right_of=data, distance=1.2, style="block")
pred = pic.node("Prediction", right_of=model, distance=1.2, style="block=Orange")
pic.arrow(data, model)
pic.arrow(model, pred)
pic.arrow(pred.south, model.south, bend=40, label="loss", dashed=True, color="Red")

pic.render("hello.pdf")     # also .png / .svg
pic.save("hello.tex")       # standalone LaTeX document
pic.save("hello.tikz")      # just the picture, to \input in a paper
```

<img src="examples/output/hello.png" height="110">

The generated TikZ is what you would write by hand:

```latex
\tikzset{%
  block/.style={draw=#1!78!black, fill=#1!14, rounded corners=3pt, align=center, inner sep=5pt,
                minimum height=8mm, minimum width=20mm},
  block/.default=Blue,
}%
\begin{tikzpicture}[color=Ink, execute at begin picture={\sffamily}, line width=0.6pt, >={Stealth[...]}, line join=round]
  \node[block=Green] (n1) {Data};
  \node[block, right=1.2cm of n1] (n2) {Model};
  \node[block=Orange, right=1.2cm of n2] (n3) {Prediction};
  \draw[->] (n1) -- (n2);
  \draw[->] (n2) -- (n3);
  \draw[->, dashed, color=Red] (n3.south) to[bend left=40] node[pos=0.5, auto, font=\small] {loss} (n2.south);
\end{tikzpicture}
```

### From the command line

```bash
pytikz render figure.py -f pdf,png          # compile every Picture the script creates
pytikz render examples/ml/*.py -o out/ -f png --dpi 200
pytikz tex figure.py                        # print the TikZ code
pytikz watch figure.py                      # re-render whenever the file is saved
```

The CLI runs the script with `__name__ == "__pytikz__"` (so `if __name__ == "__main__":`
blocks are skipped) and writes one file per picture, named after the script (or
`script-<variable>` when it creates several).

### In a LaTeX paper

`pic.save("fig.tikz")` writes the color and style definitions followed by the
`tikzpicture`. The header comment lists what the preamble needs:

```latex
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}   % as listed in fig.tikz
...
\begin{figure}
  \centering
  \input{fig.tikz}
\end{figure}
```

In Jupyter, a `Picture` displays itself (rendered to SVG).

## Core concepts

### Nodes, paths and arrows

Every drawing method returns the element it created, so you can keep references:

```python
a = pic.node("$A$", at=(0, 0), style="block")
b = pic.node("$B$", right_of=a, distance=1.5, style="block=Orange")   # positioning library
c = pic.node("$C$", below_of=b, style="block=Purple")

pic.arrow(a, b, label="$f$")                          # label on the left of the direction of travel
pic.arrow(a, c, route="|-", label=r"$g\circ f$", label_side="swap")
pic.arrow(c, a, bend=30, dashed=True)                 # positive bends turn left

pic.draw((0, 0), (1, 0), (1, 1), "cycle", fill="Yellow!25")   # consecutive points are joined by --
pic.draw((2, 0), tz.curve((2.5, 1), (3.5, -1)), (4, 0))      # Bezier segment
pic.draw((5, 0), tz.arc(0, 270, 0.6), "--", tz.rel(0.6, 0))  # arc, then a relative move
pic.draw(a, tz.label("$x$", "above"), b)                     # inline label on a segment
with pic.scope() as s:                                       # clip to a circle
    s.clip((0, 0), tz.circle(1.5))
    s.rectangle((-2, -2), (0, 0), fill="Blue!20")
```

Shapes and helpers: `line`, `polygon`, `rectangle`, `circle`, `ellipse`, `arc`, `sector`,
`grid`, `plot` (smooth curves), `function`, `parametric`, `dot` (labelled points),
`brace`, `fit` (box around nodes), `chain` (nodes in a row with arrows), `bypass`
(residual/skip connections), `angle_mark`, `right_angle_mark`, `tick_marks`, `scope`,
`background` and `raw` for verbatim TikZ.

### Options

TikZ options are keyword arguments: underscores become spaces, `True` gives a bare key,
`None`/`False` removes it, and plain numbers get a sensible unit (cm for sizes and
shifts, pt for line widths, separations and corner radii):

```python
pic.node("x", minimum_width=2, inner_sep=4, rounded_corners=3, fill="Blue!15", draw=True)
# \node[minimum width=2cm, inner sep=4pt, rounded corners=3pt, fill=Blue!15, draw] {x};
pic.draw(a, b, style="thick, dashed", arrow="<->", shorten_end=2, in_=90, out=0)
```

Raw TikZ works anywhere (`style="block=Red, thick"`), as do explicit units (`tz.mm(5)`,
`tz.pt(2)`). Python keywords are written with a trailing underscore (`in_`).
Node text is LaTeX; `\n` makes a line break, and `&`, `%`, `#` and `_` are escaped
outside math mode, so `"Add & Norm"` just works.

### Coordinates

`tz.Point` is a numeric 2D vector (`+`, `-`, `*`, `rotate`, `norm`, `lerp`, `angle`, ...),
so geometry and layouts can be computed in Python. Nodes and their anchors (`node.north`,
`node.east`, `node.anchor(30)`) are only known to TikZ; arithmetic on them becomes a
`calc` expression:

```python
a = pic.node("A", name="a")  # nodes get automatic names (n1, n2, ...) unless you choose one
b = pic.node("B", name="b", right_of=a)
a.east + (0.5, 0)            # ($(a.east)+(0.5,0)$)
tz.midpoint(a, b)            # ($(a)!0.5!(b)$)
tz.between(a, b, 0.25)       # ($(a)!0.25!(b)$): a quarter of the way
tz.xy(a.west, b.north)       # (a.west |- b.north): x from one, y from the other
tz.midpoint((0, 0), (2, 2))  # Point(1, 1): numeric inputs stay numeric
```

All of these can be passed wherever a point is expected, or formatted into raw TikZ
(`pic.raw(f"\\draw {a.east} -- {b.west};")`).

### Colors, styles and themes

Every theme has a palette (`Blue`, `Sky`, `Teal`, `Green`, `Lime`, `Yellow`, `Orange`,
`Red`, `Pink`, `Purple`, `Indigo`, `Brown`, `Gray`, `Ink`) usable by name, with xcolor
mixing: `"Blue!15"`, `"Red!60!black"`. Semantic role names (`input`, `hidden`, `output`,
`conv`, `pool`, `fc`, `attention`, `norm`, `embedding`, `primary`, `accent`, ...) are
aliases that follow the theme. Custom colors and styles:

```python
brand = tz.Color("#0B7285", name="Brand")         # \definecolor emitted when used
pic.node("x", fill=brand.lighten(0.8), draw=brand)
pic.style("pill", "draw, rounded corners=7pt", fill="Purple!12")
pic.node("y", style="pill")
```

Predefined styles: `block=<color>`, `box`, `dot`, `tag=<color>`, `group=<color>`, `op`
(circled operator such as $\oplus$), `muted`. Themes: `tz.Picture(theme="paper")` (serif,
for print), `"dark"` (slides), `"mono"`.

<img src="examples/output/themes_modern.png" height="160"> <img src="examples/output/themes_paper.png" height="160"> <img src="examples/output/themes_dark.png" height="160"> <img src="examples/output/themes_mono.png" height="160">

## Components

Components are added with `pic.add(...)` and can be mixed freely with nodes and paths.

### `Axes`: 2D plots

```python
ax = pic.add(tz.Axes(width=8, height=5, xlim=(-4, 4), xlabel="$x$", ylabel="$y$", grid=True))
ax.function(np.tanh, label=r"$\tanh x$")
ax.plot(xs, ys, dash="--", marker="o", label="data")
ax.scatter(x, y, c=values, cmap="viridis")
ax.fill_between(xs, lo, hi, opacity=0.2)
ax.contourf(lambda x, y: x**2 + 3 * y**2, levels=12)
ax.streamlines(lambda x, y: (y, -x), density=1.2)
ax.legend("upper left")
pic.arrow(ax.point(1, 2), ax.point(2, 0))   # data coordinates as picture coordinates
```

Frames: `"open"` (default), `"box"`, `"cross"` (textbook axes through the origin) and
`"none"`. Ticks: automatic, explicit values, `(value, label)` pairs, `"pi"` (multiples of
π) or `None`; log scales with `xscale="log"`. Also `bar`, `area`, `hline`, `vline`,
`axline`, `text`, `annotate`, `arrow`, `polygon`, `imshow`, `contour`, `regions`
(decision regions of a classifier), `quiver`, `vector_field` and `colorbar`. Curves are
clipped and simplified in Python; with y limits set, sampled functions break at poles
such as `tan x` instead of drawing vertical lines.

### `Axes3D`: surfaces and curves

```python
ax = pic.add(tz.Axes3D(xlim=(-2, 2), ylim=(-2, 2), azimuth=-60, elevation=30))
ax.surface(lambda x, y: np.exp(-x**2 - y**2), cmap="viridis")   # or X=, Y=, Z= meshes
ax.contour_floor(f, levels=10)
ax.plot3d(xs, ys, zs, color="Red", marker=True)
```

Orthographic camera, Lambert shading, painter's-algorithm depth sorting and matplotlib-like
back panes. Also `wireframe`, `scatter3d`, `arrow3d`, `text3d`.

### Machine learning

```python
net = pic.add(tz.NeuralNetwork([3, 5, 5, 2], labels=["$x_{i}$", None, None, r"$\hat y_{i}$"],
                               titles=True, arrows=True, max_neurons=6))
net.highlight_connection(0, 0, 1, color="Orange", label="$w_{21}$")
pic.node("loss", right_of=net.neuron(-1, 0), distance=1)  # neurons can be referenced

cnn = pic.add(tz.ConvNet())                              # 3D feature maps
cnn.input(224, 3).conv(64, 224, bands=2, label="conv1").pool(112).fc(4096, label="fc").fc(10)

pic.add(tz.Heatmap(weights, row_labels=tokens, col_labels=tokens, values=True, colorbar=True))
grid = pic.add(tz.CellGrid(values=image, cell=0.6))     # convolution / matrix illustrations
grid.outline(1, 1, 3, 3, color="Blue")
pic.add(tz.Cuboid((0, 0), 2, 1.5, 2, grid=(4, 3, 4)))    # a tensor
```

Block diagrams (Transformer, ResNet, VAE, GAN, ...) are built from `block` nodes,
`pic.chain`, `pic.bypass`, `pic.fit` and arrows; see the examples.

### Graphs, trees and diagrams

```python
g = pic.add(tz.Graph(edges=[("s", "a", 4), ("a", "t", 2)], layout="spring", directed=True))
g.highlight_path(["s", "a", "t"])
pic.add(tz.Automaton(["q_0", "q_1"], [("q_0", "q_1", "a"), ("q_1", "q_1", "b")],
                     initial="q_0", accepting=["q_1"]))
pic.add(tz.Tree(("root", ("a", "a1", "a2"), "b")))      # tidy layout, any direction
cd = pic.add(tz.CommutativeDiagram([["A", "B"], ["C", "D"]]))
cd.arrow("A", "B", "f").arrow("A", "C", "g", swap=True).arrow("C", "D", "k", kind="hook")
pic.add(tz.Matrix([[1, 2], [3, 4]], name="M")).highlight_row(0)
```

Graph layouts: `spring`, `circular`, `shell`, `layered` (DAGs), `grid`, `bipartite` or a
dict of positions. Edges that would cross a node are bent automatically and self-loops
are placed in the freest direction.

### Geometry

`pytikz.geometry` computes constructions in Python: `circumcircle`, `incircle`,
`centroid`, `orthocenter`, `foot`, `reflect`, `line_intersection`,
`circle_line_intersections`, `circle_circle_intersections`, `regular_polygon`,
`polygon_area`.

## Gallery

All examples live in [`examples/`](examples); `python examples/build_all.py` renders them
into [`examples/output/`](examples/output) (PDF, PNG and the generated `.tex`).

| | | |
|:-:|:-:|:-:|
| [<img src="examples/output/mlp.png" width="260"><br>MLP](examples/ml/mlp.py) | [<img src="examples/output/cnn.png" width="260"><br>CNN](examples/ml/cnn.py) | [<img src="examples/output/convolution.png" width="260"><br>Convolution](examples/ml/convolution.py) |
| [<img src="examples/output/vae.png" width="260"><br>VAE](examples/ml/vae.py) | [<img src="examples/output/gan.png" width="260"><br>GAN](examples/ml/gan.py) | [<img src="examples/output/rnn.png" width="260"><br>RNN](examples/ml/rnn.py) |
| [<img src="examples/output/resnet_block.png" width="160"><br>ResNet block](examples/ml/resnet_block.py) | [<img src="examples/output/gradient_descent.png" width="260"><br>Optimisers](examples/ml/gradient_descent.py) | [<img src="examples/output/svm.png" width="220"><br>SVM](examples/ml/svm.py) |
| [<img src="examples/output/decision_tree.png" width="260"><br>Decision tree](examples/ml/decision_tree.py) | [<img src="examples/output/kmeans.png" width="220"><br>k-means](examples/ml/kmeans.py) | [<img src="examples/output/confusion_matrix.png" width="220"><br>Confusion matrix](examples/ml/confusion_matrix.py) |
| [<img src="examples/output/backprop.png" width="260"><br>Backpropagation](examples/ml/backprop.py) | [<img src="examples/output/positional_encoding.png" width="260"><br>Positional encoding](examples/ml/positional_encoding.py) | [<img src="examples/output/activations.png" width="260"><br>Activations](examples/ml/activations.py) |
| [<img src="examples/output/overfitting.png" width="260"><br>Over/underfitting](examples/ml/overfitting.py) | [<img src="examples/output/learning_curves.png" width="260"><br>Learning curves](examples/ml/learning_curves.py) | [<img src="examples/output/pca.png" width="220"><br>PCA](examples/ml/pca.py) |
| [<img src="examples/output/unit_circle.png" width="220"><br>Unit circle](examples/math/unit_circle.py) | [<img src="examples/output/riemann_sum.png" width="260"><br>Riemann sum](examples/math/riemann_sum.py) | [<img src="examples/output/normal_distribution.png" width="260"><br>Normal distribution](examples/math/normal_distribution.py) |
| [<img src="examples/output/taylor_series.png" width="260"><br>Taylor series](examples/math/taylor_series.py) | [<img src="examples/output/fourier_series.png" width="260"><br>Fourier series](examples/math/fourier_series.py) | [<img src="examples/output/derivative.png" width="220"><br>Derivative](examples/math/derivative.py) |
| [<img src="examples/output/phase_portrait.png" width="260"><br>Phase portrait](examples/math/phase_portrait.py) | [<img src="examples/output/linear_map.png" width="260"><br>Linear map](examples/math/linear_map.py) | [<img src="examples/output/surfaces.png" width="260"><br>3D surfaces](examples/math/surfaces.py) |
| [<img src="examples/output/triangle_centers.png" width="220"><br>Triangle centres](examples/math/triangle_centers.py) | [<img src="examples/output/pythagoras.png" width="200"><br>Pythagoras](examples/math/pythagoras.py) | [<img src="examples/output/roots_of_unity.png" width="200"><br>Roots of unity](examples/math/roots_of_unity.py) |
| [<img src="examples/output/shortest_path.png" width="260"><br>Shortest path](examples/math/shortest_path.py) | [<img src="examples/output/automaton.png" width="260"><br>Automaton](examples/math/automaton.py) | [<img src="examples/output/commutative_diagrams.png" width="260"><br>Commutative diagrams](examples/math/commutative_diagrams.py) |
| [<img src="examples/output/venn.png" width="200"><br>Venn diagram](examples/math/venn.py) | [<img src="examples/output/matrix_product.png" width="260"><br>Matrix product](examples/math/matrix_product.py) | [<img src="examples/output/tour.png" width="260"><br>API tour](examples/basics/tour.py) |

## Troubleshooting

* **LaTeX errors** are raised as `tz.TikzCompileError` showing the error and the lines
  around it in the generated document; `pic.document()` returns the full source.
* **Missing TikZ library?** Most are detected automatically; add others with
  `pic.use_library("shadows")`. Extra packages: `pic.use_package("siunitx")`,
  `pic.add_preamble(r"\newcommand\R{\mathbb{R}}")`.
* **Fonts.** Standalone renders use Latin Modern, with Helvetica for sans-serif themes. In
  a paper, the document's fonts are used.
* **Cache.** Compiled results are cached in `~/.cache/pytikz` (`PYTIKZ_CACHE` overrides;
  `pic.render(..., cache=False)` bypasses it).

## Development

```bash
pip install -e ".[dev]"
pytest                      # unit tests + compiling every example (needs pdflatex)
pytest -m "not latex"       # without LaTeX
python examples/build_all.py
```

Layout: `pytikz/` holds the core (`picture.py`, `container.py`, `elements.py`,
`coords.py`, `options.py`, `color.py`, `theme.py`, `compiler.py`, `cli.py`) and the
components in `pytikz/components/`. Layout, tick, contour and geometry algorithms live in
`layout.py`, `ticks.py`, `contour.py` and `geometry.py`.
