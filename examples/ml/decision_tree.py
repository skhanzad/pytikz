"""A decision tree next to the partition of the feature space it induces."""

import numpy as np

import pytikz as tz

T = tz.TreeNode
COLORS = {"A": "Blue", "B": "Orange", "C": "Green"}


def leaf(cls, edge):
    return T(f"class {cls}", color=COLORS[cls], edge_label=edge)


tree = T(r"$x_1 \le 0.5$", [
    T(r"$x_2 \le 0.65$", [leaf("A", "yes"), leaf("B", "no")], edge_label="yes", color="Gray"),
    T(r"$x_2 \le 0.3$", [leaf("B", "yes"), leaf("C", "no")], edge_label="no", color="Gray"),
], color="Gray")

pic = tz.Picture()
pic.add(tz.Tree(tree, at=(0, 2.2), level_distance=1.6, gap=0.5))


def predict(x1, x2):
    return np.where(x1 <= 0.5, np.where(x2 <= 0.65, 0, 1), np.where(x2 <= 0.3, 1, 2))


ax = pic.add(tz.Axes(at=(5.2, -3.0), width=5.2, height=5.2, xlim=(0, 1), ylim=(0, 1), frame="box",
                     xlabel="$x_1$", ylabel="$x_2$", xticks=[0, 0.5, 1], yticks=[0, 0.3, 0.65, 1]))
ax.regions(predict, colors=["Blue!12", "Orange!14", "Green!14"])
ax.vline(0.5, color="Ink!70", dash=None, line_width=1.0)
ax.hline(0.65, xmax=0.5, color="Ink!70", dash=None, line_width=1.0)
ax.hline(0.3, xmin=0.5, color="Ink!70", dash=None, line_width=1.0)

rng = np.random.default_rng(2)
X = rng.random((90, 2))
y = predict(X[:, 0], X[:, 1])
flip = rng.random(90) < 0.06          # a few noisy labels
y = np.where(flip, (y + 1) % 3, y)
for k, (cls, color) in enumerate(COLORS.items()):
    sel = y == k
    ax.scatter(X[sel, 0], X[sel, 1], color=color, size=2.1, marker="os^"[k], label=f"class {cls}")
for (x, yv), cls in (((0.25, 0.33), "A"), ((0.25, 0.85), "B"), ((0.75, 0.15), "B"), ((0.75, 0.68), "C")):
    ax.text(x, yv, f"\\textbf{{{cls}}}", text_color=f"{COLORS[cls]}!70!black", font=r"\large")

if __name__ == "__main__":
    pic.render("decision_tree.pdf")
