"""A maximum-margin classifier: the decision boundary, the margins and the
support vectors that define them."""

import numpy as np

import pytikz as tz

rng = np.random.default_rng(7)

# Hyperplane w.x + b = 0 with ||w|| chosen so the margins are w.x + b = +-1.
w = np.array([1.0, 1.25])
b = -4.6


def score(x, y):
    return w[0] * x + w[1] * y + b


def sample(n, sign):
    pts = []
    while len(pts) < n:
        p = rng.normal([2.0, 2.0] if sign < 0 else [3.6, 3.4], 0.85)
        if sign * score(*p) >= 1.25 and 0.2 < p[0] < 5.8 and 0.2 < p[1] < 5.3:
            pts.append(p)
    return np.array(pts)


neg, pos = sample(16, -1), sample(16, +1)
# support vectors exactly on the margins
d = np.array([-w[1], w[0]]) / np.linalg.norm(w)        # direction along the boundary
n = w / np.linalg.norm(w)
on_boundary = -b * w / (w @ w)                          # a point with score 0
sv_neg = [on_boundary - n / np.linalg.norm(w) + d * t for t in (-1.3, 1.1)]
sv_pos = [on_boundary + n / np.linalg.norm(w) + d * 0.2]

pic = tz.Picture()
ax = pic.add(tz.Axes(width=8, height=6.6, xlim=(0, 6), ylim=(0, 5.5), equal=True, frame="box",
                     xlabel="$x_1$", ylabel="$x_2$", xticks=None, yticks=None))
ax.regions(lambda x, y: (score(x, y) > 0).astype(int), colors=["Blue!9", "Orange!11"])

# boundary and margins: w.x + b = 0, +1, -1
slope = -w[0] / w[1]
for level, dash in ((0, None), (1, "dashed"), (-1, "dashed")):
    ax.axline((0, (level - b) / w[1]), slope, color="Ink!80" if level == 0 else "Ink!50",
              line_width=1.3 if level == 0 else 0.8, dash=dash)

ax.scatter(neg[:, 0], neg[:, 1], color="Blue", size=2.6, label="class $-1$")
ax.scatter(pos[:, 0], pos[:, 1], color="Orange", marker="s", size=2.4, label="class $+1$")
svs = np.array(sv_neg + sv_pos)
ax.scatter(np.array(sv_neg)[:, 0], np.array(sv_neg)[:, 1], color="Blue", size=2.6)
ax.scatter(np.array(sv_pos)[:, 0], np.array(sv_pos)[:, 1], color="Orange", marker="s", size=2.4)
for p in svs:
    pic.circle(ax.point(*p), tz.pt(6.5), draw="Ink", line_width=0.8)

# margin width
p0 = on_boundary + d * 0.75 - n / np.linalg.norm(w)
p1 = on_boundary + d * 0.75 + n / np.linalg.norm(w)
pic.draw(ax.point(*p0), ax.point(*p1), style="<->", color="Ink")
pic.node(r"$\frac{2}{\|\mathbf{w}\|}$", at=ax.point(*p1), anchor="south west", font=r"\small")

# normal vector
start = on_boundary - d * 2.3
pic.draw(ax.point(*start), ax.point(*(start + 0.8 * n)), arrow="->", color="Red", line_width=1.2)
pic.node(r"$\mathbf{w}$", at=ax.point(*(start + 0.8 * n)), anchor="south west", text_color="Red")

ax.text(*(on_boundary - d * 1.0), r"$\mathbf{w}^\top\mathbf{x} + b = 0$", rotate=np.degrees(np.arctan(slope)),
        anchor="south", font=r"\footnotesize")
ax.legend("upper left")

if __name__ == "__main__":
    pic.render("svm.pdf")
