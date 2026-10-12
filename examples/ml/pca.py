"""Principal component analysis of a correlated 2D Gaussian: the principal
axes (scaled by the standard deviations) and the 1- and 2-sigma ellipses."""

import numpy as np

import pytikz as tz

rng = np.random.default_rng(8)
cov = np.array([[2.2, 1.3], [1.3, 1.1]])
X = rng.multivariate_normal([0.5, 0.3], cov, size=220)
mean = X.mean(0)
vals, vecs = np.linalg.eigh(np.cov(X.T))
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]

pic = tz.Picture()
ax = pic.add(tz.Axes(width=7.5, height=6, xlim=(-4.5, 5.5), ylim=(-3.5, 4.0), equal=True, frame="box",
                     xlabel="$x_1$", ylabel="$x_2$", title="Principal components"))
ax.scatter(X[:, 0], X[:, 1], color="Blue!70", size=1.6, edgecolor=None, opacity=0.7)

t = np.linspace(0, 2 * np.pi, 200)
for k, dash in ((1, None), (2, "--")):
    ellipse = mean[:, None] + vecs @ (k * np.sqrt(vals)[:, None] * np.array([np.cos(t), np.sin(t)]))
    ax.plot(ellipse[0], ellipse[1], color="Ink!60", line_width=0.9, dash=dash,
            label=f"${k}\\sigma$" if k == 1 else f"${k}\\sigma$")

for k, color in enumerate(("Red", "Orange")):
    v = vecs[:, k] if vecs[1, k] >= 0 else -vecs[:, k]     # point both axes upwards
    tip = mean + 2 * np.sqrt(vals[k]) * v
    ax.arrow(tuple(mean), tuple(tip), color=color, line_width=1.6)
    # write the label just beyond the tip, on the far side of the arrow
    angle = np.degrees(np.arctan2(v[1], v[0]))
    ax.text(*tip, f"PC{k + 1}", anchor=f"{(angle + 180) % 360:.0f}", text_color=f"{color}!80!black",
            font=r"\small\bfseries", inner_sep=3)
ax.legend("lower right")

if __name__ == "__main__":
    pic.render("pca.pdf")
