"""A midpoint Riemann sum approximating the area under a curve."""

import numpy as np

import pytikz as tz


def f(x):
    return 1.2 + np.sin(1.1 * x) + 0.18 * x


a, b, n = 0.5, 7.5, 10
dx = (b - a) / n
mids = a + dx * (np.arange(n) + 0.5)

pic = tz.Picture(theme="paper")
ax = pic.add(tz.Axes(width=9, height=5.6, xlim=(0, 8.2), ylim=(0, 4.4), frame="cross",
                     xticks=[(a, "$a$"), (b, "$b$")], yticks=None, xlabel="$x$", ylabel="$y$"))
colors = tz.get_colormap(["Blue!35", "Purple!45"]).sample(n)
for i, (m, color) in enumerate(zip(mids, colors)):
    x0, x1 = a + i * dx, a + (i + 1) * dx
    ax.polygon([(x0, 0), (x1, 0), (x1, f(m)), (x0, f(m))], fill=color, draw="Blue!60!black", line_width=0.5,
               fill_opacity=0.75)
    ax.scatter([m], [f(m)], color="Ink", size=1.3, edgecolor=None)
ax.function(f, (0.2, 7.9), color="Ink", line_width=1.6)

# width of one rectangle
i = 6
x0 = a + i * dx
pic.brace(ax.point(x0 + dx, -0.05), ax.point(x0, -0.05), r"$\Delta x$")
ax.text(7.95, f(7.95) + 0.12, "$y = f(x)$", anchor="south east")

pic.node(r"$\displaystyle\int_a^b f(x)\,dx \;\approx\; \sum_{i=1}^{n} f(x_i^*)\,\Delta x$",
         at=ax.point(4.1, 4.35), anchor="north", inner_sep=4)

if __name__ == "__main__":
    pic.render("riemann_sum.pdf")
