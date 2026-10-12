"""The derivative as the limit of secant slopes: secants through P and
P + h approach the tangent as h shrinks."""

import numpy as np

import pytikz as tz


def f(x):
    return 0.4 + 0.5 * np.exp(0.55 * x)


def df(x):
    return 0.275 * np.exp(0.55 * x)


x0 = 1.0
pic = tz.Picture(theme="paper")
ax = pic.add(tz.Axes(width=9, height=6, xlim=(-0.5, 4.6), ylim=(0, 6.2), frame="cross",
                     xticks=[(x0, "$a$")], yticks=None, xlabel="$x$", ylabel="$y$"))
ax.function(f, (-0.5, 4.6), color="Ink", line_width=1.8)

hs = [3.0, 1.9, 1.0, 0.45]
colors = tz.get_colormap(["Orange!60", "Red"]).sample(len(hs))
for h, color in zip(hs, colors):
    slope = (f(x0 + h) - f(x0)) / h
    ax.axline((x0, f(x0)), slope, color=color.expr, line_width=0.8, dash="dashed")
    ax.scatter([x0 + h], [f(x0 + h)], color=color.expr, size=2.2)
ax.axline((x0, f(x0)), df(x0), color="Blue", line_width=1.5, label="tangent")
ax.scatter([x0], [f(x0)], color="Blue", size=3.0)
ax.line([(x0, 0), (x0, f(x0))], color="Ink!40", dashed=True)

h = hs[0]
ax.line([(x0, f(x0)), (x0 + h, f(x0))], color="Ink!60", line_width=0.6)
ax.line([(x0 + h, f(x0)), (x0 + h, f(x0 + h))], color="Ink!60", line_width=0.6)
ax.text(x0 + h / 2, f(x0), "$h$", anchor="north", font=r"\small")
ax.text(x0 + h, (f(x0) + f(x0 + h)) / 2, r"$f(a+h)-f(a)$", anchor="west", font=r"\small")
ax.text(4.25, f(4.25), "$f$", anchor="south east")

pic.node(r"$f'(a) = \lim_{h\to 0}\dfrac{f(a+h)-f(a)}{h}$", at=ax.point(1.2, 5.2), font=r"\small")

if __name__ == "__main__":
    pic.render("derivative.pdf")
