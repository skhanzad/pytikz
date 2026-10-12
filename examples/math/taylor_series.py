"""Taylor polynomials of sin(x) around 0 hugging the function over a growing interval."""

import math

import numpy as np

import pytikz as tz


def taylor_sin(x, degree):
    return sum((-1) ** k * x ** (2 * k + 1) / math.factorial(2 * k + 1) for k in range(degree // 2 + 1))


pic = tz.Picture(theme="paper")
ax = pic.add(tz.Axes(width=11, height=5.5, xlim=(-2 * math.pi, 2 * math.pi), ylim=(-2.2, 2.2),
                     frame="cross", xticks="pi", yticks=[-2, -1, 1, 2], xlabel="$x$", ylabel="$y$"))

degrees = [1, 3, 5, 7, 9, 11]
colors = tz.get_colormap("plasma").sample(len(degrees) + 2)[1:-1]
for degree, color in zip(degrees, colors):
    ax.function(lambda x, d=degree: taylor_sin(x, d), samples=400, color=color.expr, line_width=1.1,
                label=f"$T_{{{degree}}}$")
ax.function(np.sin, color="Ink", line_width=2.0, label=r"$\sin x$")
ax.legend("outside right")

if __name__ == "__main__":
    pic.render("taylor_series.pdf")
