"""Partial Fourier sums of a square wave (and the Gibbs overshoot)."""

import math

import numpy as np

import pytikz as tz


def square_wave(x):
    return np.sign(np.sin(x))


def partial_sum(x, n_terms):
    return 4 / math.pi * sum(np.sin((2 * k + 1) * x) / (2 * k + 1) for k in range(n_terms))


pic = tz.Picture(theme="paper")
ax = pic.add(tz.Axes(width=11, height=5, xlim=(-math.pi, math.pi), ylim=(-1.5, 1.5), frame="cross",
                     xticks="pi", yticks=[-1, 1], xlabel="$x$", ylabel="$y$"))
xs = np.linspace(-math.pi, math.pi, 1201)
ax.plot(xs, square_wave(xs), color="Ink!40", line_width=2.4, label="square wave")
for n, color in ((1, "Blue"), (3, "Green"), (10, "Orange"), (40, "Red")):
    ax.plot(xs, partial_sum(xs, n), color=color, line_width=1.0 if n == 40 else 1.2,
            label=f"$S_{{{2 * n - 1}}}$")
ax.annotate("Gibbs overshoot", xy=(0.04, 1.18), xytext=(0.9, 1.42), anchor="west", font=r"\small")
ax.legend("outside right")
pic.node(r"$S_N(x) = \frac{4}{\pi}\sum_{k=0}^{(N-1)/2} \frac{\sin\big((2k+1)x\big)}{2k+1}$",
         at=ax.point(1.6, -0.75), font=r"\small")

if __name__ == "__main__":
    pic.render("fourier_series.pdf")
