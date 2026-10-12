"""Common activation functions (solid) and their derivatives (dashed)."""

import math

import numpy as np

import pytikz as tz


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


# name, f, f', y-limits, formula, where to write the formula
panels = [
    ("Sigmoid", sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x)), (-0.25, 1.15),
     r"$\sigma(x)=\frac{1}{1+e^{-x}}$", (-3.0, 1.0)),
    ("Tanh", np.tanh, lambda x: 1 - np.tanh(x) ** 2, (-1.2, 1.2), r"$\tanh x$", (-3.0, 0.85)),
    ("ReLU", lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float), (-0.4, 3.2),
     r"$\max(0, x)$", (-3.0, 2.8)),
    ("GELU", gelu, lambda x: np.gradient(gelu(x), x), (-0.4, 3.2), r"$x\,\Phi(x)$", (-3.0, 2.8)),
]

pic = tz.Picture()
for k, (name, f, df, ylim, formula, where) in enumerate(panels):
    ax = pic.add(tz.Axes(at=(k * 5.2, 0), width=4.2, height=3.2, xlim=(-3.2, 3.2), ylim=ylim,
                         frame="open", grid=True, title=name, xticks=[-3, 0, 3]))
    xs = np.linspace(-3.2, 3.2, 400)
    ax.plot(xs, df(xs), color="Orange", line_width=1.2, dash="--")
    ax.plot(xs, f(xs), color="Blue", line_width=1.6)
    ax.text(*where, formula, anchor="north west", font=r"\footnotesize", fill="white", fill_opacity=0.85,
            inner_sep=1.5)

# one legend for all panels
pic.draw((7.0, -1.25), (7.7, -1.25), color="Blue", line_width=1.6)
pic.node("activation", at=(7.8, -1.25), anchor="west", font=r"\small")
pic.draw((10.0, -1.25), (10.7, -1.25), color="Orange", line_width=1.2, dashed=True)
pic.node("derivative", at=(10.8, -1.25), anchor="west", font=r"\small")

if __name__ == "__main__":
    pic.render("activations.pdf")
