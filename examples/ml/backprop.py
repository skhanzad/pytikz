"""Backpropagation on a computation graph: forward values in blue above the
edges, gradients of the loss in red below them.

Computes L = (sigmoid(w*x + b) - y)^2 for one example.
"""

import math

import pytikz as tz

x, w, b, y = 1.5, -0.8, 0.4, 1.0
z1 = w * x
z2 = z1 + b
a = 1 / (1 + math.exp(-z2))
d = a - y
L = d * d
# backward pass
dL = 1.0
dd = 2 * d * dL
da = dd
dz2 = da * a * (1 - a)
dz1, db = dz2, dz2
dw, dx = dz1 * x, dz1 * w

pic = tz.Picture()


def var(text, at, color="Blue"):
    return pic.node(text, at=at, style=f"block={color}", minimum_width=0.9, minimum_height=0.7,
                    rounded_corners=10)


def op(text, at):
    return pic.node(text, at=at, style="op", minimum_size=0.8, draw="Ink!70")


nx, nw, nb, ny = var("$x$", (0, 1.2)), var("$w$", (0, -0.6)), var("$b$", (2.4, -1.9)), var("$y$", (7.2, -1.9))
mul, add, sig, sub, sq = op(r"$\times$", (2.4, 0.3)), op("$+$", (4.8, 0.3)), op(r"$\sigma$", (7.2, 0.3)), \
    op("$-$", (9.6, 0.3)), op(r"$(\cdot)^2$", (12.0, 0.3))
loss = var("$L$", (14.2, 0.3), "Red")

FWD, BWD = "Blue!80!black", "Red!85!black"


def edge(u, v, value, gradient, value_side="above", grad_side="below"):
    pic.draw(u, v, arrow="->", line_width=0.8)
    pic.path(u, tz.label(f"${value:.2f}$", value_side, sloped=True, font=r"\scriptsize", text_color=FWD,
                         pos=0.5), v)
    pic.path(u, tz.label(f"${gradient:.2f}$", grad_side, sloped=True, font=r"\scriptsize", text_color=BWD,
                         pos=0.5), v)


edge(nx, mul, x, dx)
edge(nw, mul, w, dw, "below", "above")
edge(mul, add, z1, dz1)
edge(nb, add, b, db, "below", "above")
edge(add, sig, z2, dz2)
edge(sig, sub, a, da)
edge(ny, sub, y, -dd, "below", "above")
edge(sub, sq, d, dd)
edge(sq, loss, L, dL)

# legend
pic.node("forward value", at=(3.0, -3.1), anchor="west", text_color=FWD, font=r"\small")
pic.node(r"gradient $\partial L / \partial(\cdot)$", at=(6.2, -3.1), anchor="west", text_color=BWD,
         font=r"\small")
pic.dot((2.85, -3.1), color=FWD)
pic.dot((6.05, -3.1), color=BWD)

if __name__ == "__main__":
    pic.render("backprop.pdf")
