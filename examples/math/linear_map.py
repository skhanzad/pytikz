"""A linear map acting on the plane: the grid, the basis vectors and the unit
square before and after, plus the eigenvector directions that only stretch."""

import numpy as np

import pytikz as tz
from pytikz import Point

A = np.array([[1.5, 0.75], [0.25, 1.25]])
vals, vecs = (part.real for part in np.linalg.eig(A))  # real eigenvalues; newer numpy returns complex
S = 0.95          # cm per unit
N = 4             # grid half-width


def panel(origin, M, title, show_eigen):
    """Draw the plane transformed by M, centred at ``origin``."""
    def P(v):
        w = M @ np.asarray(v, dtype=float)
        return Point(origin.x + S * w[0], origin.y + S * w[1])

    box = (origin + Point(-3.2, -3.0), origin + Point(3.2, 3.0))
    with pic.scope() as sc:
        sc.clip(box[0], "rectangle", box[1])
        for k in range(-2 * N, 2 * N + 1):
            sc.draw(P((k, -2 * N)), P((k, 2 * N)), color="Blue!25" if k else "Ink!60", line_width=0.4)
            sc.draw(P((-2 * N, k)), P((2 * N, k)), color="Blue!25" if k else "Ink!60", line_width=0.4)
        sc.polygon([P((0, 0)), P((1, 0)), P((1, 1)), P((0, 1))], fill="Yellow!35", draw="Yellow!70!black",
                   line_width=0.6, fill_opacity=0.85)
        if show_eigen:
            for k in range(2):
                v = vecs[:, k] * 6
                sc.draw(P(-v), P(v), color="Purple!70", dashed=True, line_width=0.8)
        sc.draw(P((0, 0)), P((1, 0)), arrow="->", color="Red", line_width=1.6)
        sc.draw(P((0, 0)), P((0, 1)), arrow="->", color="Green", line_width=1.6)
    pic.rectangle(*box, draw="Ink!40", line_width=0.5)
    pic.node(title, at=Point(origin.x, box[1].y), anchor="south", font=r"\small\bfseries")
    return P


left = Point(0, 0)
right = Point(10.2, 0)
pic = tz.Picture()
P0 = panel(left, np.eye(2), "before", False)
P1 = panel(right, A, r"after $A$", True)

pic.node(r"$\hat\imath$", at=P0((1, 0)), anchor="north", text_color="Red!80!black")
pic.node(r"$\hat\jmath$", at=P0((0, 1)), anchor="east", text_color="Green!60!black")
pic.node(r"$A\hat\imath$", at=P1((1, 0)), anchor="north west", text_color="Red!80!black")
pic.node(r"$A\hat\jmath$", at=P1((0, 1)), anchor="south east", text_color="Green!60!black")
pic.node(fr"area $= \det A = {np.linalg.det(A):.2f}$", at=P1((0.5, 0.5)) + (0.4, -1.9), font=r"\footnotesize",
         fill="white", inner_sep=2)

matrix = r"$A = \begin{pmatrix} 1.5 & 0.75 \\ 0.25 & 1.25 \end{pmatrix}$"
pic.arrow((3.6, 0.7), (6.6, 0.7), bend=25, line_width=1.0)
pic.node(matrix, at=(5.1, -0.5), font=r"\small")
pic.node(r"eigenvectors: $\lambda = " + f"{vals[0]:.2f}, {vals[1]:.2f}$", at=right + (0, -3.0),
         anchor="north", font=r"\footnotesize", text_color="Purple!80!black")

if __name__ == "__main__":
    pic.render("linear_map.pdf")
