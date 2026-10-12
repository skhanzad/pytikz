"""Matrix multiplication: entry (i, j) of AB is row i of A times column j of B."""

import numpy as np

import pytikz as tz

A = np.array([[1, 2, 0], [3, -1, 4]])
B = np.array([[2, 1], [0, 5], [-2, 3]])
C = A @ B
i, j = 1, 0

pic = tz.Picture(theme="paper")


def entries(M, fmt="{}"):
    return [[fmt.format(v) for v in row] for row in M]


ma = pic.add(tz.Matrix(entries(A), at=(0, 0), name="A", label=r"$A$ \; ($2\times 3$)"))
mb = pic.add(tz.Matrix(entries(B), at=(4.4, 0), name="B", label=r"$B$ \; ($3\times 2$)"))
mc = pic.add(tz.Matrix(entries(C), at=(8.4, 0), name="C", label=r"$AB$ \; ($2\times 2$)"))
ma.highlight_row(i, "Blue")
mb.highlight_column(j, "Orange")
mc.highlight([(i, j)], "Green")

pic.node(r"$\times$", at=(2.25, 0), font=r"\Large")
pic.node(r"$=$", at=(6.5, 0), font=r"\Large")

terms = " + ".join(f"({A[i, k]})({B[k, j]})" for k in range(3))
pic.node(f"$c_{{{i + 1}{j + 1}}} = \\sum_k a_{{{i + 1}k}}\\, b_{{k{j + 1}}} = {terms} = {C[i, j]}$",
         at=(4.4, -2.6), font=r"\small")

if __name__ == "__main__":
    pic.render("matrix_product.pdf")
