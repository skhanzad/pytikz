"""A 2D convolution: a 3x3 kernel slides over the input; one output value is
the sum of the element-wise products in the highlighted window."""

import numpy as np

import pytikz as tz

pic = tz.Picture()

image = np.array([
    [3, 3, 2, 1, 0, 1],
    [0, 0, 1, 3, 1, 2],
    [3, 1, 2, 2, 3, 0],
    [2, 0, 0, 2, 2, 1],
    [2, 0, 0, 0, 1, 3],
    [1, 2, 3, 1, 0, 2],
])
kernel = np.array([[0, 1, 2], [2, 2, 0], [0, 1, 2]])
out = np.array([[int((image[i:i + 3, j:j + 3] * kernel).sum()) for j in range(4)] for i in range(4)])

row, col = 1, 2  # the window we explain

cell = 0.62
inp = tz.CellGrid(values=image, at=(0, 0), cell=cell, label="input $I$ ($6\\times 6$)")
inp.color_block(row, col, row + 2, col + 2, "Blue!14")
inp.outline(row, col, row + 2, col + 2, color="Blue", line_width=1.6)

ker = tz.CellGrid(values=kernel, at=(4.4, 0), cell=cell, fill="Orange!16", label="kernel $K$")
ker.outline(0, 0, 2, 2, color="Orange", line_width=1.6)

res = tz.CellGrid(values=out, at=(8.6, 0), cell=cell, label="output $I * K$ ($4\\times 4$)")
res.color_cell(row, col, "Green!22")
res.outline(row, col, color="Green", line_width=1.6)

pic.add(inp, ker, res)

# dashed "projection" lines: window -> kernel -> output cell
window = inp.block_corners(row, col, row + 2, col + 2)
for a, b in (("north east", "north west"), ("south east", "south west")):
    pic.draw(window[a], ker.anchor(b), dashed=True, color="Blue!70", line_width=0.5)
target = res.block_corners(row, col, row, col)
for a, b in (("north east", "north west"), ("south east", "south west")):
    pic.draw(ker.anchor(a), target[b], dashed=True, color="Orange!80!black", line_width=0.5)

pic.node(r"$\ast$", at=(2.65, 0), font=r"\Large")
pic.node(r"$=$", at=(6.25, 0), font=r"\Large")

# the arithmetic for the highlighted value
patch = image[row:row + 3, col:col + 3]
terms = " + ".join(f"{p}\\cdot{k}" for p, k in zip(patch.ravel(), kernel.ravel()))
pic.node(f"${terms} = \\mathbf{{{out[row, col]}}}$", at=(4.4, -2.75), font=r"\small")

if __name__ == "__main__":
    pic.render("convolution.pdf")
