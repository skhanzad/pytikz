"""A recurrent network, folded (left) and unrolled through time (right)."""

import pytikz as tz

pic = tz.Picture()


def cell(x, text="$A$"):
    return pic.node(text, at=(x, 0), style="block=Green", minimum_width=1.2, minimum_height=1.0)


def io(x, y, text, color):
    return pic.node(text, at=(x, y), style=f"block={color}", minimum_width=0.9, minimum_height=0.7,
                    rounded_corners=10)


# folded network with its recurrence
a = cell(0)
pic.arrow(io(0, -1.7, "$x_t$", "Blue"), a)
pic.arrow(a, io(0, 1.7, "$h_t$", "Purple"))
pic.draw((0.6, 0.15), (1.05, 0.15), (1.05, 0.95), (0.3, 0.95), (0.3, 0.5),
         arrow="->", rounded_corners=6, color="Green!60!black")

pic.node("$=$", at=(2.3, 0), font=r"\Large")

# unrolled through time
xs = [4.0, 6.2, 8.4, 11.6]
names = ["0", "1", "2", "t"]
cells = []
for x, t in zip(xs, names):
    c = cell(x)
    pic.arrow(io(x, -1.7, f"$x_{t}$", "Blue"), c)
    pic.arrow(c, io(x, 1.7, f"$h_{t}$", "Purple"))
    cells.append(c)

for left, right in zip(cells[:2], cells[1:3]):
    pic.arrow(left, right, color="Green!60!black")
pic.arrow(cells[2], (9.55, 0), color="Green!60!black")
pic.node(r"$\cdots$", at=(10.0, 0), font=r"\Large")
pic.arrow((10.45, 0), cells[3], color="Green!60!black")

pic.node(r"$h_t = \tanh\left(W_{hh}\,h_{t-1} + W_{xh}\,x_t + b_h\right)$", at=(7.8, -2.75),
         font=r"\small")

if __name__ == "__main__":
    pic.render("rnn.pdf")
