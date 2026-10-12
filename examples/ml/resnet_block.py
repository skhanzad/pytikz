"""A residual block (He et al., 2016): F(x) + x with an identity shortcut."""

import pytikz as tz

pic = tz.Picture()

x_in = pic.node("$\\mathbf{x}$", at=(0, 0))
w1 = pic.node("weight layer", at=(0, -1.3), style="block=Blue", minimum_width=3.2)
relu1 = pic.node("ReLU", at=(0, -2.45), style="block=Teal", minimum_width=3.2, minimum_height=0.6)
w2 = pic.node("weight layer", at=(0, -3.6), style="block=Blue", minimum_width=3.2)
add = pic.node("$+$", at=(0, -4.9), style="op", minimum_size=0.65)
relu2 = pic.node("ReLU", at=(0, -6.0), style="block=Teal", minimum_width=3.2, minimum_height=0.6)
out = pic.node("$\\mathcal{F}(\\mathbf{x}) + \\mathbf{x}$", at=(0, -7.1))

pic.arrow(x_in, w1)
pic.arrow(w1, relu1)
pic.arrow(relu1, w2)
pic.arrow(w2, add)
pic.arrow(add, relu2)
pic.arrow(relu2, out)

# identity shortcut on the right
pic.dot((0, -0.5), size=3.5)
pic.draw((0, -0.5), (2.6, -0.5), tz.xy((2.6, 0), add), add.east, arrow="->", rounded_corners=6)
pic.node("$\\mathbf{x}$", at=(2.6, -2.45), anchor="west", xshift=0.1)
pic.node("identity", at=(2.6, -2.85), anchor="west", xshift=0.1, font=r"\footnotesize", text_color="Ink!65")

# what the stacked layers compute
pic.brace((-1.75, -4.1), (-1.75, -0.8), r"$\mathcal{F}(\mathbf{x})$")

if __name__ == "__main__":
    pic.render("resnet_block.pdf")
