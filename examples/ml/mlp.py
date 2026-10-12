"""A multilayer perceptron with labelled neurons and the forward-pass equations."""

import pytikz as tz

pic = tz.Picture()

net = pic.add(
    tz.NeuralNetwork(
        [3, 4, 4, 2],
        labels=["$x_{i}$", "$h^{(1)}_{i}$", "$h^{(2)}_{i}$", r"$\hat y_{i}$"],
        titles=["Input\\\\layer", "Hidden\\\\layer 1", "Hidden\\\\layer 2", "Output\\\\layer"],
        layer_spacing=2.6,
        neuron_size=0.72,
        arrows=True,
    )
)

# Emphasise one weight and the neuron it feeds.
net.highlight_connection(0, 0, 1, color="Orange", width=1.3, label="$w^{(1)}_{21}$")
net.highlight(1, 1, "Orange")

# The forward pass, aligned at the equals signs, under the network.
forward = r"""$\begin{aligned}
  h^{(1)} &= \sigma\big(W^{(1)}x + b^{(1)}\big)\\
  h^{(2)} &= \sigma\big(W^{(2)}h^{(1)} + b^{(2)}\big)\\
  \hat y  &= \operatorname{softmax}\big(W^{(3)}h^{(2)} + b^{(3)}\big)
\end{aligned}$"""
pic.node(forward, at=(3.9, -2.6), anchor="north", font=r"\small", text_color="Ink!85")

if __name__ == "__main__":
    pic.render("mlp.pdf")
