"""A variational autoencoder with the reparameterisation trick.

The encoder/decoder trapezoids are plain polygons; the input and the
reconstruction are small pixel grids.
"""

import numpy as np

import pytikz as tz

pic = tz.Picture()
rng = np.random.default_rng(4)

# A tiny "digit" and a blurrier reconstruction of it.
digit = np.zeros((7, 7))
digit[1:6, 3] = 1.0
digit[1, 2] = digit[5, 2] = digit[5, 4] = 0.9
blurred = np.clip(digit * 0.8 + 0.12 * rng.random((7, 7)), 0, 1)

gray = tz.get_colormap(["white", "Ink"])
x_img = tz.CellGrid(values=None, shape=(7, 7), at=(0, 0), cell=0.3,
                    fill=lambda i, j: gray(digit[i, j]), line_color="Gray!40", line_width=0.3,
                    label=r"$\mathbf{x}$")
x_hat = tz.CellGrid(shape=(7, 7), at=(11.6, 0), cell=0.3, fill=lambda i, j: gray(blurred[i, j]),
                    line_color="Gray!40", line_width=0.3, label=r"$\hat{\mathbf{x}}$")
pic.add(x_img, x_hat)


def trapezoid(x0, x1, h0, h1, color, text):
    """A trapezoid between x0 and x1, centred on y = 0, with heights h0 and h1."""
    pts = [(x0, -h0 / 2), (x1, -h1 / 2), (x1, h1 / 2), (x0, h0 / 2)]
    pic.polygon(pts, fill=f"{color}!18", draw=f"{color}!75!black", line_width=0.8, rounded_corners=2)
    return pic.node(text, at=((x0 + x1) / 2, 0), font=r"\small", text_color=f"{color}!60!black")


enc = trapezoid(1.6, 4.0, 3.0, 1.5, "Blue", "Encoder\n$q_\\phi(\\mathbf{z}\\mid\\mathbf{x})$")
dec = trapezoid(7.6, 10.0, 1.5, 3.0, "Purple", "Decoder\n$p_\\theta(\\mathbf{x}\\mid\\mathbf{z})$")

mu = pic.node(r"$\boldsymbol{\mu}$", at=(4.9, 0.55), style="block=Orange", minimum_width=0.8, minimum_height=0.6)
sigma = pic.node(r"$\boldsymbol{\sigma}$", at=(4.9, -0.55), style="block=Yellow", minimum_width=0.8,
                 minimum_height=0.6)
z = pic.node(r"$\mathbf{z}$", at=(6.6, 0), style="block=Green", minimum_width=0.8, minimum_height=0.8)
eps = pic.node(r"$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$", at=(6.6, -1.75),
               font=r"\small")

pic.arrow(x_img.east + (0.1, 0), (1.6, 0))
pic.arrow((4.0, 0.55), mu.west)
pic.arrow((4.0, -0.55), sigma.west)
pic.arrow(mu.east, z, route="-|", label_side=None)
pic.arrow(sigma.east, z, route="-|")
pic.arrow(eps, z.south)
pic.arrow(z.east, (7.6, 0))
pic.arrow((10.0, 0), x_hat.west - (0.1, 0))

pic.node(r"$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$",
         at=(6.6, 1.25), font=r"\small", text_color="Green!50!black")

# The two terms of the loss.
pic.node(
    r"$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi}\!\left[\log p_\theta(\mathbf{x}\mid\mathbf{z})\right]}"
    r"_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x})"
    r"\,\|\,p(\mathbf{z})\right)}_{\text{regularisation}}$",
    at=(5.8, -2.9), font=r"\small",
)

if __name__ == "__main__":
    pic.render("vae.pdf")
