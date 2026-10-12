"""A generative adversarial network: the generator maps noise to samples, the
discriminator tells them apart from real data."""

import numpy as np

import pytikz as tz

pic = tz.Picture()
rng = np.random.default_rng(5)

# noise vector z as a column of cells
noise = tz.CellGrid(shape=(6, 1), at=(0, 1.6), cell=0.38,
                    fill=lambda i, j: tz.get_colormap("greys")(0.15 + 0.6 * rng.random()),
                    line_color="Gray!60", label="noise $\\mathbf{z}$")
pic.add(noise)

gen = pic.node("Generator\n$G_\\theta$", at=(2.6, 1.6), style="block=Blue", minimum_width=2.2,
               minimum_height=1.6)
pic.arrow(noise.east + (0.1, 0), gen)

# fake and real samples as tiny images
grey = tz.get_colormap(["white", "Ink"])


def sample(at, label, sharp):
    img = np.zeros((5, 5))
    img[1:4, 2] = 1
    img[1, 1] = img[3, 1] = img[3, 3] = 0.8
    if not sharp:
        img = np.clip(img * 0.7 + 0.35 * rng.random((5, 5)), 0, 1)
    grid = tz.CellGrid(shape=(5, 5), at=at, cell=0.26, fill=lambda i, j: grey(img[i, j]),
                       line_color="Gray!40", line_width=0.3, label=label)
    pic.add(grid)
    return grid


fake = sample((5.4, 1.6), "fake $G(\\mathbf{z})$", sharp=False)
real = sample((5.4, -1.0), "real $\\mathbf{x}$", sharp=True)
pic.arrow(gen, fake.west - (0.1, 0))
pic.node("training data", at=(5.4, -2.75), font=r"\footnotesize", text_color="Ink!60")

disc = pic.node("Discriminator\n$D_\\phi$", at=(8.9, 0.3), style="block=Orange", minimum_width=2.4,
                minimum_height=1.6)
pic.arrow(fake.east + (0.1, 0), disc.west, bend=-15)
pic.arrow(real.east + (0.1, 0), disc.west, bend=15)

verdict = pic.node("real or\nfake?", at=(11.5, 0.3), style="block=Green", minimum_width=1.6)
pic.arrow(disc, verdict)

# feedback: gradients flow back into both networks
pic.draw(verdict.south, (11.5, -2.1), (8.9, -2.1), disc.south, arrow="->", dashed=True, color="Red",
         rounded_corners=5)
pic.draw(verdict.north, (11.5, 3.1), (2.6, 3.1), gen.north, arrow="->", dashed=True, color="Red",
         rounded_corners=5)
pic.node("update $\\phi$", at=(10.2, -2.1), anchor="north", font=r"\footnotesize", text_color="Red")
pic.node("update $\\theta$", at=(7.0, 3.1), anchor="south", font=r"\footnotesize", text_color="Red")

pic.node(r"$\min_\theta \max_\phi \; \mathbb{E}_{\mathbf{x}}\left[\log D_\phi(\mathbf{x})\right] + "
         r"\mathbb{E}_{\mathbf{z}}\left[\log\left(1 - D_\phi(G_\theta(\mathbf{z}))\right)\right]$",
         at=(5.8, -3.5), font=r"\small")

if __name__ == "__main__":
    pic.render("gan.pdf")
