"""Three surfaces: a saddle, a radial ripple and a parametric torus."""

import numpy as np

import pytikz as tz

pic = tz.Picture()

saddle = pic.add(tz.Axes3D(at=(0, 0), size=(3.6, 3.6, 2.6), xlim=(-1, 1), ylim=(-1, 1), title="$z = x^2 - y^2$",
                           xlabel="$x$", ylabel="$y$"))
saddle.surface(lambda x, y: x**2 - y**2, resolution=26, cmap="coolwarm", edges="dark")

ripple = pic.add(tz.Axes3D(at=(6.8, 0), size=(3.8, 3.8, 2.2), xlim=(-6, 6), ylim=(-6, 6), elevation=35,
                           title=r"$z = \frac{\sin r}{r}$", ticks=False))
ripple.surface(lambda x, y: np.sinc(np.hypot(x, y) / np.pi), resolution=40, cmap="viridis", edges="none")

u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 44), np.linspace(0, 2 * np.pi, 22))
R, r = 1.0, 0.42
X, Y, Z = (R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)
torus = pic.add(tz.Axes3D(at=(13.4, 0), size=(3.6, 3.6, 1.6), xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
                          zlim=(-0.6, 0.6), elevation=38, panes=False, ticks=False, title="torus"))
torus.surface(X=X, Y=Y, Z=Z, color="Orange", edges="dark")

if __name__ == "__main__":
    pic.render("surfaces.pdf")
