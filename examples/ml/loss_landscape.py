"""A non-convex loss surface in 3D with contour lines on the floor and a
gradient-descent trajectory ending in a local minimum."""

import numpy as np

import pytikz as tz


def loss(x, y):
    return (0.25 * (x**2 + y**2) - 1.1 * np.exp(-((x - 1.0) ** 2 + (y - 0.6) ** 2))
            - 0.7 * np.exp(-((x + 1.1) ** 2 + (y + 0.9) ** 2) / 0.6) + 1.6)


def grad(p, h=1e-5):
    x, y = p
    return np.array([(loss(x + h, y) - loss(x - h, y)) / (2 * h), (loss(x, y + h) - loss(x, y - h)) / (2 * h)])


p = np.array([-2.1, 1.7])
path = [p.copy()]
for _ in range(40):
    p = p - 0.35 * grad(p)
    path.append(p.copy())
path = np.array(path)

pic = tz.Picture()
ax = pic.add(tz.Axes3D(size=(5, 5, 3.2), xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), azimuth=-58, elevation=30,
                       xlabel="$\\theta_1$", ylabel="$\\theta_2$", zlabel="$\\mathcal{L}$"))
ax.contour_floor(loss, levels=12, cmap="viridis", line_width=0.5)
ax.surface(loss, resolution=36, cmap="viridis", edges="dark", opacity=None)
ax.plot3d(path[:, 0], path[:, 1], loss(path[:, 0], path[:, 1]), color="Red", line_width=1.6, marker=True,
          marker_size=1.4, lift=0.04)
# the same path projected onto the floor
z0 = ax.zlim[0]
ax.plot3d(path[:, 0], path[:, 1], np.full(len(path), z0), color="Red", line_width=0.9, dash="densely dashed")

if __name__ == "__main__":
    pic.render("loss_landscape.pdf")
