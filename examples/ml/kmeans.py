"""k-means clustering: points colored by cluster, the Voronoi cells of the
centroids, and the path each centroid took during the iterations."""

import numpy as np

import pytikz as tz

rng = np.random.default_rng(3)
centers = np.array([[-2.0, 1.2], [1.8, 1.6], [0.2, -1.8]])
X = np.vstack([rng.normal(c, 0.65, size=(45, 2)) for c in centers])

# Lloyd's algorithm from poor initial centroids, keeping the history
mu = np.array([[-0.6, -0.2], [0.0, 0.3], [0.6, -0.4]])
history = [mu.copy()]
for _ in range(8):
    labels = np.argmin(((X[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)
    mu = np.array([X[labels == k].mean(0) for k in range(3)])
    history.append(mu.copy())
history = np.array(history)
labels = np.argmin(((X[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)


def nearest(x, y):
    d = [(x - m[0]) ** 2 + (y - m[1]) ** 2 for m in mu]
    return np.argmin(np.stack(d), axis=0)


COLORS = ["Blue", "Orange", "Green"]
pic = tz.Picture()
ax = pic.add(tz.Axes(width=7.5, height=6.5, xlim=(-4.2, 4.2), ylim=(-3.8, 3.6), frame="box",
                     xticks=None, yticks=None, title="$k$-means, $k = 3$"))
ax.regions(nearest, colors=[f"{c}!10" for c in COLORS], boundary=True, boundary_color="Ink!35",
           boundary_width=0.6)
for k, color in enumerate(COLORS):
    sel = labels == k
    ax.scatter(X[sel, 0], X[sel, 1], color=color, size=2.0, label=f"cluster {k + 1}")
for k, color in enumerate(COLORS):
    ax.plot(history[:, k, 0], history[:, k, 1], color=f"{color}!60!black", line_width=0.9, dash=":")
    ax.scatter(history[:-1, k, 0], history[:-1, k, 1], color=f"{color}!60!black", size=1.1, edgecolor=None)
    ax.scatter([mu[k, 0]], [mu[k, 1]], marker="*", size=5.5, color=f"{color}!70!black", edgecolor="white")
ax.scatter([], [], marker="*", size=4, color="Ink", label="centroid")
ax.legend("lower left")

if __name__ == "__main__":
    pic.render("kmeans.pdf")
