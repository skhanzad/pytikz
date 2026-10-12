"""Optimiser trajectories on an ill-conditioned quadratic: plain gradient
descent zig-zags, momentum and Adam go more directly to the minimum."""

import numpy as np

import pytikz as tz

# f(w) = 1/2 w^T A w with A rotated so the valley is diagonal
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
A = R @ np.diag([1.0, 20.0]) @ R.T


def loss(x, y):
    return 0.5 * (A[0, 0] * x * x + 2 * A[0, 1] * x * y + A[1, 1] * y * y)


def grad(w):
    return A @ w


def run(update, steps=30, start=(-2.45, -0.75)):
    w = np.array(start, dtype=float)
    state = {}
    path = [w.copy()]
    for t in range(1, steps + 1):
        w = update(w, grad(w), state, t)
        path.append(w.copy())
    return np.array(path)


def sgd(w, g, s, t, lr=0.09):
    return w - lr * g


def momentum(w, g, s, t, lr=0.012, beta=0.88):
    s["v"] = beta * s.get("v", 0) + g
    return w - lr * s["v"]


def adam(w, g, s, t, lr=0.16, b1=0.85, b2=0.99):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g * g
    m_hat, v_hat = s["m"] / (1 - b1**t), s["v"] / (1 - b2**t)
    return w - lr * m_hat / (np.sqrt(v_hat) + 1e-8)


pic = tz.Picture()
ax = pic.add(tz.Axes(width=10, height=6, xlim=(-3, 3), ylim=(-1.8, 1.8), frame="box",
                     xlabel="$w_1$", ylabel="$w_2$", title="Optimisers on an ill-conditioned quadratic"))
ax.contourf(loss, levels=np.geomspace(0.05, 40, 13), cmap=tz.get_colormap(["white", "Blue!32"]),
            line_color="Blue!45", line_width=0.35)

for name, update, color, marker in (("Gradient descent", sgd, "Red", "o"),
                                    ("Momentum", momentum, "Orange", "s"),
                                    ("Adam", adam, "Green", "^")):
    path = run(update)
    ax.plot(path[:, 0], path[:, 1], color=color, line_width=1.1, marker=marker, marker_size=1.7, label=name)

ax.scatter([0], [0], marker="*", size=4.5, color="Ink", edgecolor="white")
ax.text(0.1, -0.25, "minimum", anchor="north west", font=r"\footnotesize")
ax.legend("upper right")

if __name__ == "__main__":
    pic.render("gradient_descent.pdf")
