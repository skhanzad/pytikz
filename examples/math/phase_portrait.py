"""Phase portrait of a damped Duffing oscillator x'' = x - x^3 - 0.25 x':
two stable spirals, a saddle and the separatrix that splits their basins."""

import numpy as np

import pytikz as tz

DAMPING = 0.25


def field(x, y):
    return y, x - x**3 - DAMPING * y


def trajectory(p, t_end, dt=0.01):
    """RK4 integration (dt < 0 integrates backwards in time)."""
    pts = [np.array(p, dtype=float)]
    for _ in range(int(abs(t_end / dt))):
        x = pts[-1]
        k1 = np.array(field(*x))
        k2 = np.array(field(*(x + dt / 2 * k1)))
        k3 = np.array(field(*(x + dt / 2 * k2)))
        k4 = np.array(field(*(x + dt * k3)))
        nxt = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if np.abs(nxt).max() > 6:
            break
        pts.append(nxt)
    return np.array(pts)


pic = tz.Picture(theme="paper")
ax = pic.add(tz.Axes(width=9, height=6.6, xlim=(-2.2, 2.2), ylim=(-1.6, 1.6), frame="box",
                     xlabel="$x$", ylabel=r"$\dot x$", title=r"$\ddot x = x - x^3 - 0.25\,\dot x$"))
ax.streamlines(field, density=1.1, cmap=tz.get_colormap(["Blue!35", "Blue!80!black"]), line_width=0.7)

# nullclines: y = 0 and y = (x - x^3) / damping
ax.hline(0, color="Orange", dash="--", line_width=0.9, label=r"$\dot x$-nullcline")
xs = np.linspace(-2.2, 2.2, 300)
ax.plot(xs, (xs - xs**3) / DAMPING, color="Green", dash="--", line_width=0.9, label=r"$\ddot x$-nullcline")

# the separatrix: the stable manifold of the saddle at the origin
lam = (-DAMPING - np.sqrt(DAMPING**2 + 4)) / 2          # stable eigenvalue of the Jacobian at 0
v = np.array([1.0, lam]) / np.hypot(1.0, lam)
for sign in (1, -1):
    sep = trajectory(sign * 1e-3 * v, -12, dt=-0.01)
    ax.plot(sep[:, 0], sep[:, 1], color="Red", line_width=1.5, label="separatrix" if sign == 1 else None)

ax.scatter([-1, 1], [0, 0], color="Ink", size=3.0, edgecolor="white", label="stable focus")
ax.scatter([0], [0], color="white", size=3.0, edgecolor="Ink", label="saddle")
ax.legend("upper left", font=r"\scriptsize")

if __name__ == "__main__":
    pic.render("phase_portrait.pdf")
