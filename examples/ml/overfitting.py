"""Under- and overfitting: polynomials of increasing degree fitted to noisy
samples of a smooth function."""

import numpy as np

import pytikz as tz

rng = np.random.default_rng(0)


def truth(x):
    return np.sin(2 * np.pi * x)


x = np.sort(rng.random(14))
y = truth(x) + rng.normal(0, 0.22, x.shape)
grid = np.linspace(0, 1, 400)

pic = tz.Picture()
for k, (degree, title) in enumerate(((1, "Underfitting"), (4, "Good fit"), (13, "Overfitting"))):
    coeffs = np.polyfit(x, y, degree)
    ax = pic.add(tz.Axes(at=(k * 5.6, 0), width=4.8, height=3.6, xlim=(0, 1), ylim=(-1.8, 1.8), frame="box",
                         title=f"{title} (degree {degree})", xticks=None, yticks=None))
    ax.plot(grid, truth(grid), color="Gray!70", dash="--", line_width=1.0)
    ax.plot(grid, np.polyval(coeffs, grid), color=["Orange", "Green", "Red"][k], line_width=1.6)
    ax.scatter(x, y, color="Blue", size=2.0)
    mse = np.mean((np.polyval(coeffs, x) - y) ** 2)
    ax.text(0.03, -1.7, f"train MSE {mse:.3f}", anchor="south west", font=r"\scriptsize", text_color="Ink!70")

if __name__ == "__main__":
    pic.render("overfitting.pdf")
