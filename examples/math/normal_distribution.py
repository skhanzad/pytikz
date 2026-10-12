"""The normal distribution and the 68-95-99.7 rule."""

import numpy as np

import pytikz as tz


def pdf(x):
    return np.exp(-x * x / 2) / np.sqrt(2 * np.pi)


pic = tz.Picture(theme="paper")
ticks = [(k, "$\\mu$" if k == 0 else f"$\\mu {'+' if k > 0 else '-'} {abs(k) if abs(k) > 1 else ''}\\sigma$")
         for k in range(-3, 4)]
ax = pic.add(tz.Axes(width=10, height=5.6, xlim=(-3.8, 3.8), ylim=(0, 0.54), frame="open", xticks=ticks,
                     yticks=None))

shades = [(3, "Blue!12"), (2, "Blue!25"), (1, "Blue!42")]
for k, color in shades:
    xs = np.linspace(-k, k, 200)
    ax.fill_between(xs, pdf(xs), 0, color=color, opacity=1.0)
for k in range(-3, 4):
    ax.line([(k, 0), (k, pdf(k))], color="Blue!60!black" if k else "Ink", line_width=0.6,
            dashed=k != 0)
ax.function(pdf, (-3.8, 3.8), color="Blue!70!black", line_width=1.8)

# percentages between the lines
for x, text in ((0.5, "34.1\\%"), (-0.5, "34.1\\%"), (1.5, "13.6\\%"), (-1.5, "13.6\\%"),
                (2.5, "2.1\\%"), (-2.5, "2.1\\%")):
    ax.text(x, 0.035 if abs(x) > 2 else 0.06, text, font=r"\scriptsize" if abs(x) > 2 else r"\small")

# brackets above the curve
for k, y, text in ((1, 0.43, "68.3\\%"), (2, 0.475, "95.4\\%"), (3, 0.52, "99.7\\%")):
    left, right = ax.point(-k, y), ax.point(k, y)
    pic.draw(left, right, style="|<->|", color="Ink!70", line_width=0.6)
    pic.node(text, at=ax.point(0, y), fill="white", inner_sep=1.2, font=r"\footnotesize")

pic.node(r"$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}$",
         at=ax.point(2.6, 0.3), font=r"\small")

if __name__ == "__main__":
    pic.render("normal_distribution.pdf")
