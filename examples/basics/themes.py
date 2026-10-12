"""One diagram, four themes. Every picture produced by this script is a
separate figure (``pytikz render themes.py`` writes one file per theme)."""

import numpy as np

import pytikz as tz


def figure(theme):
    pic = tz.Picture(theme=theme)
    x = pic.node("$x$", at=(0, 0), style="block=input", minimum_width=1.0)
    h = pic.node("hidden", right_of=x, distance=0.9, style="block=hidden")
    y = pic.node(r"$\hat y$", right_of=h, distance=0.9, style="block=output", minimum_width=1.0)
    pic.arrow(x, h)
    pic.arrow(h, y)
    pic.node(theme, at=(2.5, 1.0), font=r"\small\bfseries")
    ax = pic.add(tz.Axes(at=(0, -3.6), width=5.2, height=2.6, xlim=(-3, 3), grid=True))
    xs = np.linspace(-3, 3, 200)
    ax.plot(xs, np.tanh(xs), label="tanh")
    ax.plot(xs, 1 / (1 + np.exp(-xs)), label="sigmoid")
    ax.legend("upper left")
    return pic


modern = figure("modern")
paper = figure("paper")
dark = figure("dark")
mono = figure("mono")

if __name__ == "__main__":
    for name, pic in (("modern", modern), ("paper", paper), ("dark", dark), ("mono", mono)):
        pic.render(f"theme_{name}.pdf")
