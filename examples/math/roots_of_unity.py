"""The n-th roots of unity on the unit circle, and Euler's formula."""

import pytikz as tz
from pytikz import Point

n = 7
R = 2.6
pic = tz.Picture(theme="paper")
pic.draw((-1.35 * R, 0), (1.4 * R, 0), arrow="->", color="Ink!65")
pic.draw((0, -1.3 * R), (0, 1.35 * R), arrow="->", color="Ink!65")
pic.node(r"$\operatorname{Re}$", at=(1.4 * R, 0), anchor="west")
pic.node(r"$\operatorname{Im}$", at=(0, 1.35 * R), anchor="south")
pic.circle((0, 0), R, color="Ink!40", dashed=True)

roots = [Point.polar(360 * k / n, R) for k in range(n)]
pic.polygon(roots, fill="Blue!8", draw="Blue!70", line_width=0.9)
for k, p in enumerate(roots):
    pic.draw((0, 0), p, color="Blue!40", line_width=0.5)
    pic.dot(p, f"$\\omega^{{{k}}}$" if k else r"$1$", position=360 * k / n if k else "below right",
            color="Blue!70!black", size=4.2)

pic.angle_mark(roots[0], (0, 0), roots[1], r"$\frac{2\pi}{7}$", radius=0.9, fill="Orange!25",
               label_distance=1.45)
pic.node(r"$\omega = e^{2\pi i/7} = \cos\frac{2\pi}{7} + i\sin\frac{2\pi}{7}$", at=(0, -1.3 * R - 0.25),
         anchor="north", font=r"\small")
pic.node(r"$z^7 = 1$", at=(-1.25 * R, 1.2 * R), font=r"\large")

if __name__ == "__main__":
    pic.render("roots_of_unity.pdf")
