"""The Pythagorean theorem: the squares on the legs add up to the square on
the hypotenuse."""

import pytikz as tz
from pytikz import Point

a, b = 3.0, 4.0
C = Point(0, 0)          # right angle
A = Point(b, 0)
B = Point(0, a)


def square(p, q, color, text):
    """Square on segment pq, on its right-hand side (outside the triangle)."""
    d = q - p
    n = Point(d.y, -d.x)
    corners = [p, q, q + n, p + n]
    pic.polygon(corners, fill=f"{color}!22", draw=f"{color}!75!black", line_width=0.9)
    centre = p + d / 2 + n / 2
    pic.node(text, at=centre, font=r"\Large", text_color=f"{color}!60!black")
    return corners


pic = tz.Picture(theme="paper")
square(C, A, "Blue", "$b^2$")
square(B, C, "Orange", "$a^2$")
square(A, B, "Purple", "$c^2$")
pic.polygon([C, A, B], fill="Yellow!30", line_width=1.2)
pic.right_angle_mark(A, C, B, size=0.35)

pic.node("$b$", at=C.lerp(A, 0.5), anchor="south", yshift=0.05)
pic.node("$a$", at=C.lerp(B, 0.5), anchor="west", xshift=0.05)
pic.node("$c$", at=A.lerp(B, 0.5), anchor="north east")
pic.node(r"$a^2 + b^2 = c^2$", at=(-1.9, -1.6), font=r"\Large")
pic.node(r"$3^2 + 4^2 = 5^2$", at=(-1.9, -2.4), font=r"\normalsize", text_color="Ink!70")

if __name__ == "__main__":
    pic.render("pythagoras.pdf")
