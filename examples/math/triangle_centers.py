"""Triangle centres: circumcentre O, centroid G and orthocentre H lie on the
Euler line; the incircle touches all three sides."""

import pytikz as tz
from pytikz import Point
from pytikz.geometry import centroid, circumcircle, foot, incircle, orthocenter

A, B, C = Point(0, 0), Point(8.0, 0), Point(1.5, 5.0)
O, R = circumcircle(A, B, C)
I, r = incircle(A, B, C)
G = centroid(A, B, C)
H = orthocenter(A, B, C)

pic = tz.Picture(theme="paper")
pic.circle(O, R, color="Blue!70", line_width=0.8)
pic.polygon([A, B, C], fill="Orange!6", draw="none")
pic.circle(I, r, color="Green!60!black", line_width=0.8, fill="Green!8")
pic.polygon([A, B, C], line_width=1.1)

# altitudes (thin, dashed) meeting at H
for v, (p, q) in ((A, (B, C)), (B, (A, C)), (C, (A, B))):
    f = foot(v, p, q)
    pic.draw(v, f, color="Red!60", dashed=True, line_width=0.5)
    pic.right_angle_mark(v, f, q, size=0.18, color="Red!60", line_width=0.4)

# medians (thin) meeting at G
for v, (p, q) in ((A, (B, C)), (B, (A, C)), (C, (A, B))):
    pic.draw(v, p.lerp(q, 0.5), color="Purple!50", line_width=0.4)

# the Euler line through O, G and H, extended a little
d = (H - O).unit()
pic.draw(O - d * 1.5, H + d * 1.2, color="Ink!80", line_width=1.0, dashed=True)

# angle marks
pic.angle_mark(B, A, C, r"$\alpha$", radius=0.7, fill="Orange!25")
pic.angle_mark(C, B, A, r"$\beta$", radius=0.7, fill="Orange!25")
pic.angle_mark(A, C, B, r"$\gamma$", radius=0.6, fill="Orange!25")

for p, name, pos in ((A, "A", "below left"), (B, "B", "below right"), (C, "C", "above")):
    pic.dot(p, f"${name}$", position=pos, size=3.5)
for p, name, pos, color in ((O, "O", "below right", "Blue"), (G, "G", "below right", "Purple"),
                            (H, "H", "above right", "Red"), (I, "I", "left", "Green!60!black")):
    pic.dot(p, f"${name}$", position=pos, color=color, size=3.8, label_color=color)

angle = d.angle()
if angle > 90:
    angle -= 180
elif angle <= -90:
    angle += 180
pic.node("Euler line", at=O - d * 1.5, anchor="south", font=r"\footnotesize", rotate=angle)
pic.node("circumcircle", at=O + Point.polar(-60, R), anchor="north west", font=r"\footnotesize",
         text_color="Blue!80!black")
pic.node("incircle", at=I + Point(0, -0.72 * r), font=r"\footnotesize", text_color="Green!50!black")

if __name__ == "__main__":
    pic.render("triangle_centers.pdf")
