"""sin, cos and tan on the unit circle."""

import math

import pytikz as tz
from pytikz import Point

R = 2.8                     # drawing radius of the unit circle (cm)
theta = 40.0                # the angle we illustrate (degrees)
c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))

O = Point(0, 0)
P = Point(R * c, R * s)
foot = Point(R * c, 0)
T = Point(R, R * math.tan(math.radians(theta)))

pic = tz.Picture(theme="paper")

# axes, circle and special angles
pic.draw((-1.25 * R, 0), (1.35 * R, 0), arrow="->", color="Ink!70")
pic.draw((0, -1.2 * R), (0, 1.25 * R), arrow="->", color="Ink!70")
pic.node("$x$", at=(1.35 * R, 0), anchor="west")
pic.node("$y$", at=(0, 1.25 * R), anchor="south")
pic.circle(O, R, line_width=1.0)
for deg in range(0, 360, 30):
    pic.dot(Point.polar(deg, R), size=2.6, color="Ink!45")
for deg in (45, 135, 225, 315):
    pic.dot(Point.polar(deg, R), size=2.6, color="Ink!45")
for p, text, pos in (((R, 0), "$1$", "below left"), ((-R, 0), "$-1$", "below left"),
                     ((0, R), "$1$", "above left"), ((0, -R), "$-1$", "below left")):
    pic.node(text, at=p, anchor={"below right": "north west", "below left": "north east",
                                 "above left": "south east"}[pos], font=r"\small")

# tangent line x = 1 and the extended radius
pic.draw((R, -0.35 * R), (R, 1.2 * R), color="Green!60", dashed=True)
pic.draw(P, T, color="Ink!50", dashed=True)

# the three segments
pic.draw(O, foot, color="Blue", line_width=2.2)
pic.draw(foot, P, color="Red", line_width=2.2)
pic.draw((R, 0), T, color="Green", line_width=2.2)
pic.node(r"$\cos\theta$", at=O.lerp(foot, 0.5), anchor="north", text_color="Blue!80!black", yshift=-0.05)
pic.node(r"$\sin\theta$", at=foot.lerp(P, 0.5), anchor="east", text_color="Red!80!black")
pic.node(r"$\tan\theta$", at=Point(R, T.y / 2), anchor="west", text_color="Green!60!black")

# radius, angle and right angle
pic.draw(O, P, line_width=1.2)
pic.node("$1$", at=O.lerp(P, 0.55), anchor="south east", font=r"\small")
pic.angle_mark((R, 0), O, P, r"$\theta$", radius=0.75, fill="Orange!20", color="Orange!80!black",
               label_distance=1.05)
pic.right_angle_mark(O, foot, P, size=0.22, color="Ink!70")
pic.dot(P, r"$P = (\cos\theta, \sin\theta)$", position="above left", color="Ink")
pic.dot(T, position="right", color="Green!60!black")

if __name__ == "__main__":
    pic.render("unit_circle.pdf")
