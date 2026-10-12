"""A tour of the core API, one small section per feature."""

import pytikz as tz
from pytikz.geometry import regular_polygon

pic = tz.Picture()


def caption(x, text):
    pic.node(text, at=(x, -2.4), font=r"\small\itshape", text_color="Ink!60")


# your own named style, used like the built-in ones
pic.style("pill", "draw, rounded corners=7pt, inner sep=5pt, minimum width=1.4cm", fill="Purple!12",
          draw="Purple!70!black")

# 1. nodes, relative placement and labelled arrows
a = pic.node("$A$", at=(0, 0.6), style="block", minimum_width=1.1)
b = pic.node("$B$", right_of=a, distance=1.4, style="block=Orange", minimum_width=1.1)
c = pic.node("$C$", below_of=b, distance=0.9, style="pill")
pic.arrow(a, b, label="$f$")
pic.arrow(b, c, label="$g$")
pic.arrow(a, c, label=r"$g\circ f$", label_side="swap", route="|-", label_pos=0.75, color="Ink!70")
caption(1.25, "nodes, styles, arrows")

# 2. straight paths, curves, arcs and relative moves
pic.draw((4.6, -1.4), (5.5, 0.9), (6.4, -1.4), "cycle", fill="Yellow!25", rounded_corners=3)
pic.draw((7.0, -1.4), tz.curve((7.4, 1.3), (8.7, -2.0)), (9.2, 0.6), arrow="->", color="Blue", line_width=1.2)
pic.draw((10.6, -0.4), tz.arc(0, 270, 0.7), "--", tz.rel(0.7, 0), arrow="->", color="Green!60!black",
         line_width=1.0)
caption(7.6, "polygons, Bézier curves, arcs")

# 3. calc: points between nodes and anchors with offsets
p = pic.node("$P$", at=(12.6, 0.8), style="block=Teal", minimum_width=0.9)
q = pic.node("$Q$", at=(14.6, -0.9), style="block=Teal", minimum_width=0.9)
pic.draw(p, q, dashed=True, color="Ink!50")
for t in (0.25, 0.5, 0.75):
    pic.dot(tz.between(p, q, t), color="Red", size=3.5)
pic.node(r"$\tfrac12(P+Q)$", at=tz.midpoint(p, q) + (-0.15, -0.1), anchor="north east", font=r"\footnotesize")
pic.draw(q.north, q.north + (0, 0.9), arrow="->", color="Orange")
caption(13.6, "calc coordinates")

# 4. a rotated scope: everything inside is transformed
with pic.scope(shift=(17.6, -0.2), rotate=25) as s:
    s.rectangle((-0.9, -0.6), (0.9, 0.6), fill="Purple!12", draw="Purple!70!black")
    s.node("rotated", font=r"\small")
caption(17.6, "scopes")

# 5. Python geometry feeding TikZ
hexagon = regular_polygon(6, radius=0.85, center=(21.0, -0.2))
pic.polygon(hexagon, fill="Orange!15", draw="Orange!70!black")
pic.circle((21.0, -0.2), 0.85, dashed=True, color="Ink!50")
for v in hexagon:
    pic.dot(v, size=3)
pic.brace((20.15, 0.95), (21.85, 0.95), "brace")
caption(21.0, "geometry, braces")

if __name__ == "__main__":
    pic.render("tour.pdf")
