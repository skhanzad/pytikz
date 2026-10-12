"""A three-set Venn diagram with translucent fills and a labelled region."""

import pytikz as tz
from pytikz import Point

r = 1.9
centres = {"A": Point(0, 0.6), "B": Point(2.3, 0.6), "C": Point(1.15, -1.4)}
colors = {"A": "Blue", "B": "Orange", "C": "Green"}

pic = tz.Picture()
pic.rectangle((-2.6, -4.0), (4.9, 3.2), draw="Ink!60", line_width=0.7, rounded_corners=4)
pic.node(r"$U$", at=(-2.6, 3.2), anchor="north west", font=r"\large")

for name, c in centres.items():
    pic.circle(c, r, fill=f"{colors[name]}!25", draw="none", fill_opacity=0.6)

# A ∩ B ∩ C highlighted: clip to A, then to B, then fill C
with pic.scope() as sc:
    sc.clip(centres["A"], tz.circle(r))
    sc.clip(centres["B"], tz.circle(r))
    sc.circle(centres["C"], r, fill="Purple!75", draw="none")

for name, c in centres.items():
    pic.circle(c, r, draw=f"{colors[name]}!75!black", line_width=1.0)
for name, offset in (("A", (-1.3, 1.25)), ("B", (1.3, 1.25)), ("C", (0, -1.55))):
    pic.node(f"${name}$", at=centres[name] + offset, font=r"\Large", text_color=f"{colors[name]}!70!black")

pic.node(r"$A\cap B\cap C$", at=(1.15, -0.05), font=r"\scriptsize", text_color="white")
pic.node(r"$|A\cup B\cup C| = |A|+|B|+|C|-|A\cap B|-|A\cap C|-|B\cap C|+|A\cap B\cap C|$",
         at=(1.15, -4.35), anchor="north", font=r"\small")

if __name__ == "__main__":
    pic.render("venn.pdf")
