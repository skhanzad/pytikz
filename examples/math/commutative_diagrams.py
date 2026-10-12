"""Commutative diagrams: a pullback with its universal property, and the
first isomorphism theorem."""

import pytikz as tz

pic = tz.Picture(theme="paper")

# pullback square with the universal arrow from Q
pb = tz.CommutativeDiagram([[r"Q", None, None], [None, r"P", r"X"], [None, r"Y", r"Z"]],
                           col_sep=2.4, row_sep=1.9, name="pb")
pb.arrow((1, 1), (1, 2), r"p_1").arrow((1, 1), (2, 1), r"p_2", swap=True)
pb.arrow((1, 2), (2, 2), "f").arrow((2, 1), (2, 2), "g", swap=True)
pb.arrow((0, 0), (1, 1), r"\exists!\,u", kind="dashed")
pb.arrow((0, 0), (1, 2), r"q_1", bend=20)
pb.arrow((0, 0), (2, 1), r"q_2", bend=-20, swap=True)
pic.add(pb)
# the pullback corner mark
pic.draw(pb.position((1, 1)) + (0.35, -0.6), pb.position((1, 1)) + (0.6, -0.6),
         pb.position((1, 1)) + (0.6, -0.35), line_width=0.5)

# first isomorphism theorem
iso = tz.CommutativeDiagram([[r"G", r"H"], [r"G/\ker\varphi", None]], at=(7.6, -1.1), col_sep=3.0,
                            row_sep=2.1, name="iso")
iso.arrow((0, 0), (0, 1), r"\varphi")
iso.arrow((0, 0), (1, 0), r"\pi", kind="epi", swap=True)
iso.arrow((1, 0), (0, 1), r"\tilde\varphi", kind="hook", swap=True)
pic.add(iso)
pic.node(r"$G/\ker\varphi \;\cong\; \operatorname{im}\varphi$", at=(9.1, -4.0), font=r"\small")

if __name__ == "__main__":
    pic.render("commutative_diagrams.pdf")
