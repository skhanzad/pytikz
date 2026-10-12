"""A DFA that accepts binary numbers divisible by 3.

State q_r means "the bits read so far have remainder r mod 3"; reading bit b
moves from remainder r to (2r + b) mod 3."""

import pytikz as tz

states = ["q_0", "q_1", "q_2"]
transitions = [(f"q_{r}", f"q_{(2 * r + b) % 3}", str(b)) for r in range(3) for b in (0, 1)]

pic = tz.Picture(theme="paper")
dfa = pic.add(tz.Automaton(states, transitions, initial="q_0", accepting=["q_0"],
                           layout={"q_0": (0, 0), "q_1": (3.2, 0), "q_2": (6.4, 0)}, node_size=0.95))
pic.node(r"accepts $\{\,w \in \{0,1\}^* : w \text{ is a multiple of } 3 \text{ in binary}\,\}$",
         at=(3.2, -2.0), font=r"\small")

if __name__ == "__main__":
    pic.render("automaton.pdf")
