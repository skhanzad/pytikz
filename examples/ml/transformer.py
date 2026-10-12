"""The Transformer architecture (Vaswani et al., 2017): encoder on the left,
decoder on the right.

Block positions are computed in Python. Residual connections, attention
inputs and the encoder-decoder link are ordinary paths; the "N x" groups are
boxes fitted around the blocks.
"""

import math

import pytikz as tz

pic = tz.Picture()

W = 3.0             # block width
ENC, DEC = 0.0, 5.6  # x centres of the encoder and decoder stacks
SIDE = 1.9          # how far residual connections stand off the stack
SPREAD = 0.85       # horizontal spacing of the V, K, Q inputs


def block(text, x, y, color, height=0.8):
    """A sub-layer block centred at (x, y)."""
    return pic.node(text, at=(x, y), style=f"block={color}", minimum_width=W,
                    minimum_height=height, font=r"\small")


def residual(x, y, add_node, side):
    """Skip connection from the main line at (x, y) around into ``add_node``."""
    sx = x + side * SIDE
    anchor = add_node.east if side > 0 else add_node.west
    pic.dot((x, y), size=3.5)
    pic.draw((x, y), (sx, y), tz.xy((sx, 0), anchor), anchor, arrow="->", rounded_corners=4)


def self_attention_inputs(att, x, y):
    """V, K and Q all come from the main line, which ends at (x, y)."""
    split = y + 0.3
    for dx in (-SPREAD, 0.0, SPREAD):
        pic.draw((x, y), (x, split), (x + dx, split), tz.xy((x + dx, 0), att.south),
                 arrow="->", rounded_corners=3)


def positional_encoding(x, y, side):
    """A circle with a sine wave feeding the (+) node at (x, y)."""
    cx = x + side * 1.1
    pic.circle((cx, y), 0.25, fill="white")
    pic.function(lambda t: y + 0.11 * math.sin((t - cx) / 0.17 * math.pi), (cx - 0.17, cx + 0.17),
                 samples=40, line_width=0.8)
    pic.node("Positional\nEncoding", at=(cx + side * 0.35, y), anchor="west" if side > 0 else "east",
             font=r"\footnotesize")
    pic.draw((cx - side * 0.25, y), (x + side * 0.25, y), arrow="->")


def nx_group(nodes, side, x):
    """Grey box around a stack of layers, labelled N x on the outside."""
    box = pic.fit(*nodes, style="group=Gray", padding=7)
    pic.node(r"$N\times$", at=box.east if side > 0 else box.west,
             anchor="west" if side > 0 else "east", font=r"\large", xshift=0.15 * side)


# ------------------------------------------------------------------ encoder
pic.node("Inputs", at=(ENC, -0.75), font=r"\small")
e_emb = block("Input\nEmbedding", ENC, 0.45, "Pink", 0.9)
e_add = pic.node("$+$", at=(ENC, 1.55), style="op")
positional_encoding(ENC, 1.55, -1)
pic.draw((ENC, -0.5), e_emb.south, arrow="->")
pic.draw(e_emb.north, e_add.south, arrow="->")

e_mha = block("Multi-Head\nAttention", ENC, 3.2, "Orange", 0.95)
e_an1 = block("Add & Norm", ENC, 4.25, "Yellow", 0.55)
e_ff = block("Feed\nForward", ENC, 5.5, "Sky", 0.95)
e_an2 = block("Add & Norm", ENC, 6.55, "Yellow", 0.55)

pic.draw(e_add.north, (ENC, 2.15))
self_attention_inputs(e_mha, ENC, 2.15)
residual(ENC, 2.15, e_an1, -1)
pic.draw(e_mha.north, e_an1.south, arrow="->")
pic.draw(e_an1.north, (ENC, 4.75))
pic.draw((ENC, 4.75), e_ff.south, arrow="->")
residual(ENC, 4.75, e_an2, -1)
pic.draw(e_ff.north, e_an2.south, arrow="->")
nx_group([e_mha, e_an2, tz.Point(ENC - SIDE, 2.15)], -1, ENC)

# ------------------------------------------------------------------ decoder
pic.node("Outputs\n(shifted right)", at=(DEC, -0.85), font=r"\small")
d_emb = block("Output\nEmbedding", DEC, 0.45, "Pink", 0.9)
d_add = pic.node("$+$", at=(DEC, 1.55), style="op")
positional_encoding(DEC, 1.55, +1)
pic.draw((DEC, -0.45), d_emb.south, arrow="->")
pic.draw(d_emb.north, d_add.south, arrow="->")

d_mmha = block("Masked\nMulti-Head\nAttention", DEC, 3.3, "Orange", 1.15)
d_an1 = block("Add & Norm", DEC, 4.45, "Yellow", 0.55)
d_mha = block("Multi-Head\nAttention", DEC, 6.1, "Orange", 0.95)
d_an2 = block("Add & Norm", DEC, 7.15, "Yellow", 0.55)
d_ff = block("Feed\nForward", DEC, 8.4, "Sky", 0.95)
d_an3 = block("Add & Norm", DEC, 9.45, "Yellow", 0.55)

pic.draw(d_add.north, (DEC, 2.15))
self_attention_inputs(d_mmha, DEC, 2.15)
residual(DEC, 2.15, d_an1, +1)
pic.draw(d_mmha.north, d_an1.south, arrow="->")

# encoder-decoder attention: queries from below, keys and values from the encoder
branch = 4.95
pic.draw(d_an1.north, (DEC, branch))
residual(DEC, branch, d_an2, +1)
pic.draw((DEC, branch), (DEC + SPREAD, branch), tz.xy((DEC + SPREAD, 0), d_mha.south),
         arrow="->", rounded_corners=3)
kv = 5.35
pic.draw(e_an2.north, (ENC, 7.55), (DEC - 1.9, 7.55), (DEC - 1.9, kv), (DEC - SPREAD, kv),
         tz.xy((DEC - SPREAD, 0), d_mha.south), arrow="->", rounded_corners=5)
pic.dot((DEC - SPREAD, kv), size=3.5)
pic.draw((DEC - SPREAD, kv), (DEC, kv), tz.xy((DEC, 0), d_mha.south), arrow="->", rounded_corners=3)

pic.draw(d_mha.north, d_an2.south, arrow="->")
pic.draw(d_an2.north, (DEC, 7.65))
pic.draw((DEC, 7.65), d_ff.south, arrow="->")
residual(DEC, 7.65, d_an3, +1)
pic.draw(d_ff.north, d_an3.south, arrow="->")
nx_group([d_mmha, d_an3, tz.Point(DEC + SIDE, 2.15)], +1, DEC)

# ------------------------------------------------------------------ output head
linear = block("Linear", DEC, 10.85, "Purple", 0.6)
softmax = block("Softmax", DEC, 11.85, "Green", 0.6)
pic.draw(d_an3.north, linear.south, arrow="->")
pic.draw(linear.north, softmax.south, arrow="->")
probs = pic.node("Output\nProbabilities", at=(DEC, 12.95), font=r"\small")
pic.draw(softmax.north, probs.south, arrow="->")

if __name__ == "__main__":
    pic.render("transformer.pdf")
