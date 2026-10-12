"""Scaled dot-product attention: the computation graph and the attention
weights of one head over a short sentence."""

import numpy as np

import pytikz as tz

pic = tz.Picture()

# ---------------------------------------------------------------- left: the operation
def op(text, y, color):
    return pic.node(text, at=(0, y), style=f"block={color}", minimum_width=2.6, minimum_height=0.65,
                    font=r"\small")


q = pic.node("$Q$", at=(-0.8, -0.2), style="block=Purple", minimum_width=0.7)
k = pic.node("$K$", at=(0.0, -0.2), style="block=Purple", minimum_width=0.7)
v = pic.node("$V$", at=(1.6, -0.2), style="block=Purple", minimum_width=0.7)
matmul1 = op("MatMul", 1.1, "Blue")
scale = op("Scale", 2.15, "Yellow")
mask = op("Mask (opt.)", 3.2, "Pink")
softmax = op("SoftMax", 4.25, "Green")
matmul2 = pic.node("MatMul", at=(0.8, 5.5), style="block=Blue", minimum_width=2.6, minimum_height=0.65,
                   font=r"\small")

pic.arrow(q, tz.xy(q, matmul1.south))
pic.arrow(k, tz.xy(k, matmul1.south))
for a, b in ((matmul1, scale), (scale, mask), (mask, softmax)):
    pic.arrow(a, b)
pic.arrow(softmax.north, tz.xy(softmax, matmul2.south))
pic.arrow(v, tz.xy(v, matmul2.south))
pic.arrow(matmul2, (0.8, 6.5))
pic.node(r"$\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$", at=(0.8, 6.85),
         font=r"\small")

# ---------------------------------------------------------------- right: the weights
tokens = ["the", "cat", "sat", "on", "the", "mat"]
rng = np.random.default_rng(11)
d = 8
E = rng.normal(size=(len(tokens), d))
E[4] = E[0] + 0.1 * rng.normal(size=d)          # both "the" are alike
E[5] = 0.6 * E[1] + 0.4 * rng.normal(size=d)    # "mat" relates to "cat"
scores = E @ E.T / np.sqrt(d)
weights = np.exp(scores - scores.max(1, keepdims=True))
weights /= weights.sum(1, keepdims=True)

hm = pic.add(tz.Heatmap(weights, at=(4.2, 5.6), cell=0.75, cmap="purples", values=True,
                        row_labels=tokens, col_labels=tokens, col_label_angle=45, colorbar=True,
                        title="Attention weights", xlabel="keys", ylabel="queries"))
hm.outline(5, 1, color="Orange", line_width=1.8)

if __name__ == "__main__":
    pic.render("attention.pdf")
