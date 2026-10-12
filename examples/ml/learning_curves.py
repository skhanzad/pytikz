"""Training and validation loss curves with early stopping."""

import numpy as np

import pytikz as tz

rng = np.random.default_rng(1)
epochs = np.arange(1, 101)
train = 0.08 + 1.6 * np.exp(-epochs / 14) + 0.012 * rng.normal(size=epochs.size)
val = 0.25 + 1.5 * np.exp(-epochs / 13) + 0.0022 * np.maximum(0, epochs - 30) ** 1.25 \
    + 0.02 * rng.normal(size=epochs.size)
smooth_val = np.convolve(val, np.ones(5) / 5, mode="same")
best = int(np.argmin(smooth_val[5:-5])) + 5

pic = tz.Picture()
ax = pic.add(tz.Axes(width=9, height=5.2, xlim=(0, 100), ylim=(0, 1.9), grid=True, xlabel="epoch",
                     ylabel="loss", title="Learning curves"))
ax.fill_between([epochs[best], 100], 0, 1.9, color="Red", opacity=0.07)
ax.plot(epochs, train, color="Blue", line_width=1.4, label="training")
ax.plot(epochs, val, color="Orange", line_width=1.4, label="validation")
ax.vline(epochs[best], color="Ink!60", line_width=0.9)
ax.scatter([epochs[best]], [val[best]], color="Orange", size=3.2, edgecolor="white")
ax.annotate("early stopping", xy=(epochs[best], val[best]), xytext=(epochs[best] + 16, 1.0),
            font=r"\small", anchor="west")
ax.text(88, 1.75, "overfitting", font=r"\small", text_color="Red!80!black")
ax.legend("upper center")

if __name__ == "__main__":
    pic.render("learning_curves.pdf")
