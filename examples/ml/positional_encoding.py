"""Sinusoidal positional encodings: PE(pos, 2i) = sin(pos / 10000^(2i/d)),
PE(pos, 2i+1) = cos(...), shown as an image plus three of its columns."""

import numpy as np

import pytikz as tz

positions, d_model = 48, 64
pos = np.arange(positions)[:, None]
i = np.arange(d_model)[None, :]
angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))

pic = tz.Picture()
ax = pic.add(tz.Axes(width=8.5, height=6.4, frame="box", xlabel="embedding dimension $i$",
                     ylabel="position", title="Positional encoding",
                     xlim=(-0.5, d_model - 0.5), ylim=(-0.5, positions - 0.5),
                     xticks=[0, 16, 32, 48, 63], yticks=[0, 10, 20, 30, 40]))
ax.imshow(pe, cmap="rdbu", vmin=-1, vmax=1)
ax.colorbar(ticks=[-1, 0, 1])

lines = pic.add(tz.Axes(at=(11.3, 0), width=6, height=6.4, xlim=(0, positions - 1), ylim=(-1.15, 1.6),
                        frame="open", grid=True, xlabel="position", title="Three dimensions"))
for dim, color in ((4, "Blue"), (12, "Orange"), (24, "Green")):
    lines.plot(np.arange(positions), pe[:, dim], color=color, label=f"$i = {dim}$")
lines.legend("upper right", columns=3)

if __name__ == "__main__":
    pic.render("positional_encoding.pdf")
