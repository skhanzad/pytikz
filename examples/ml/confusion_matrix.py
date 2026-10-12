"""A confusion matrix with counts, row-normalised colors and per-class recall."""

import numpy as np

import pytikz as tz

classes = ["cat", "dog", "bird", "fish", "frog"]
counts = np.array([
    [52, 6, 1, 0, 1],
    [8, 47, 2, 0, 3],
    [2, 1, 55, 1, 1],
    [0, 0, 2, 56, 2],
    [1, 4, 0, 3, 52],
])
recall = counts / counts.sum(axis=1, keepdims=True)

pic = tz.Picture()
hm = pic.add(tz.Heatmap(recall, cell=1.0, cmap="blues", vmin=0, vmax=1, row_labels=classes,
                        col_labels=classes, col_labels_position="bottom", values=False, colorbar=True,
                        colorbar_label="fraction of true class", xlabel="predicted label",
                        ylabel="true label", title="Confusion matrix"))

# counts (and the fraction on the diagonal) inside the cells
for i in range(len(classes)):
    for j in range(len(classes)):
        text_color = "white" if recall[i, j] > 0.55 else "Ink"
        text = f"\\textbf{{{counts[i, j]}}}" if i == j else str(counts[i, j])
        pic.node(text, at=hm.cell(i, j), font=r"\small", text_color=text_color)
        if i == j:
            pic.node(f"{100 * recall[i, j]:.0f}\\%", at=hm.cell(i, j) + (0, -0.27), font=r"\tiny",
                     text_color=text_color)

accuracy = np.trace(counts) / counts.sum()
pic.node(f"accuracy {100 * accuracy:.1f}\\%", at=(2.5, -6.6), font=r"\small", text_color="Ink!70")

if __name__ == "__main__":
    pic.render("confusion_matrix.pdf")
