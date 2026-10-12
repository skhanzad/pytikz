"""The smallest useful pytikz script.

Run it with ``python hello.py`` (writes hello.pdf) or ``pytikz render hello.py``,
and look at the generated TikZ with ``pytikz tex hello.py``.
"""

import pytikz as tz

pic = tz.Picture()
data = pic.node("Data", style="block=Green")
model = pic.node("Model", right_of=data, distance=1.2, style="block")
pred = pic.node("Prediction", right_of=model, distance=1.2, style="block=Orange")
pic.arrow(data, model)
pic.arrow(model, pred)
pic.arrow(pred.south, model.south, bend=40, label="loss", dashed=True, color="Red")

if __name__ == "__main__":
    pic.save("hello.tex")      # standalone LaTeX document
    pic.render("hello.pdf")    # compiled figure (needs pdflatex)
