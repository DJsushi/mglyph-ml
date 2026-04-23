#import "template.typ": *
#show: template

#page(margin: 0cm, image("res/title-page.svg"))
#page(margin: 0cm, image("res/assignment.svg"))

#let thesis = (
  title: [Exploring the Possibilities and Limitations of Computer Vision in Conjunction with Malleable Glyphs],
  first-name: [Martin],
  last-name: [Gaens],
)


#abstract-keywords(
  abstract-en: [
    This bachelor's thesis explores the potentials and limits of computer vision regarding an evaluation of Malleable Glyphs, a visualization technique that encodes a scalar value by the shape of graphical glyphs. This work includes background research in computer vision and machine learning as well as the design and implementation of a dataset generation pipeline that can iteratively produce and refine glyph samples in a systematic way. We conducted several experiments on the resulting datasets to explore whether a neural model can retrieve the encoded value from rendered glyph images, and how the outcome varies with different glyph constructions, sampling strategies, and visual variations. #TODO[RESULTS... did we find out something? I guess the findings were surprising in a way that the NN was able to see the glyphs even with a distorted image.] The main contribution is a reusable workflow that can be used to generate new datasets, glyphs, and controlled experiments to further explore what CNNs are able to learn.
  ],
  abstract-sk: [
    #TODO[Do tohoto odstavce bude zapsán výtah (abstrakt) práce v českém (slovenském) jazyce.]
  ],
  keywords-en: [
    Machine Learning, Malleable Glyph, Visualization, Visual Comparison, Graphical Design, Quantity Visualization, Dataset Creation, Regression, Binned Regression, Experiments
  ],
  keywords-sk: [
    #TODO[Sem budou zapsána jednotlivá klíčová slova v českém (slovenském) jazyce, oddělená čárkami.]
  ],
  reference: [
    #upper(thesis.last-name), #text(thesis.first-name). #text(thesis.title, style: "italic"). Brno, 2026. Bachelor's thesis. Brno University of Technology, Faculty of Information Technology. Supervisor prof. Ing. Adam Herout, Ph.D.
  ],
)

#pagebreak()

#text(size: 10pt, title(thesis.title))

#v(1em)

#heading(outlined: false, level: 2)[Declaration]

I hereby declare that I have prepared this bachelor's thesis independently under the supervision of prof. Ing. Adam Herout, Ph.D. #TODO[Additional information]?. I have listed all literary sources, publications, and other references, which I have used during the preparation of the thesis. #TODO[Copilot]?

#v(1em)

#align(
  right,
  stack(
    spacing: 2mm,
    raw("." * 20),
    [Martin Gaens],
    [April 18, 2026],
  ),
)

#v(4em)

#heading(outlined: false, level: 2)[Acknowledgements]

#TODO[Thank Herout]

#pagebreak()
#set page(numbering: "1", number-align: center)
#counter(page).update(1)

// enable heading numbers
#set heading(numbering: "1.1")
#outline(depth: 2)

#include "content.typ"

#pagebreak()
#bibliography("references.bib", style: "res/iso690-numeric-en-fitvut.csl")
