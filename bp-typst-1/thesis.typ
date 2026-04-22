#import "template.typ": *
#show: template

#page(margin: 0cm, image("title-page.svg"))
#page(margin: 0cm, image("assignment.svg"))

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

== Declaration

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

== Acknowledgements

#TODO[Thank Herout]

#pagebreak()
#set page(numbering: "1", number-align: center)
#counter(page).update(1)

// enable heading numbers
#set heading(numbering: "1.1")
#outline(depth: 2, target: heading.where(numbering: "1.1"))

// ===== START OF MAIN CONTENT =====

= Introduction

#lorem(500)

= Plan

*Explain malleable glyphs:*
- what they are
- why research on them is important
- technical details on how they work (how can we encode a floating-point scalar value into a 1x1 in image?)
- show some examples of glyphs
- maybe touch on Q-methodology? Or just def mention Herout's article

*Explain machine learning:*
- what machine learning is
- deep neural networks
- convolutional neural networks
- structure of neural network
 - maybe put like a diagram here of my NN (or maybe keep that later for the implementation?... or maybe just put a diagram of a generic CNN?)
- explain regression
- explain binned regression


*My implementation:*
- Python project using UV as build system
 - idk if it's relevant but i can explain why it's better than pip because dependencies are declared in a single file and UV makes sure the environment mirrors the declaration at all times (idempotency)
- organization of the project
 - notebooks are separate from code
 - working code is extracted into functions that live outside notebooks
- structure of datasets
 - mention splits, manifest...
- usage of ClearML for reporting and analysis
- I have a "base" experiment that's used as the template for other experiments... It has been perfected
- I use Papermill to inject different parameters, and run it on Sophie using `tmux` so I can sleep while Sophie is workin' hard
- maybe I can explain that I struggled with fast data loading
 - Extracting all the PNGs into RAM in an decoded format takes up a lot of memory, so datasets are limited to a smaller size
 - In the end, the decoding doesn't take up too much time, it's okay to load the images compressed and decode on-the-fly

*What i've found:*
- NNs are surprisingly good
 - even smaller networks are actually able to memorize the backgrounds from glyphs and thus generalize worse
 - so we need various augmentations and random backgrounds behind the glyphs to make it a challenge for the NN
- with the "template" experiment, the predictions lie on the y=x line perfectly
 - major hiccup was the "tail"... That one still pops up every once in a while, but augmentations help quite a bit
- augmentations:
 - AlbumentationsX, explain which augmentations have been done
- Explain binned regression
 - bins, logits, ...
- There was also regression, but it had a huge "tail" that I didn't manage to remove
- I played around with the learning rate, tried different schedulers
- Explain that the NN has to be better than the worst-case, which is $"error" = 25 "units of x"$
