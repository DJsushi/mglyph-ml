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

// ===== START OF MAIN CONTENT =====

= Introduction

This text serves as an example content of this template and as a recap of the most important information from regulations, it also provides additional useful information, that you will need when you write a technical report for your academic work. Check out appendix A before you use this template as it contains vital information on how to use it.

Even though some students only need to know and comply with the official formal requirements stated in regulations as well as typographical principles to write a good diploma thesis (a bachelor's thesis is a diploma thesis too -- you get a diploma for it), it is never a bad idea to familiarize yourself with some of the well-established procedures for writing a technical text and make things easier for yourself. Some supervisors had prepared breakdowns of proven procedures that have led to tens of successfully presented academic works. A selection of the most interesting procedures available to the authors of this work at the time of writing can be found in the chapters below. If your supervisor has their own web page with recommended procedures, you can skip these chapters and follow their instructions instead. If that is not the case, you should read the respective chapters prior to consulting your supervisor about the structure and contents of your academic work.

Diploma thesis is an extensive work and the technical report should reflect it. It is not easy for everyone to sit down and simply write it. You need to know where to begin and how to progress. One of many viable approaches is to start with keywords and abstract, this helps you establish what the most important part of your work is. More on that in chapter 2.

Once the abstract is finished, you can start with the text of the technical report. The first thing you should do is create a structure for your work, that you'll later fill with text. Chapter 3 provides basic information and hints on writing a technical text, that can help you avoid mistakes beginners make, create chapter titles, and figure out what the approximate length of individual chapters should be. The chapter concludes with an approach that should make writing a thesis much easier.

Diploma theses in the field of information technology have a specific structure. The introduction is followed by a chapter or chapters dealing with the summary of the current state. The next chapter should evaluate the current state and provide a solution, that will be implemented and tested. The conclusion should contain evaluated results and ideas for future development. Even though the chapter titles and their length may differ from other theses, you can always find chapters that correspond with this structure. Chapter 4 deals with the contents of chapters that commonly occur in diploma theses in the field of information technology. Most students will only use a subset of all the described chapters as not everything will be relevant to their thesis. The descriptions and hints provided help students with the inner structure and the contents of chapters as well as decide whether or not they should even include a given chapter.

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

= The Malleable Glyph

This chapter introduces the _malleable glyph_, explains its purpose, the current state of research surrounding it, #TODO[...]

== What Is a Malleable Glyph?

The concept of a malleable glyph was first introduced by #cite(<BibMalleableGlyph>, form: "prose"). A malleable glyph is an image of a fixed size (e.g. $1 times 1$ inch), which carries encoded information in its appearance. For the purpose of our research, we gave this encoded information a specific format: a single scalar value we call $x$, with $x in [0.0, 100.0]$. This scalar value is then used to dictate the appearance of the malleable glyph itself. To better understand, let's look at some examples.

#figure(
  image("img/glyphs/green-triangle.png", width: 80%),
  caption: [A malleable glyph showing a green triangle that encodes the value of $x$ in the size of the glyph. A small value of $x$ renders a small triangle, and as $x -> 100.0$, the triangle fills the entire glyph.],
)

#lorem(100)

== The Challenge



Quickly just introduce the challenge.
Illiteracy rule... Žiadne "řády".

==

==

Definition and Origin: Introduce MGlyphs as a visualization technique originating from UX research (card sorting) and the "Illiteracy Rule"—the idea that observers should perceive "how much" rather than "how many".
Encoding Mechanism: Describe how a scalar value is deterministically mapped into visual shape properties.
The Challenge: Reference the public challenge and Adam Herout’s work to anchor your thesis in the current scientific landscape.
Visual Examples: Include a figure showing how a glyph changes visually as x progresses from 0 to 100.

#pagebreak()
#bibliography("references.bib")
