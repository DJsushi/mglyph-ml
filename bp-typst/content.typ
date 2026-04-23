#import "template.typ": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node, shapes

= Introduction

#TODO[intro]

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

The concept of a malleable glyph was first introduced by #cite(<BibMalleableGlyph>, form: "prose"). They defined the malleable glyph as an image of a fixed size (e.g. $1 times 1$ inch), which carries encoded information in its appearance. For the purpose of their research, they gave this encoded information a specific format: a single scalar value they call $x$, with $x in [0.0, 100.0]$. This scalar value is then used to dictate the appearance of the malleable glyph itself. To better understand how this works, let's look at some examples.

#figure(
  image("fig/glyphs/green-triangle.png"),
  caption: [A malleable glyph showing a green triangle that encodes the value of $x$ in the size of the glyph. A small value of $x$ renders a small triangle, and as $x -> 100.0$, the triangle fills the entire glyph.],
) <fig-green-triangle>

@fig-green-triangle shows how a malleable glyph might look like. The triangle's size is linearly mapped to the value of $x$, interpolating from infinitesimal to edge-to-edge. The author of the glyph can choose to vary any aspect of the glyph in accordance with the parameter $x$. The size, color, shape, roundedness, spikiness, contrast, or even fractal structure #footnote[https://github.com/adamherout/mglyph] just to name a few. Another example of what's possible can be seen in @fig-fractal-tree.

#figure(
  image("fig/glyphs/fractal-tree.png"),
  caption: [A fractal tree with a depth of 10, that varies the branching angle based on $x$, as],
) <fig-fractal-tree>

== The Challenge

#TODO[Quickly just introduce the challenge.]

Having so many different options with

== The Illiteracy Rule

While creating the best glyph possible, the author of the malleable glyph might be tempted to simply write the value of $x$ onto the glyph. This, however, is against the _illiteracy rule_, which states:

#quote(attribution: <BibMalleableGlyph>, block: true)[
  We make it a rule for malleable glyphs that they must not use writing (both letters and numerals), nor must they use numerals or values arranged in individual orders even without using writing.
]

In other words, glyphs containing so-called _orders_ are *not* considered malleable glyphs. Examples of such glyphs are shown in @fig-bad-glyphs.

#figure(
  square(TODO[bad glyphs image]),
  caption: [An example of uneligible glyphs. The clock has multiple hands (orders), and thus, not a malleable glyph. The textual glyph has orders in the text itself -- the decimal numerical system, like any other numerical system, inherently contains orders.],
) <fig-bad-glyphs>

== The `mglyph` Python Library

The manual creation of malleable glyphs might prove challenging, as creating and packaging thousands of images into a very specific packaged format is manual labor ???. That is one of the main reasons why the `mglyph` Python library was created; to abstract away the complexities of creating and distributing the glyphs in the correct format from the artist, and to make the whole process of creating glyphs as frictionless as possible.

The library was published on PyPi #footnote[https://pypi.org/project/mglyph/] under the name `mglyph`. It is open-source and the source code is available on GitHub #footnote[https://github.com/adamherout/mglyph]. The library is essentially a framework to facilitate the job of designing, producing, and distributing malleable glyphs. Its core is written in the Python programming language. The reason why Python was used is that it is a widely-adopted popular language, and many developers already know how to work with it. Its syntax offers the flexibility that was needed for such a library.

When one wants to create their own malleable glyph, it's best to first look at the tutorial that's linked in the repository's `README.md` file. The tutorial showcases what's possible, illustrating many interesting ideas of what's possible. After reading the tutorial, one can tweak some parameters to see how the glyphs change when the parameters are tweaked.

At the core of any glyph, there's a function called a `Drawer`. This function might look something similar to this:

```py
import mglyph as mg

def simple_circle(x: float, canvas: mg.Canvas) -> None:
    radius = mg.lerp(x, 0, canvas.xsize / 2)
    canvas.circle(center=canvas.center, radius=radius, color='red')
```

This function has a signature of type `mg.Drawer`. A `mg.Drawer` is a function that accepts 2 parameters: a `float` and a `mg.Canvas`. `x` is passed into the function by value, while the `mg.Canvas` is passed by reference. The designer of a glyph may use these two variables that are available inside the function's context to draw something onto the canvas. In this case, the radius is calculated by linearly interpolating the value of $x in [0.0, 100.0]$ onto the interval $[0.0, "canvas.xsize" / 2]$. The canvas has a size of 2.0, however, it's best to always specify the values in a _relative_ manner. That way, the intentions of the author of the glyph are clear and everyone reading the source code of the glyph can see what the author of the glyph had in mind while creating the glyph. Like in this case, the circle's radius goes from 0.0 until half of the canvas' horizontal size (`xsize`), thus, spanning from edge to edge of the canvas.

Now, that the `mg.Drawer` (```py simple_circle()```) is defined, we need a way to see what the glyph actually looks like with different values of $x$. Thankfully, the creators of the `mglyph` library provided multiple ways to visualize the glyphs. One of the most important functions is the ```py show()``` function. It allows us to see what the glyph looks like with different values of $x$ passed down to the function. It accepts many parameters, but here are the two most important ones:

```py
def show(
    drawer: mg.Drawer | list[mg.Drawer] | list[list[mg.Drawer]],
    x: int | float | list[float] | list[int] | list[list[float]] |
       list[list[int]] = [5, 25, 50, 75, 95],
)
```

The exact mechanisms behind ```py show()``` for the purposes of this work are not important, however, it is important to understand how the `mg.Drawer` is used by the `mglyph` library.

#let dark-blue = rgb("#0074d8")
#let light-blue = rgb("#e5f1fb")
#let light-red = rgb("#f8cfcf")
#let dark-red = rgb("#ed7575")

#figure(
  image("fig/diagrams/show-function.drawio.svg"),
  caption: [Diagram of the inner workings of the ```py show()``` function from the `mglyph` library. For every $x$ that gets passed into ```py show()```, it instantiates a new `mg.Canvas`, and passes it down to the ```py drawer()``` so that it can draw the glyph based on the passed argument `x`.],
)

= #TODO[The Foundations of Machine Learning Visual something something...]
