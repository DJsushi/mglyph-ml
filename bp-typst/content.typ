#import "template.typ": *
#import "lib.typ": aligned-terms, init-raw-annot, number-line, raw-annot
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node, shapes
#import "@preview/cheq:0.3.0": checklist
#import "@preview/cetz:0.5.0": canvas, draw

#show: checklist
#show: init-raw-annot

= Introduction

A Malleable Glyph is a small graphical design that usually fits within a 1#(sym.times)1 inch square. This image encodes a scalar we call $x$ and its value can range from 0.0 (including) up to 100.0 (including). The way this value $x$ is encoded in the glyph is by varying some visual aspect of the glyph with the value of $x$, for example the size, color, roundness, zoom, a combination of any of these, or any other visual aspects. The exact way this scalar value is encoded in the glyph is up to the author of the glyph. The essence of malleable glyph is to carry as much information as possible about the exact value of $x$ inside the static area of the glyph, making even a change of 0.01 $x$ units noticeable.

These glyphs can then be generated in batch, generating tens of thousands of glyphs in the range $x in [0.0, 100.0]$. These can afterwards be packaged into a dataset. These datasets are then used to train neural network models to predict the value of $x$.

The purpose of this thesis was to create a set of tools (framework also) that facilitate the design and implementation of experiments on neural networks, and then, to create a few experiments, run them, and discover some new information about the neural networks. These experiments study how the neural networks are able to learn the glyphs. They can study if the neural network actually understands what in the glyph makes the $x$ ..

The structure of this thesis is a little unconventionnal, but it's like that on purpose after discussing with my supervisor. Instead of following the background, design and implementation structure, I first explain the malleable glyph, then I explain some machine learning, after that, I dive into the intricacies of a good ML development environment and infrastructure, after that, I explain how to use my framework to create high quality datasets for the purpose of experimenting with glyphs. Then, I introduce some of the experiments I have designed and run and explain how to make them yourself, and lastly, I conclude and explain my ideas that I didn't have time to implement but would love to if I had more time on the bachelor's thesis.

In @chapter-the-malleable-glyph, I briefly explain malleable glyphs in more detail, explaining the how and why of its existence, how it's linked to this thing called q-methodology, and I explain why some glyphs are not considered proper malleable glyphs and some are by introducing something called the "illiteracy rule". Afterwards, in @chapter-ml-fundamentals, I dive into the details of machine learning, specifically, what I've learned during the creation of this thesis about machine learning, I explain a special type of regression called "binned regression" and how I am using it for experiments. I also explain the development environment that I set up, and why I did certain things in the way I did. @chapter-creating-datasets introduces a comprehensive guide on how to create your own datasets, how to export them properly, different types of datasets, and gives some advice on how to create good quality datasets that are reusable across multiple experiments. Afterwards, we dive right into the experiments in @chapter-experiments, where I explain a couple of experiments I have done.


#TODO[
  1. The Context (The Hook)
  Start with a brief, non-technical explanation of the problem. Explain that Malleable Glyphs are small graphical designs (1 #sym.times 1 inch) used to encode a scalar value x∈.

  Mention the "Illiteracy Rule": Explain the core principle that these glyphs must trigger a "how much?" intuition rather than a "how many?" (counting) process.
  Motivation: Briefly note that these originated in UX research (Q-methodology) to replace cards with numbers, which people sort "digit-by-digit" instead of by feeling.

  2. The Research Problem
  Transition into the computer vision aspect. State the core question of your thesis: Can a machine "see" and decode these glyphs as accurately as (or better than) a human?.

  Explain that while humans compare glyphs subjectively, your work explores the possibilities and limitations of Convolutional Neural Networks (CNNs) in predicting the exact parameter x from rendered images.

  3. Thesis Goals (The "What I Did" section)
  Explicitly state your objectives. Do not use the exact words from the assignment; use your own. Your goals included:

  Establishing a robust pipeline: Creating a reusable workflow for generating datasets and running reproducible experiments using tools like the uv build system, Papermill, and ClearML.
  Perfecting the "Straight Line": Achieving high-precision scalar prediction (y=x) through the implementation of Binned Regression.
  Scientific Exploration: Testing the boundaries of the network's understanding through "gap" experiments (interpolation) and "baby" networks to prevent background memorisation.

  4. The "Manual for Franta" (Chapter Overview)
  Conclude the introduction with a walkthrough of the thesis. This acts as a manual for your successor so they know where to find specific information:

  Chapter 2: Theoretical background of Malleable Glyphs and the Illiteracy Rule.
  Chapter 3: The technical "laboratory"—explaining binned regression and the MLOps infrastructure (uv, Sophie server).
  Chapter 4: Data engineering—how the .mglyph format works and how to use the mglyph library to build new datasets.
  Chapter 5: The heart of the thesis—a series of controlled experiments answering specific hypotheses about generalization and network capacity.
  Chapter 6: Final results, a summary of contributions, and concrete ideas for future work.
]

= The Malleable Glyph <chapter-the-malleable-glyph>

This chapter introduces the _malleable glyph_, explains its purpose, the current state of research surrounding it, #TODO[let ai write this chapter summary]

== What Is a Malleable Glyph?

The concept of a malleable glyph was first introduced by #cite(<BibMalleableGlyph>, form: "prose"). They defined the malleable glyph as an image of a fixed size (e.g. $1 times 1$ inch), which carries encoded information in its appearance. For the purpose of their research, they gave this information a specific format: a single scalar value they call $x$, with $x in [0.0, 100.0]$. This scalar value is then used to dictate the appearance of the malleable glyph itself. To better understand how this works, let's look at some examples.

#figure(
  image("fig/glyphs/green-triangle.png"),
  caption: [A malleable glyph showing a green triangle that encodes the value of $x$ in the size of the glyph. A small value of $x$ renders a small triangle, and as $x -> 100.0$, the triangle fills the entire glyph.],
) <fig-green-triangle>

In @fig-green-triangle, we can see how a malleable glyph might look like. The triangle's size is linearly mapped to the value of $x$, interpolating from infinitesimal to edge-to-edge. The author of the glyph can choose to vary any aspect of the glyph in accordance with the parameter $x$. The size, color, shape, roundedness, spikiness, contrast, or even fractal structure #footnote[https://github.com/adamherout/mglyph] just to name a few. So, @fig-fractal-tree shows a more interesting example of a fractal tree opening up as $x$ approaches 100.

#figure(
  image("fig/glyphs/fractal-tree.png"),
  caption: [A fractal tree with a depth of 10, that varies the branching angle based on $x$, as],
) <fig-fractal-tree>

== Connection to Q-methodology

Q methodology is a research technique for studying subjectivity. It's a standard method developed to study peoples' opinions. Q-methodology is executed by a process called Q-sorting. The basic idea of Q-sorting is that a subject is given a deck of cards. The number of cards can vary from a few up to a hundred or even more. These cards contain various phrases, sentences, utterances, like "JavaScript was designed in 10 days and you can tell", or "CSS is not a real programming language, it's just vibes with semicolons". Subsequently, the subject is instructed to sort these cards on a scale. This scale can be as arbitrary as the researchers need it to be:

- "completely disagree" #sym.arrow.l.r.long "completely agree",
- "this is clean code" #sym.arrow.l.r.long "this is pure spaghettification",
- "I would merge this PR" #sym.arrow.l.r.long "I would close this PR without comment".

Once several people have done the Q-sorting, their rankings are correlated, and run through factor analysis. This way, the researchers can find patterns in thinking, and people usually cluster natually into groups of people with similar opinions on multiple subjects. This way, the researchers don't create their own categories to group the subjects into, but the subjects' opinions form these groupings on their own #cite(<BibPrimerOnQMethodology>).

Malleable glyphs were born out of Q-methodology research. #cite(<BibMalleableGlyph>, form: "prose") were building a modern digital interface for Q-sorting, but since they were seeking an optimal user experience, it was best to remove the subjectivity of Q-sorting, #quote(attribution: <BibMalleableGlyph>)[that is, to remove subjective statements from the sorted cards and replace them with objective content]. Their first attempt at creating objective cards was writing numerals in text form into the cards. This attempts



The connection is quite direct — malleable glyphs were actually born out of Q methodology research.
The authors were building a modern digital interface for conducting Q sorts, and ran into a practical problem: what do you put on the cards? Real Q studies use subjective statements, but to test the usability of the sorting interface itself (rather than participants' opinions), they needed neutral, content-free cards.
Their first attempt was cards with numbers written as words ("three hundred and fifty-six"). This failed because people sorted numbers the way they learned in school — digit by digit — which is a completely different cognitive process from how you'd sort opinion statements by feeling.
Their second attempt was a simple scaled geometric shape on each card. This was better, but hit a different wall: there just aren't enough visually distinguishable sizes of a simple shape to cover a large deck of cards (dozens to hundreds).
That problem — how do you pack the most distinguishable quantity information into a small fixed card area, without using numbers or text — is exactly what a malleable glyph is designed to solve. The "illiteracy rule" (no numbers, no digit-like ordering) comes directly from this origin: the glyph needs to trigger the intuition "how much?" rather than "how many?", mimicking the feeling of sorting statements by agreement rather than counting.
So malleable glyphs started as a very practical UX problem inside Q methodology research, and grew into a broader visualization challenge of its own.

== The Challenge

prof. Herout introduced with the malleable glyph also a public challenge. He wanted to see a bunch of creative people creating their own glyphs and submitting them. The reason why he wanted this is so that we can find some glyph, which is able to show miniscule differences in $x$ visually. #TODO[cite author of bachelor's thesis of the web of the challenge] created a website where users see two glyphs next to each other, and they are tasked with determining the ordering of the glyphs. They have to say whether the glyph on the left is "larger", "smaller", or "the same" as the glyph on the right. Not in a sense of _size_, but in a sense of the value of $x$ itself. Remember, that glyphs can show the value of $x$ not only in their size, but many other visual aspects (ways).

== The Illiteracy Rule

While creating the best glyph possible, the author of the malleable glyph might be tempted to simply write the value of $x$ onto the glyph. This, however, is against the _illiteracy rule_, which states:

#quote(attribution: <BibMalleableGlyph>, block: true)[
  We make it a rule for malleable glyphs that they must not use writing (both letters and numerals), nor must they use numerals or values arranged in individual orders even without using writing.
]

In other words, glyphs containing so-called _orders_ are *not* considered malleable glyphs. Also, an illiterate person should be able to give 2 distinct glyphs an ordering. Examples of such glyphs are shown in @fig-bad-glyphs.

#figure(
  square(TODO[bad glyphs image]),
  caption: [An example of uneligible glyphs. The clock has multiple hands (orders), and thus, not a malleable glyph. The textual glyph has orders in the text itself -- the decimal numerical system, like any other numerical system, inherently contains orders.],
) <fig-bad-glyphs>

= Machine Learning Fundamentals For Malleable Glyph Regression <chapter-ml-fundamentals>

The whole point of the thesis is to explore what we can do with machine learning computer vision models, and how far we can push them, discovering their limitations and strengths. One really good way of doing so is to use malleable glyphs, because they offer a very controlled way of creating datasets of images, that all have labels, and can be shown to a neural network. Using the technology of malleable glyph, we are able to generate thousands of very tailored training/validation/testing samples to show our neural networks.

This chapter explains some of the basics of machine learning that are relevant to the context of this research and essential for understanding the point of the experiments shown later. Because of the fact that the field of machine learning is so vast, I won't be diving too much into the details of all the topics, I will mostly just try to cover up to the boundary of what's necessary to know to understand the work of this thesis. However, for getting new ideas for new experiments, maybe a more in-depth understanding might be necessary. And thus, I can recommend some resources where you can learn the little intricacies from. A good starting point is the Nature article _Deep learning_ by #cite(<BibDeepLearningLeCun>, form: "prose"). It serves as an overview of the current state of deep learning and acts as a good introduction. For learning the basics, I recommend Andrew Ng's course of Coursera #footnote[https://www.coursera.org/specializations/machine-learning-introduction]. At the moment of writing, it has a free trial for 7 days, and it's more than possible to do it in a week. Most of the course is available on YouTube #footnote[https://www.youtube.com/playlist?list=PLBAGcD3siRDguyYYzhVwZ3tLvOyyG5k6K] for free indefinitely. Another useful resource for learning that I used was the book _Deep Learning_ by #cite(<BibDeepLearningBook>, form: "prose"). It covers the basic math necessary to understand artificial networks, chapter 5 on machine learning basics, and chapter 9 on convolutional neural networks is useful for this thesis.

== The Artificial Neuron

#figure(
  diagram(
    // debug: true,
    spacing: (1.7cm, 0.5cm),
    // cell-size: (0pt, 1em),
    {
      // input and weight nodes
      for x in "012nN" {
        let x-int = if x == "n" { 4 } else {
          if x == "N" { 5 } else {
            int(x) + 1
          }
        }
        node((0, x-int), $x_#(x)$)
        edge("*->")
        node((1, x-int), $w_#(x)$, shape: circle, fill: teal-fill, stroke: teal-stroke, name: label("w" + x))
        edge(<sum>, "->")
      }

      // summation node
      node(
        (rel: (1, 0), to: (1, 2.5)),
        text(size: 3em, $Sigma$),
        shape: circle,
        fill: purple-fill,
        stroke: purple-stroke,
        name: <sum>,
      )

      // activation block
      node(
        (rel: (1, 0), to: <sum>),
        text(baseline: -1pt, size: 2em, $phi$),
        shape: rect,
        fill: purple-fill,
        stroke: purple-stroke,
        name: <act>,
        inset: 1em,
      )
      edge(<act>, <last>, "->")
      node((rel: (1, 0), to: <act>), [$y$ \ Output (activation)], name: <last>)
      edge(<sum>, <act>, "->")
      node((rel: (0pt, 5mm), to: <sum.north>), [Transfer Function])
      node((rel: (0pt, -5mm), to: <act.south>), [Activation Function])
    },
  ),
  caption: [A diagram of an artificial neuron. The inputs $x_n$ are multiplied by their respective weights $w_n$ and then combined at the transfer function (usually just added together). The summed value is sent through the activation function $phi$, which produces the output $y$ called the _activation_ (diagram credit user Funcs on Wikipedia).],
  placement: auto,
) <fig-artificial-neuron>

At the heart of any artificial neural network is a neuron. It's the smallest unit and the main building block of an artificial neural network #cite(<BibDeepLearningBook>). The artificial neuron mimics a biological, living neuron, although simplifies it drastically #cite(<BibMcCulloch1943>). Every artificial neuron has a set of inputs. These inputs can be coming from the data fed into the neural network, or they can come from other neurons, just like human neurons are either connected to receptors, or to other neurons. After receiving an input, the input is multiplied by a scalar value called a _weight_. My intuition behind this name is that it _weighs_ the input, it determines whether a certain input is important to the neuron or not. The weight $w_n$ has the potential to make some inputs very significant to the neuron by multiplying their values by a large positive or negative weight (values far away from 0), but it also has the potential to make an input completely insignificant to the neuron by multiplying the input $x_n$ by values close to 0.

After the weights $w_n$ have been applied to their respective inputs $x_n$, the results of these multiplications are combined into a single scalar value using the _transfer function_. In the diagram shown in @fig-artificial-neuron, the transfer function is illustrated using the summation sign, $sum$. This is because most of the time, this transfer function is indeed just a simple summation.

The last step in the pipeline is the _activation function_. The output of the transfer function becomes the input for the activation function, which is a mathematical function with specific properties. Its purpose is usually to normalize the output of the transfer function, which can be a very large or very small number, since it's a summation of $N$ numbers with arbitrary values. This function ca be chosen depending on the scenario. @fig-activation-functions illustrates some of the most common activation functions used in machine learning.

So, when we put all these steps together, the mathematical formula for calculating the output of an artificial neuron can be expressed like this:

$ y = phi(sum_(n=1)^N x_n w_n) = phi(x_0 w_0 + x_1 w_1 + dots + x_N w_N) $ <math-artificial-neuron>

Please note that in this mathematical formula, the transfer function is a simple summation. Although different transfer functions can be used, in our case and in most of machine learning, a weighted sum like shown in @math-artificial-neuron is used #cite(<BibDeepLearningBook>).

#figure(
  image("fig/graphs/activations.svg", width: 100%),
  caption: [The four most common activation functions. _Identity_ is basically a skip of the activation function, it returns the same thing that gets passed in. _Sigmoid_ has an S-shape that is used to constrain the output of the neuron between 0 and 1, while making a smooth transition between the two values as the input grows from 0 to #sym.infinity. _Binary step_ outputs 0 when $x < 0$, and 1 when $x >= 1$, making a not so smooth transition #TODO[rephrase]. _ReLU_ stands for _rectified linear unit_. It's 0 for $x <= 0$ and $x$ for $x > 0$.],
  placement: auto,
) <fig-activation-functions>

== The Basics of Artificial Neural Networks

#let nnlayer(l) = $L^((#l))$
#let nnact(sub, l) = $a_(#sub)^((#l))$
#let nnweight(sub, l) = $w_#sub^((#l))$
#let nnbias(i, l) = $b_#i^((#l))$
#let nninput(i: none) = $x_#i$
#let nnoutput(i: none) = $hat(y)_#i$
#let nnweightm(l) = $W^((#l))$
#let nnpreact(sub, l) = $z_#sub^((l))$

When we wire multiple of these artificial neurons together in a strategic manner, we are able to create more complex structures that are "smarter" than a single neuron. This newly created structure is called an _artificial neural network_, or ANN for short.

=== Neurons as The Building Blocks of Neural Networks <section-neurons>

#figure(
  diagram(
    // debug: true,
    spacing: (1.2cm, 16pt),
    {
      let neuron(pos, layer, name: none, fake: false, content: []) = {
        let color = (
          (layer == 1, purple-fill, purple-stroke),
          (layer == 2, teal-fill, teal-stroke),
          (layer == 3, green-fill, green-stroke),
        )
          .find(t => t.at(0))
          .slice(1, 3)
        node(
          pos,
          content,
          shape: circle,
          fill: color.at(0),
          stroke: stroke(paint: color.at(1), dash: if fake { "dashed" } else { "solid" }),
          radius: 16pt,
          name: name,
        )
      }

      let layer(number, content) = {
        node(
          enclose: ((number, -2), (number, 6)),
          stroke: stroke(paint: gray, dash: "dashed"),
          fill: gray.lighten(80%),
          corner-radius: 12pt,
          shape: rect,
          layer: -1,
          width: 2cm,
          name: label("l" + str(number)),
        )
        node((rel: (0pt, 12pt), to: label("l" + str(number) + ".north")), content)
      }

      // inputs (X) and input layer (1)
      for row in range(1, 6) {
        node((0, row), nninput(i: row), shape: circle, inset: 4pt, name: label("x_" + str(row)))
        neuron((1, row), 1, fake: true, name: label("n1_" + str(row)))
        edge(label("x_" + str(row)), label("n1_" + str(row)), "*->")
      }

      // Hidden layer (2)
      for row in range(1, 7) {
        neuron((2, row), 2, name: label("n2_" + str(row)))
      }

      // output layer and outputs
      for row in range(3) {
        neuron((3, row + 2), 3, name: label("n3_" + str(row)))
        node((4, row + 2), nnoutput(i: row), name: label("y_" + str(row)))
        edge(label("n3_" + str(row)), label("y_" + str(row)), "->")
      }

      for i in range(6) {
        for j in range(7) {
          edge(label("n1_" + str(i)), label("n2_" + str(j)), "->")
        }
      }

      for i in range(7) {
        for j in range(3) {
          edge(label("n2_" + str(i)), label("n3_" + str(j)), "->")
        }
      }

      neuron((1, 0), 1, name: <n1_0>, fake: true, content: [1])
      neuron((2, 0), 2, name: <n2_0>, fake: true, content: [1])

      node((rel: (-1.0cm, -1.4cm), to: <n2_5.west>), nnweight($i j$, 1))
      node((rel: (1.1cm, -0.4cm), to: <n2_5.east>), nnweight($i j$, 2))
      node((rel: (-1.0cm, 0.4cm), to: <n2_0.west>), nnbias($i$, 1))
      node((rel: (1.1cm, -1.2cm), to: <n2_0.east>), nnbias($i$, 2))

      layer(0, [Inputs (#nninput())])
      layer(1, [Input layer (*0*)])
      layer(2, [Hidden layer (*1*)])
      layer(3, [Output layer (*2*)])
      layer(4, [Outputs (#nnoutput())])
    },
  ),
  caption: [A simple feed-forward neural network with one hidden layer. Inputs #nninput(i: $i$) enter from the left, connections between layers carry weights $w_(i j)^((l))$, and each neuron output is passed forward as an activation to the next layer. On the right side, we can see the outputs of the neural network as $y_i =$. At the top of layer 1 and 2, we can see a special _bias_ neuron, carrying the value 1. Neurons with a dashed outline aren't real neurons, but they are usually represented in diagrams as neurons for clarity.],
  // placement: bottom,
) <fig-neural-network>

ANNs are usually composed of so-called _layers_. The first layer is called the _input layer_, as it connects the inputs to the rest of the network. At the end of the network, we have the _output layer_. This layer's neurons' activations (outputs of the activation function $phi$) become the actual output of the neural network. And lastly, in the middle, we can have an arbitrary amount of _hidden layers_. These are called "hidden" because they are sandwiched between the input and output layers, hidden from plain sight #footnote[These layers are not truly hidden; they can still be inspected just like the input or output layers. They're still represented in the computer as matrices of numbers that aren't really hidden, the name just comes from the fact that they're between two layers that act as interfaces to the outside world (the input and output layer).].

Between the individual layers, we have these connections called _weights_. Their mathematical notation is the letter $w$. These weights are where the neural network's "smartness" comes from. When creating a new neural network, these weights are usually given random values, and during the training of the neural network, these weights are nudged around a little (they slowly change their values). This nudging of the weights is what makes the network learn.

Every neuron in the ANN usually has a special input, called a _bias_. In @fig-neural-network, we can see a special neuron at the bottom of the input and hidden layer. It has the value "1" written inside, because it acts like a fake neuron that always outputs a constant value of 1 (its output of its activation function $phi$ is always 1). It is then connected to every neuron in the subsequent layer through a fake weight, and this fake weight is what's called a _bias_ and is instead denoted by the letter $b$.

And lastly, at the right side of the diagram, at the end of the neural network, we have its output. This is its final prediction and it's denoted by the value $hat(Y)$ (read "_y-hat_"). The values of the individual outputs $hat(y)_n$ are the same as the activations of the neurons of the last layer. The last layer of the neural network will have as many neurons as there are outputs. In some cases, we want a single neuron as output, in some, we want hundreds. It depends on the use-case, requirements, and the achitecture used.

=== Expressing Neural Networks in Mathematical Notation

#figure(
  diagram(
    // debug: true,
    spacing: (0.5cm, 2cm),
    // cell-size: (0pt, 1em),
    {
      // input and weight nodes
      for x in range(3) {
        node((0, x), nnact(x + 1, 0))
        edge(<sum>, nnweight($c, #(x + 1)$, 1), label-sep: 10pt, "o->", label-anchor: "center")
      }

      // summation node
      node(
        (6, 1),
        text(size: 3em, $Sigma$),
        shape: circle,
        fill: purple-fill,
        stroke: purple-stroke,
        name: <sum>,
      )
      node(
        (rel: (0pt, -1.5cm), to: <sum.south>),
        $w_(c, 0) = b_c^((1))$,
        name: <bias>,
      )
      edge(<bias>, <sum>, "o->")

      // activation block
      node(
        (9, 1),
        text(baseline: -1pt, size: 2em, $phi$),
        shape: rect,
        fill: purple-fill,
        stroke: purple-stroke,
        name: <act>,
        inset: 1em,
      )
      node(
        enclose: (<sum>, <connector>),
        fill: gray.lighten(70%),
        inset: 2em,
        layer: -1,
        stroke: stroke(paint: gray, dash: "dashed"),
        corner-radius: 1em,
        name: <neuron>,
      )
      node((rel: (0pt, 10pt), to: <neuron.north>), [Neuron number $c$ in layer $L^((1))$])
      node((12, 1), shape: circle, radius: 1mm, stroke: black, name: <connector>)

      edge(<sum>, <act>, $z_c^((1))$, "->")
      edge(<act>, <connector>, nnact($c$, 1), "->")

      for x in range(3) {
        edge(<connector>, (17, x), $w_(#(x + 1), c)^((2))$, label-sep: 10pt, label-anchor: "center", "->")
        node((17, x), $w_(#(x + 1), c)^((3)) a_c^((1))$)
      }
    },
  ),
  caption: [A diagram of a single neuron inside the neural network inside layer $L^((2))$ with the position $c$. From the left side, activations from the neurons from the previous layer multiplied by the weights as well as the bias are all passed into the transfer function. The result of the transfer function (called the pre-activation) is passed to the activation function $phi$. Afterwards, the activation is multiplied by the individual weights and passed to the neurons in the next layer.],
  placement: top,
) <fig-neuron-in-network>

Let's introduce some basic mathematical notation so that we can make the explanations of later concepts easier to understand. Let's start with the _layers_. The individual layers of the neural network are labeled #nnlayer[$l$], with $l$ representing the number of the layer starting with 0 at the input layer. So, #nnlayer(0) is a direct reference to the input layer --- layer number 0.

Inside these layers, we have _neurons_. @fig-neuron-in-network shows a single neuron inside an example network. On this illustration, we can see some of the important mathematical notation that's used when describing the parts of a neural network. This neuron lives in the layer #nnlayer($l$). It's connected via weights to neurons in the previous layer #nnlayer($l - 1$) and to neurons in the next layer #nnlayer($l + 1$).

Like _neurons_ live _inside_ a layer, _weights_ live _between_ two layers. If a weight lives between layers #nnlayer($l - 1$) and #nnlayer($l$), and it connects neuron at index $j$ in layer #nnlayer($l - 1$) with the neuron at index $i$ in layer #nnlayer($l$), then we denote it using the notation #nnweight($i j$, $l$). Every weight that connects two neurons is just a scalar value. Since most of the time, the weights connect every neuron in layer #nnlayer($l - 1$) to every neuron in #nnlayer($l$), if we are connecting two layers that contain $m$ and $n$ neurons, respectively, we will need a total of $m times n$ weights. These weights are usually represented not individually, but as a matrix of weights. This matrix is denoted by using #nnweightm($l$), with $l$ being the number of the layer where the weights connect to. The _bias_ is a special value that is added to every neuron's transfer function separately. Its mathematical notation is #nnbias($i$, $l$).

#TODO[explain why the bias exists #cite(<BibDeepLearningBook>).]

Every layer's outputs are called _activations_ of the neurons. These activations of the neuron $i$ in a given layer #nnlayer($l$) are denoted by #nnact($i$, $l$). These represent the output of the neuron. A special case of the activation function is #nnact($i$, 0) (activation of #nnlayer(0)), because they correspond to the input vector #nninput(). So, we can say that $#nnact([], 0) = #nninput()$. Here we can see how a pre-activation is computed:

$ #nnpreact($$, $l$) = #nnweightm($l$) #nnact($$, $l - 1$) + #nnbias($$, $l$) $

Now, with all the notation defined, we can define a _forward pass_. A forward pass is the process of taking activations from one layer, computing pre-activations, applying the activation function, and passing the resulting activations to the next layer. For a whole layer $l$, the forward pass is:

$
  #nnact($$, $l$) = phi(#nnpreact($$, $l$)) = phi(#nnweightm($l$) #nnact($$, $l - 1$) + #nnbias($$, $l$))
$ <math-forward-pass>

In @math-forward-pass, we define the activation (output) of a single layer as the activation function $phi$ applied to the pre-activation of the current layer.

== Using ANNs to Solve Problems

We can use neural networks for solving a huge variety of tasks. However, in 99% of cases, these tasks all boil down to three main categories of tasks: _clustering_, _classification_, and _regression_. Since this thesis is concerned about supervised learning, I will skip clustering entirely and just focus on regression and classification. Clustering is a branch of machine learning called _unsupervised learning_, where our data we train the ANN on isn't labeled, which means that there isn't a concept of a "right output". We let the ANN determine the right answer my itself. On the other hand, classification and regression are part of _supervised learning_. This branch of ML requires labeled data, which means that for every set of inputs, we have a "right output".

=== Regression

#figure(
  image("fig/graphs/house-regression.svg", width: 100%),
  caption: [A graph showing a simple regression task. It only has one input parameter (because if we wanted to visualize more, we would need one extra dimension for every single input parameter). The input parameter is the size of the house, the output is the price in thousands of EUR. The model is is able to draw a line across the data that approximates the relationshipo between the house's size and its price. Because of the fact that the model draws a line, it's called a linear model. In neural networks, the model is able to draw all sorts of squiggly lines, not just straight lines through the data.],
  placement: auto,
)

Regression is a problem where our neural network is predicting a single value #nnoutput(i: 1) from the given set of inputs $#nninput(i: 1) ... #nninput(i: $n$)$. One good example of a regression problem is a house price prediction task. We have a dataset that contains information about individual houses, like their size in m#super[2], number of rooms, age, and neighborhood quality. The dataset also contains the house's estimated price. We train our neural network on this data, and if the training goes well, we will have an ANN that will be able to predict relatively accurately the estimated price of any house, given that we provide the neural network with the house's properties. In this case, the house's properties are the inputs to the neural network, $#nninput(i: 1) ... #nninput(i: 4)$. The predicted price of the house is the output #nnoutput(i: 1).

=== Classification

Classification is when we train a neural network to sort data into a finite number of _classes_ or categories. A class is just a fancy term for a single type of data. A simple example of a classification task would be to determine whether a certain image is a picture of either a taco, cat, goat, cheese or pizza. The neural network then outputs a set of probabilities that the image falls into each one of these classes.

#figure(
  image("fig/graphs/classification-boundary.svg", width: 100%),
  caption: [An example of a classification task where the ANN is tasked to draw a line between a cat and a taco. On the horizontal axis, we have a rating of protein content and on the vertical axis, we can see a cuteness rating in the interval $[0.0, 1.0]$. The neural network learns to distinguish cats from tacos based on these two parameters and predicts a probability that an input (a pair of values of protein content and cuteness) is either a taco or a cat.],
  placement: auto,
)

== Loss Functions -- How Do ANNs Learn?

Training a neural network is similar to teaching a child to draw the letter "A". The teacher shows the child and an A looks like. The child tries to draw the letter. The teacher compares the written A with their own correct version of an A in their head. Then, the teacher gives feedback to the child: "the A is too slanted", or "the middle bar is too low". The child then listens to the advice, and tries to draw an A again. This time, the A is a little more correct than the previous one. This process is repeated.

In the context of a neural network that is being trained on a regression task, the process is very similar. Firstly, we show the ANN some input data. by "showing data" i mean that we pass the data as an input to the neural network. we do a single forward pass of the data, and then, we see what the neural network predicts. After the prediction is made, we can assess how well the neural network performed. For this assessment, we need to somehow compare the predicted output #nnoutput() to the expected output $y$. For this, we can use a variety of so-called _loss functions_.

=== Mean Squared Error (MSE)

The mean squared error's name indicates exactly what it does. It is the mean (average) of the errors _squared_. To compute the _error_, we compute the difference between the expected output $y_i$ and the predicted output #nnoutput(i: $i$). We compute this difference for all the outputs of the neural network, square all of them, add them all together and divide by the number of outputs $n$. Here's a mathematical formula for MSE:

$ "MSE" = 1 / n sum_(i=1)^n (y_i - #nnoutput(i: $i$))^2 $

A good feature of this loss function is that it always outputs a positive loss because of the fact that we square every error. Also, since there's a square, it punishes large error quite a lot compared to small errors. This helps the neural network to learn faster #TODO[check if makes sense and citation needed].

MSE is used mostly for regression tasks.

=== Mean Absolute Error (MAE)

This function is very similar to MSE, except that it doesn't square the errors, it just computed the average of the absolute values of all the errors. Thus, its mathematical formulation looks like this:

$ "MSE" = 1 / n sum_(i=1)^n |y_i - #nnoutput(i: $i$)| $

== Binned Regression <section-binned-regression>

Binned regression is a term #cite(<BibMohanedPairwise>, form: "author") introduced in the unpublished manuscript _Learning Glyph Value Estimation via Pairwise Comparison_. The term is new, however, the concept behind it isn't.   mix between regression and classification. Instead of having a single neuron in the output layer, we have multiple neurons on the output. Each of these neurons corresponds to a _centroid_. A centroid is simply a number that is represented by that neuron.

The binned regression that is implemented in this project went through multiple iterations. Here, I will explain the first iteration stolen from Mohaned #cite(<BibMohanedPairwise>) and then also the improved implementation and a possible explanation of why the previous one didn't work and why it needed an improvement.

The first implementation looked something like this:

We divide the interval [0..100] into so-called "divisions", whose count is denoted by the letter $D$. We then calculate the distance between so-called "centroids". This distance is calculated in this manner:

$ Delta_c = 100 / D $

We then calculate the number of _centroids_:

$ C = D + 1 $

After that, we can simply compute the bin centers as evenly-spaced points across the interval $[0, 100]$:

$
  c_i = i dot ("end" - "start") / (C - 1) = i dot (100 - 0) / (C - 1) = i dot 100 / (C - 1) quad "for" quad i = 0, 1, ..., C - 1
$

For a number of divisions $D = 5$, we would get a centroid count of $C = D + 1 = 6$.


#figure(
  number-line(
    points: (0, 25, 50, 75, 100),
    start: -25,
    end: 125,
  ),
  caption: [The distribution of centroids on the number line (first iteration). $C$ centroids are evenly distributed from 0 until 100.],
)

The issue with the first iteration is a boundary effect. If the centroids live only inside $[0, 100]$, then a target near the edge, say $x approx 0$ or $x approx 100$, can only be represented by centroids on one side. When the model turns the bin probabilities back into a scalar by computing an expected value,

$
  hat(x) = sum_i p_i c_i,
$

the prediction is pulled toward the interior because there are no centroids outside the interval to balance the distribution. In practice, this creates worse errors at the edges than in the middle of the range.

By extending the centroid set to $[-Delta, 100 + Delta]$, the model gets one extra centroid on each side of the interval. That gives it room to place probability mass slightly outside the valid range, so the weighted average can still land near the boundary instead of being biased inward. After decoding, the final value is clamped back to $[0, 100]$, but the extra centroids help the regression behave more symmetrically near both ends.

#TODO[i confirmed by experiment `experiment-centroid-distribution.ipynb` that indeed the NN is performing worse with the first version of the centroids... Now... this is an experiment, so do I put it into the next part? Or do I mention it here? The actual explanation of binned regression belongs here but there's also an experiment that provided me with the information that the version 2 is better and idk where to mention that...]

#figure(
  number-line(
    points: (-25, 0, 25, 50, 75, 100, 125),
    start: -50,
    end: 150,
  ),
  caption: [blablabla],
)

#figure(
  rect([visualization], width: 100%, height: 7cm),
  caption: [A diagram of the last layer of the neural network, with the number of neurons corresponding to the value of $C$.],
)

== Development Enviromnment: `pip`, `poetry` and `uv`

i was choosing between pip, poetry and uv.

Pip is an old classic, it's not bad but it has the problem that if a package is not needed anymore, removing it from the requirements.txt doesn't remove it from the environment `.venv`. So then, I have to either keep my environment polluted, or every once in a while, remove the entirety of the .venv folder, and re-create the environment using pip again.

poetry is more modern, it's not bad because it uses the more modern `pyproject.toml` file for specifying dependencies. declaring dependencies this way is super cool because removing a dependency from the list actually removes it form the venv. This makes sure the enviroment always mirrors the pyproject.toml file, making sure all devs are always working int the same environment. The downside and the reason why i didn't choose it was that Poetry doesn't manage Python versions. Poetry simply uses the system Python binary. Inside the pyproject.toml file, there is a way to restrict the project's Python version in which it runs:

```
requires-python = ">=3.13"
```

however, this only restricts the Python version. So the person running the project has to manually install the correct python version. This might be a tedious process, depending on the OS.

On the other hand, uv has a special file `.python-version`, which specifies the version of the Python to use for the project, automatically downloads it and runs the code with that version of python. This makes sure that the reproducibility of the environment is very high; every developer has the same environment and thus if any bugs arise, they have a very low chance of being caused by the development environment.

Also, I chose uv because it's much faster than Poetry (it's written in Rust), and offers some very helpful commands.

=== Using `uv`

First of all, if you wanna run anything using uv, you can run it using `uv run ...`, so for example `uv run python main.py`. This run the Python file using the interpreter located at `.venv/bin/python` instead of the system binary.

idk if i wanna mention anything else about uv... #TODO[find out if mentioning anything else about `uv` is useful]

=== Using Environment Variables for Specifying Developer-specific Configurations

I am using the file `.env` located in the root of the project. This file can be created by creating a copy of the `.env.template` file. This file is useful because it isn't committed to the Git repository and thus stays on the developer's machine and the developer can specify their own machine-specific settings. In my case, there's only a single setting located in the .env file, and that is:

- `MGML_DEVICE`: It's used to specify the device for training
- #TODO[maybe other shit]


== Software Architecture:

Package Structure: Describe the mglyph_ml Python package in src/ and why functional code was extracted from notebooks for reusability.

Notebook Workflow: Explain the separation of notebooks for interactive experimentation versus core logic.


*Data Engineering Pipeline:*
Archive Format: Detail the .mglyph zip-like structure and the manifest.json schema (using Pydantic models like DatasetManifest).
Optimized Loading: Describe your solution for memory constraints: loading compressed PNGs and decoding them on-the-fly using cv2.imdecode and ThreadPoolExecutor for parallelism.
*Experiment Orchestration and Tracking:*
ClearML Integration: How you used ClearML for logging metrics, parameters, and maintaining experiment history.
Automation with Papermill: Describe using convert-notebook.sh to inject parameters into notebooks for batch processing.
Remote Execution: Practical use of the "Sophie" server with tmux to run long-term training sessions.
*Testing and Validation:*
Unit Tests: Briefly mention using pytest to verify critical logic, such as the labels_to_bins mapping.

= Development Environment & Infrastructure

#TODO[here i actually explain uv, environment setup, clearml, papermill...]

== ClearML

ClearML is an online MLOps platform. MLOps is a paradigm that encompasses a set of principles, components, roles, and architectures that all aim to deliver and run machine learning models more efficiently and reliably #cite(<BibMLOps>). ClearML has a big ecosystem, but for this thesis, we are only gonna use clearml for its monitoring capabilities and experiment comparison capabilities. Their free tier as of today offers 1GB of free metrics space, and 100GB of artifact storage (for storing models themselves) which is more than enough.

At the core of clearml is a _task_. A task is a single run of any experiment and it's stored on the clearml servers. When a task is run, clearML creates an instance of the task on the server. It also reports all the hyperparameters to the ClearML server #TODO[insert reference to hyperparam section].

== Hyperparameter optimizers

ClearML offers a feature called hyperparameter optimizering. Quite a lot of times, when training a neural network, there is a lot of stuff we can tune. The number of layers, the size of each layer, which activation functions to use, how to augment the data while training, etc. Most (if not all) of these parameters have some impact on how the final model behaves. That means, that if we have some kind of metric (like let's say the average of the last 10 losses #TODO[make sure to mention what a loss is before somewhere]), we can run the experiment with a certain set of parameters, see the value of the metric, then we run the experiment with a slightly different set of parameters, see how the metric changed, and of the metric went in the correct direction (in this case, down, because we want to minimize loss), that means, that we are doing something good. Then, we explore this so-called "hyperparameter space" (i think it's called that waym correct me if i'm wrong...) and an algorithm like Optuna, RandomSearch or GridSearch tries to find an optimal configuration of hyperparameters that yields in this case the lowest loss (but it can be any metric and it can either maximize the metric or get it as low as possible).

However, when you wanna use HO for your task, you need the full ClearML setup. This includes all the agents as well as a correctly set up repository.

== Unsuccessful Pipeline Approach

ClearML has a concept of pipelines.

#figure(
  image("fig/diagrams/clearml-pipeline.drawio.svg"),
  caption: [A visualization of my idea of using clearml pipelines for experiments. hyperparams are injected into the pipeline itself, and can be used by all steps of the pipeline. When a dataset is generated, it's cached in ClearML's dataset cache and since the dataset creation is deterministic, we can re-use the same dataset for future experiments if the initial dataset creation parameters are the same. #TODO[this caption might be too long...]],
  placement: top,
)

At the beginning of my journey, I wanted to use these pipelines for experiments. The idea was to have a clearML hyperparameter optimizer try to search the hyperparameter value space (idk if this term exists...) and to run the pipeline for every single run of the experiment. The pipeline itself is a ClearML task, so it made things simpler. However, my problem with this was that the usage of pipelines in clearml requires the entire setup with agents and everything. My experiment itself was usually only a few seconds long, and the overhead of clearml on top of the experiment was making things too complicated to debug. If you wanna run a HPO, you gotta first run the experiment task once, so that it gets registered onto the ClearML servers. After that, ClearML assigns a unique ID to the experiment's task. This unique ID has to be manually pasted into the HPO so it knows which task to clone. What clearML does is that when you actually run the task even on your local computer, it reports the current commit hash to the clearml server as well as all the uncommitted changes. It then clones the repo onto the agent, and checkouts the commit. It then applies the uncommitted changes, and starts setting up the virtual environmnent. After that, it finally starts the Python script. Mind you, clearML cannot run Jupyter notebooks in this pipeline-oriented fashion. So instead of experimenting in your notebooks, you gotta experiment in actual Python files which is in my opinion super impractical since the whole point of Jupyter notebooks is that one can try out different things and debug super easily since you can strategically run parts of your code and see how it behaves. If something went wrong, I had to use the Clearml web UI to inspect the logs. The whole process was tedious and the feedback loop from pressing the "Enter" button on my laptop and getting error messages was just too long and not sustainable. So, in the end, I realized that /tudy cesta nevede/ and I decided it's best to switch to a more local-oriented approach for running experiments.

== The Final Approach Using Papermill

Finally, I decided to adopt a much simpler approach to running experiments. prof. Herout recommended to me that he uses a Python parameter injection library called Papermill #footnote[https://papermill.readthedocs.io/en/latest/]. If one wants to use papermill, this is what you gotta do:
- install Papermill as one of the project's dependencies
- label one cell in your Jupyter notebook with the tag "parameters" (yes, Jupyter cells can be tagged)
- use the command:
```shell-unix-generic
papermill input-notebook.ipynb output-notebook.pynb -p a 42 -p b 'value of b'
``` #TODO[i dont like this formatting of the code]
- wait for papermill to finish and the executed notebook with the injected parameters will be available in `output-notebook.ipynb`.

Papermill works by simply inserting a new artificial cell into the notebook rigt after the `parameters`-tagged cell. This cell simply contains for this example two lines: ```py a=42``` and ```py b='value of b'```. When the notebook is run, the cell after the `parameters` cell simply overrides the global variables defined in the `parameters` cell. In practice, I do not call Papermill directly. Instead, I use a small convenience wrapper script called `convert-notebook.sh` #footnote[Credit: prof. Herout], which standardizes the command-line interface for all experiments. The script automatically handles output placement and naming, sanitizes run names, and converts `key=value` arguments into the correct Papermill `-p` format. This reduces repetitive boilerplate, keeps experiment outputs consistently organized in a nice `out/` directory, and helps with naming each experiment by a unique name that usually contains the names and values of varied parameters for that specific experiment.

== Trouble With ClearML

During my work on the bachelor's thesis, i hit multiple roadblocks when using clearml. I think it's important to mention all of the roadblocks here, so that the future person working on this won't hit the same roadblocks.

First of all, clearML is VERY complex. There are agents, pipelines, datasets, tasks, workers, queues... Every one of these components has a gazillion parameters that have to be tweaked for the system to work correctly.

ClearML has a pretty extensive documentation #footnote[https://clear.ml/docs/latest/docs/], but since ClearML is not very popular as an MLops platform, not a lot of questions about clearml can be found on the open web (like stackoverflow and such). So, you simply gotta read through the documentation and learn for yourself. Many things are not well described in the docs, and you need to dig through the source code to find the information you need. I found a useful tool called DeepWiki #footnote[https://deepwiki.com/clearml/clearml] which has thousands of Git repositories indexed (including the ClearML one) and is able to answer questions about the library with answers grounded in references to the source code. They also have a Slack channel #footnote[https://clearml.slack.com], where the employees answer questions.


2. Failure of the Pipeline Approach
Ultimately, you felt you didn't truly understand the system despite the time invested. You noted that the pipelines never reached a state where you were satisfied or where they functioned reliably. Consequently, you reverted your code by about 15 commits, discarded the pipeline branch, and switched to what you called a "braindead method"—which worked for your needs.
3. Shift to Alternative Orchestration (Papermill)
Because of the friction with ClearML's native optimization tools, you shifted your strategy for running multiple experiments:

Instead of ClearML pipelines, you used Papermill to inject parameters into your Jupyter notebooks.
You managed these runs manually on the "Sophie" server using tmux, allowing you to run batches while you slept without relying on ClearML's orchestrator.

4. Resource and Storage Constraints
Even while using ClearML just for logging, you encountered further issues:

Metric Storage Limits: You eventually hit the 1GB storage limit for metrics on the ClearML free tier.
Maintenance Overhead: You had to spend time deleting old experiment history just to keep the system running, which discouraged keeping a long history of hyperparameter trials.

5. Supervisor's Own Struggles
It is worth noting that your supervisor, Herout, also attempted to install a local ClearML server and concluded that it was a significant "pain," eventually admitting that the installation "laughed in his face." This reinforced the decision to stick to the hosted version for basic reporting rather than trying to build a complex, local optimization cluster.

== No Label Normalization

#TODO[explain that I was using normalized labels from 0..100 to 0..1, but it was completely useless, the NN is able to predict values from 0 to 100 no problem.]

= Creating High-Quality Datasets <chapter-creating-datasets>

This chapter explains the basics of the `mglyph` library, and later explains how the library can be used for the creation of high quality datasets that can be used in experiments.

== The `mglyph` Python Library

The manual creation of malleable glyphs might prove challenging. Generating and packaging thousands of images in a strict output format is inherently repetitive and labor-intensive. One of the primary motivations behind the `mglyph` Python library was therefore to abstract this complexity away from the glyph designer and make the process of creating and distributing glyphs as frictionless as possible.

The library is published on PyPI #footnote[https://pypi.org/project/mglyph/] under the name `mglyph`. It is open-source, and its source code is available on GitHub #footnote[https://github.com/adamherout/mglyph]. In practice, it serves as a framework for designing, generating, and distributing malleable glyphs. Its core is implemented in Python, a widely adopted language with which many developers are already familiar. Python's syntax also offers the flexibility required for a library of this kind.

When one wishes to create their own malleable glyphs, it's best to start at the tutorial that's linked in the repository's `README.md` file. The tutorial showcases what's possible, illustrating many interesting ideas and techniques. After reading the tutorial, one can tweak some parameters to see how the glyphs react.

At the core of any glyph is a _drawer_ function. In `mglyph`, `Drawer` is a `Callable` a.k.a. function with the following signature:

```py
type Drawer = Callable[[float, Canvas], None]
```

One possible implementation of this callable type may take the following form:

```py
import mglyph as mg

def simple_circle(x: float, canvas: mg.Canvas) -> None:
    radius = mg.lerp(x, 0, canvas.xsize / 2)
    canvas.circle(center=canvas.center, radius=radius, color='red')
```

A `mg.Drawer` takes two parameters: a `float` and an `mg.Canvas`. In Python, both arguments are passed by assignment (object reference): `x` is an immutable numeric value, while `canvas` is a mutable object that the function can draw into. In this example, the radius is computed by linearly interpolating $x in [0.0, 100.0]$ onto $[0.0, "canvas.xsize" / 2]$. The canvas size is 2.0, but it is better to express dimensions relatively. This makes the design intent clearer and keeps the glyph definition readable. Here, the circle radius spans from 0.0 to half of the canvas width (`xsize`), so it grows from a point to edge-to-edge.

Now that the `mg.Drawer` (```py simple_circle()```) is defined, we need a way to see what the glyph actually looks like with different values of $x$. Thankfully, the creators of the `mglyph` library provided multiple ways to visualize the glyphs. One of the most important functions is the ```py show()``` function. It allows us to see what the glyph looks like with different values of $x$ passed down to the function. It accepts many parameters, but here are the two most important ones:

```py
def show(
    drawer: mg.Drawer | list[mg.Drawer] | list[list[mg.Drawer]],
    x: int | float | list[float] | list[int] | list[list[float]] |
       list[list[int]] = [5, 25, 50, 75, 95],
)
```

The exact mechanisms behind ```py show()``` for the purposes of this work are not important. However, it is important to understand how the `mg.Drawer` is used by the `mglyph` library.

#figure(
  image("fig/diagrams/show-function.drawio.svg", width: 100%),
  caption: [Diagram of the inner workings of the ```py show()``` function from the `mglyph` library. For every $x$ that gets passed into ```py show()```, it instantiates a new `mg.Canvas`, and passes it down to the ```py drawer()``` so that it can draw the glyph based on the passed argument `x`. #TODO[this diagram needs shadows and nicer arrows... make it nicer u know]],
  placement: top,
)

#TODO[much more needs to be explained about the library... we need some basic glyph creation examples maybe]

== Inside The `.mglyph` Dataset File

I would define a dataset as a collection of samples that can be used to train, validate, and test a neural network in some way. I decided to represent a dataset as a single file with the extension `.mglyph` that contains all the samples that can be used by the person designing the experiment. It's essentially just a ZIP file disguised under a different file extension. The structure of the ZIP file is as follows:

// idk what the 't' stands for but it provides the exact syntax highlighting I need lol
```t
dataset.mglyph
├── manifest.json
├── 0000.png
├── 0001.png
└── ...
```

It contains a file called `manifest.json`, which contains all the information about the dataset, the author of the dataset, and the glyphs contained in the dataset, and it also contains all the glyphs rendered as PNG files numbered in a fashion starting from 0000.png (the number of digits depends on the total number of glyphs, so for datasets containing less glyphs, the number can be 2 or 3 digits, and for datasets containing 100 000 samples, it will be 6 digits).

Let's dig a little bit into the `manifest.json` file. An example of how such a file might look like is shown here:

#raw-annot(
  (line: 2, symb: [1], label: <code-manifest-name>),
  (line: 3, symb: [2], label: <code-manifest-creation-time>),
  (line: 4, symb: [3], label: <code-manifest-samples>),
)
```json
{
  "name": "Simple Star",
  "creation_time": "2026-04-15T10:30:00",
  "samples": {
    "0": [
      {
        "x": 12.34,
        "filename": "0000.png",
        "metadata": {
          "shape": "triangle"
        }
      }
    ],
    "1": [
      {
        "x": 56.78,
        "filename": "0001.png",
        "metadata": {
          "shape": "circle"
        }
      }
    ]
  }
}
```

At line @code-manifest-name, we can see inside the JSON the name of the dataset. This name can be as long as you need, and should reflect exactly what's inside the dataset. On line @code-manifest-creation-time, we store the time the dataset was created. This might be helpful if you create a dataset and forget about it and then later need to pinpoint exactly when and why you created it in the first place. It's stored as an ISO 8601 timestamp. Next, at line @code-manifest-samples, we have a JSON object called "samples" which contains key-value pairs that represent so-called _splits_. A split is a kind of a folder inside a dataset. We can divide the samples into these "folders" and then access the samples in each folder individually. A sample always belongs to one and only one split. Putting a single sample into multiple splits is not supported, but there isn't really a reason for us to do so, as the splits are usually used to separate training and testing data, and these two groups are mutually exclusive (they shouldn't have any overlap). Then, every split is a JSON list that contains objects of type ```py ManifestSample```. This object is defined in Python in the following way:

```py
class ManifestSample(BaseModel):
    x: float
    filename: str
    metadata: dict
```

Note that I used the Pydantic Python library for defining the manifest. This is super handy because Pydantic takes care of all the JSON parsing and validation and makes everything in the code type-safe (i get nice hints in my IDE when working with typed objects instead of classic Python ```py dict```s). Every sample that is in the dataset has its instance of ```py ManifestSample``` when the dataset is loaded. This sample is composed of the label of the sample `x`, the `filename` of the file where the sample located within the `.mglyph` dataset file, and a `metadata` dictionary. This is a key-value based structure where the creator of the dataset can embed special data about each glyph. This can be useful for example when your dataset contains multiple types of glyphs and you need to filter them out at training time or testing time.

== Creating a New Dataset

So, you decided that you want to build your own dataset. The first prerequisite is having a glyph that you wanna put into the dataset. When I am talking about a glyph, I am in reality talking about a `Drawer`. You need a `Drawer`. After you have a `Drawer` that can successfully draw a glyph in a reasonable about of time, you can go onto the next step. You can create a dataset in two ways.

=== Creating Dual Datasets

The first (and much simpler) way to create a dataset is by invoking the ```py create_and_export_dual_dataset()``` function from the `mglyph_ml.export` module. This function handles the creation of a very standard dataset type, which I called the _dual_ dataset. It's called _dual_ because it contains two splits, named "0" and "1". The reason why this type of dataset is used often is that we usually have one set of glyphs which are used for the training of the NN, and we have a separate set of glyphs which are used for the testing (for more info see @section-train-val-test). We simply provide the function with the `Drawer`, tweak a couple of basic parameters, and we have a new dataset ready to be used in an experiment:

```py
create_and_export_dual_dataset(
    name="Varying Star 1k Dual",
    path=Path("data/varying-star-1k-dual.mglyph"),
    drawer=varying_star,
    n_samples=10_000,
)
```

This single function creates the dataset at the specified path, using the specified drawer, and generates `n_samples` samples in _each_ split (so in this case, 20 000 samples get generated, 10 000 in each split). It's also important to note that this function doesn't generate the samples uniformly across the entire interval $[0.0, 100.0]$. Instead, it generates `n_samples` random floating point numbers between 0 and 100 and generates the glyphs from these values. After a discussion with my supervisor, we concluded that this method is better because it gives the NN more variety in the data. Instead of only seeing values like 0.1, 0.2, 0.3, 0.4, ..., it sees values like 0.1032, 0.1563, 0.2315, ... .

=== Creating Custom Datasets

If the dual dataset doesn't fulfill your needs, no need to worry! You can also create a dataset completely manually using a `DatasetBuilder` using the builder pattern. Here's a code sample demonstrating how to create a simple dataset manually:

#raw-annot(
  (line: 1, symb: [1], label: <code-dataset-builder-create>),
  (line: 2, symb: [2], label: <code-dataset-builder-linspace>),
  (line: 4, symb: [3], label: <code-dataset-builder-add-sample>),
  (line: 5, symb: [4], label: <code-dataset-builder-export>),
)
```py
dataset = create_dataset(name="Single-split Simple Star Uniform")
xvalues = np.linspace(start=0.0, stop=100.0, num=10_000)
for x in xvalues_train:
    dataset.add_sample(drawer, x, split="main", metadata={"shape": "star"})
dataset.export(path=Path("single-split-simple-star-uniform.mglyph"))
```

Firstly, we create the dataset at @code-dataset-builder-create using the ```py create_dataset()``` function. This function returns a ```py _DatasetBuilder``` object, which we can use to set up the dataset before we export it at @code-dataset-builder-export. After creating the builder object, we use the ```py np.linspace()``` function to generate values in a uniform fashion across the entire interval of 0..100 @code-dataset-builder-linspace. When we want to add a new sample into the dataset, we invoke the ```py _DatasetBuilder.add_sample()``` method, specifying the `drawer`, the value of `x` and the `split` where to put the sample. All of these parameters are required. Note that we specify the drawer for every single glyph we add to the dataset. This is super useful because this means that we can put multiple types of glyphs into a single dataset. On that same line, we also add metadata to the sample using the `metadata` parameter of the ```py add_sample()``` method. In this case, it's not really necessary, because every single glyph in the dataset will get the same metadata, which kinda defeats the purpose of the metadata, which is to defferentiate different glyphs in the dataset. But I added it to this example so that you can see how it's done. Lastly, we export the dataset on line @code-dataset-builder-export, specifying the `path`. After running this code, a new file called `single-split-simple-star-uniform.mglyph` will get created in the current working directory.

=== Training vs. Validation vs. Testing <section-train-val-test>

#TODO[maybe separate section? idk where to put this] \
Just a quick reminder on what the difference between a _training_, _validation_ and _test_ set is. In the context of malleable glyphs, we need a set of glyphs that are used to train the neural network. We can also take glyphs from the same set to kind of "steer" the training in the right direction. An example of this "steering" would be to check every $n$ steps or every $m$ epochs how the training is going by letting the neural network predict a couple labels. We could then use this information during the training to either reduce the learning rate, change the optimizer, or make some other decisions. This is totally okay. However, what is not okay is using _any_ of the data that has been used in some way during the training (even if the NN hasn't actually been trained on the data but it was only used for this 'steering'), we cannot use this data at the end of the training to validate how well a NN performs. This is why we need 2 sets of glyphs. The second split is used as a "testing" set at the end of the training to see how well the NN performs and this is the set that gives us information about how well the experient went.


#TODO[
  - [ ] what makes a dataset reusable...
  - [ ] link to some tutorials... we can't explain everything about the library here

]

= Experiments With Malleable Glyphs <chapter-experiments>

The whole point of this thesis was to find the limitations of computer vision in conjunction with the malleable glyphs. For running experiments, i created a series of Jupyter notebooks located at `notebooks/` named `experiment-*.ipynb`.

What's the point of an experiment? Well... we first ask questions and formulate a hypothesis. "Does augmentation help with the generalization of a NN?". A couple of hypotheses could be "We think that a neural network will generalize better when we show it augmented data because ...". Then, we test the hypothesis by constructing an experiment. We then run this experiment multiple times with different parameters (#TODO[insert link to hyperparams section]... these are explained in section blabla...) to confirm/reject the hypothesis. Based on the results, we might be asking some questions depending on what result we get, we modify the experiment a little, and we try again.

== Why Malleable Glyph Is Perfect For Exploring Computer Vision Limits

Finally, after explaining the mglyph, machine learning, and how datasets work, I can explain why the malleable glyph tech is so perfect for this task. The reason why the mglyph technology is so good for exploring the CV limits is that we are able hand-craft huge datasets of labeled image data that we can feed directly into the neural network. We are able to create a set of thousands of training/validation/testing samples, in a matter of minutes, that contain literally anything we want. We're also able to craft these images with labels with arbitrary xvalue resolutions, even down to 0.0001 of $x$ units. In the real world computer vision, the datasets are usually noisy and contain different perspectives, lighting conditions, background noise... They are not optimal for testing the neural networks themselves. With malleable glyphs, we are able to isolate these factors and vary one at a time, while observing what changing this one factor does to the traiining process of the neural network. This way, we are able to learn more about what the neural networks behave under certain conditions.

== The Base Experiment

The base of every experiment is a notebook called `experiment-base.ipynb`. This notebook is documented in a very detailed manner and it serves as a base notebook for most other experiments.

At the beginning of the notebook, there is a markdown title and explanation of the experiment's purpose (the base experiment as well). This is important because if anyone opens the notebook, they can immediately see what to expect inside the notebook without going through the code (they can see if they're interested in the notebook... the title serves as an _abstract_).

Now, every experiment has this thing called _hyperparameters_. These are parameters of a single run of a experiment. Usually, experiment Jupyter notebooks are not run a single time, but they're run multiple times with different combinations of hyperparameters. Then, the different runs of the experiment are compared to each other to answer a hypothesis. These hyperparameters are stored in a special Python `dataclass`:


```py
@dataclass(frozen=True)
class RunConfig(RunConfigBase):
    # The ClearML task tag.
    task_tag: str = "tag-2"

    # Where the dataset lies.
    dataset_path: Path = Path("data/simple-star-20k-dual.mglyph")

    # This seed is used by RNGs in the experiment to make it reproducible.
    seed: int = 328

    ...
```

There is a reason why the hyperparameters are specified in a dataclass like this and not just as global variables. My first approach was actually having the parameters laid out like global variables inside a single cell, just like this:

```py
task_tag: str = "tag-2"
dataset_path: Path = Path("data/simple-star-20k-dual.mglyph")
seed: int = 328
...
```

However, this approach has a significant drawback. Although being simple, there was no simple way to report the variables to the ClearML servers without duplicating every single variable name at least once. This is because the reporting to the ClearML servers is done via a method call ```py clearml.task.Task.connect(parameters)```, there was no way to somehow "package" these global variables into a single object that can be passed into the ```py Task.connect()``` method. So, what I had to do, was something similar to this:

```py
params = {
  "task_tag": task_tag,
  "dataset_path": dataset_path,
  "seed": seed
}

task.connect(params)
```

This approach means that for every new parameter added to the main `parameters` cell, one had to modify this new `params` dictionary to encompass the new parameter, writing every single parameter name three times. It's not the end of the world, but this kind of code duplication introduces space for bugs such as when you add a new parameter to the global `parameters` cell and forget to add it to the `params` dictionary, ClearML won't know about that parameter and hyperparameter optimization might break. After this, I tried capturing all the names of the global variables _before_ executing the `parameters` cell, and capturing all the names of the global variables _after_ running the `parameters` cell. This way, I could diff the two lists of variable names and see which variables were introduced. This method looks like it might work on the first look, however, it also has a significant drawback, and the drawback is that when you run the notebook once, all the global variables are created and everything works as expected. However, when you tweak some of the parameters, and run the cell again, the capturing logic that surrounds the `parameters` cell doesn't catch anything because the global variables all already exist in the global context, and the content of the `globals_before` and `globals_after` lists that should contain the names of the global variables introduced in the `parameters` cell will be identical, making the diff empty.

Finally, I settled on the approach of having a dataclass represent

// this is also related to Papermill...
// TODO write a papermill section as well

== Trouble With The Straight Line

The whole point of the base experiment is that it has to be perfect. The neural network has to be able to predict the value of $x$ pretty much perfectly. Prof. Herout told me that we cannot continue with creating new experiment before we have a perfect (or near-perfect) base experiment. What we defined as perfect is that the NN is trained on some glyphs, and it should be able to predict the value of $x$ _very_ accurately when tested on the training set (a.k.a. we show it the same samples that the NN was trained on). We visualized the neural network's prediction on the training set using a plot seen in @fig-perfect-predictor. We can see that it's basically a perfectly straight line, which means that the predicted $x$ is very close to the real value of $x$, as the graph doesn't deviate at all from the $x_"pred" = x_"real"$ line.

#figure(
  image("fig/graphs/truth-vs-x-base-experiment-perfect.svg", width: 80%),
  caption: [Plot showing the real value of $x$ on the horizontal axis, and the predicted value of $x$ on the vertical axis. In blue, we can see how a perfect model predicts. The red line is a visualization of how a troubled model might predict.],
  placement: auto,
) <fig-perfect-predictor>

Creating this perfect experiment proved much more difficult than anticipated. Firstly, I had trouble with getting the neural network to learn reliably. About half of the trainings, the neural network simply learned the average value of the dataset and just always outputted the same value. Instead of the $x_"pred" = x_"real"$ line, the line looked more like $x_"pred" = 50$ (we can also see this line in @fig-perfect-predictor). Just a flat line. I don't have a clear explanation of why the model refused to learn properly, but I solved it by increasing the number of training samples, and increasing the complexity of the shape learned. This phenomenon happened only in simple single-colored triangles, when the number of training samples was #(sym.lt.approx)1000. By changing the shape from the triangle or square to a star, and increasing the number of training samples to above 1000, the chance of the neural network falling into this trap decreased to nearly zero. Every once in a while, the network still fell into the same valley. However, enabling training data augmentations completely eliminated this. Enabling a rotation augmentation in the range of $[-1 degree, +1 degree]$ as well as a translation augmentation ranging at $[-1%, +1%]$ solved the issue and the base task now consistently produces perfect or near-perfect models.

== How Do I Design And Run My Own Experiment?

First of all, we need a bit of curiosity. We need to as ourselves a question that we want answered regarding computer vision and malleable glyphs. Examples of such questions include:

- Does augmenting my training data help the NN to generalize better?
- What will happen if we train the neural network on glyphs where $x in [0.0, 40.0] union [60.0, 100.0]$ and ask it to guess $x$ on glyphs that are in between the range $x in (40.0, 60.0)$? Will the NN interpolate or will it just guess values close to 40.0 and 60.0 depending to whichever the glyph in question is closer to #TODO[rephrase better]?
- How does reducing the number of parameters in the network (e.g. creating a 'baby network') affect its ability to generalize versus its tendency to overfit?
- Does the technique _hard sample mining_ reduce the loss on the few hard samples?

Natural curiosity is key here and the more questions we get answered, the more questions we will be asking. After interestedly #TODO[maybe find a better word :P] asking questions, we need to form a hypothesis. A hypothesis is a proposed explanation. It explains why we think a certain phenomenon occurs. In the context of the questions above, hypotheses include but are not limited to:

#TODO[JUJ ja neviem ci toto tu ma byt... mozno len popisem jednotlive experimenty... ale tam je dizajn uzko spojeny s implementaciou samotneho experimentu...]

After that, it's time to design an experiment. An experiment is programmed inside a Jupyter notebook. I recommend that you simply create a copy of the Jupyter notebook located at `notebooks/experiment-base.ipynb` and rename the notebook to something like `experiment-my-experiment-name.ipynb`. Choose a name that captures the essence of your experiment. Of course you can also create your own notebook as the base of your experiment, but you can do that once you're more comfortable with how the framework works. For now, stick to creating experiments by cloning the base experiment notebook.

The next step is to add/tweak some hyperparameters. These can be found under the section of the Jupyter notebook called "Hyperparameters" and they're all contained within the ```py class RunConfig(RunConfigBase)``` dataclass. Under these hyperparameters, you can tweak various parameters of the training process. This class looks similar to this:

```py
@dataclass(frozen=True)
class RunConfig(RunConfigBase):
    dataset_path: Path = Path("data/simple-star-20k-dual.mglyph")
    seed: int = 42
    learning_rate: float = 0.0005
    augment: bool = True
    task_name: str = f"Task Name [seed={seed},lr={learning_rate}]"
```

This is a simplified snippet of what the class looks like; by default, it has more parameters, and you are free to add as many as you like.

== Finding The Optimal Learning Rate

#TODO[this section explains the experiment I (and Mohaned) conducted to find the optimal learning rate scheduler.]

== The Gap Experiment

This experiment aims to answer questions garding the NN's capability to interpolate.

== Experimenting With Centroids

As I mentioned previously in @section-binned-regression, centroids are the building block of binned regression...

=== Finding The Optimal Number Of Centroids

#TODO[also explain that i did experiments to determine an optimal number of centroids... 3 is too little, 5 is ok, 10 is also ok, 20 is also ok... i'm sticking with less because the NN is smaller, simpler, and there's no need for more centroids]

=== Two Extra Centroids

In this experiment, I am trying to answer the question whether it's better to have $C$ binned regression centroids distributed on the interval $[0.0, 100.0]$, or whether it's better to have $C + 2$ centroids withthe two extra centroids located outside of the interval at $-Delta_C$ and $100 + Delta_C$. The reason why I designed this experiment was that when I was trying to create the base experiment, I was having trouble with getting a completely straight line. The line had always these small "tails" at the ends.

#figure(
  stack(
    image("fig/graphs/truth-vs-x-centroids.svg", width: 50%),
    image("fig/graphs/truth-vs-x-zoomed-centroids.svg", width: 50%),
    dir: ltr,
  ),
  caption: [
    Plot (and its zoomed version from x=0..10) that has on the horizontal axis the true value of $x$ and on the vertical axis the predicted value of $x$ by the network. It's evaluated on 2500 random samples from the test set (the NN hasn't seen any of the samples). We can see the imperfect endings at $x=0.0$ and $x=100.0$.

    #TODO[maybe increase the size of the font in the graphs here]
  ],
) <fig-truth-vs-x-centroids>

Me and prof. Herout hypothesized that it could be because the centroids at the edge have a smaller saying in what the final output of the network is, due to the way in which the NN calculates the final output. If we have a sample of $x=0.0$, the centroid at $x=0.0$ can have a very high output value, however, it is highly disadvantaged compared to the other centroids. As soon as one of the other centroids has a non-zero value, there is *no* way for the $x=0$ centroid to completely pull the final output of the NN to zero. It will always get pulled a bit away from $x=0$. We can actually see this in the zoomed plot (right side) in @fig-truth-vs-x-centroids, as at $x=0.0$, the predicted value of $x$ is actually _pulled_ by other bins away from 0, we can see that the tail at x=[0..1] is actually curved in the direction that represents the prediction of higher values $x$ than the real value, supporting this hypothesis.

#TODO[
  Maybe a good way to explain this is to show what happens when the last centroid is at 5 and 95. The network is physically incapable of predicting values below 5 and above 95, because of the weighted average principle. When we have centroids at 0 and 100, the network is capable, but the output is usually pulled in by the other bins a little bit.
]

#figure(
  // image("fig/graphs/loss-vs-x-centroids.png", width: 80%),
  // image("fig/graphs/loss-vs-x-centroids.png", width: 80%),
  rect(
    [graph showing loss vs x. #TODO[maybe unnecessary as the previous graph explains it pretty well...]],
    width: 100%,
    height: 8cm,
  ),
  caption: [Corresponds to @fig-truth-vs-x-centroids, and is just a different way to look at the data. Shows the loss on the vertical axis and the value of $x$ on the horizontal axis. As we can see, with the $C$ centroids evenly distributed on the interval $[0.0; 100.0]$, we have large losses at the edges of the graph and relatively small losses (in comparison) in the middle.],
)

=== Gap Between Centroids

#TODO[
  another experiment is:
  - Stred gap-u medzi centroidmi = ok?
  - Stred gap-u rovno na centroide = shit?

  to je hypoteza, mozeme otestovat a zistit ci naozaj ked je stred gap-u medzi centroidmi tak to bude lepsie ako ked bude stred rovno na centroide
]

= Results, Conclusion & Future Work <chapter-results-conclusion>

== Things I Would Have Done Have I Had More Time On My Hands

Sometimes, when I run the base experiment, it still gets large losses at the edges of the interval (at x=0 and x=100). I am really not sure why this happens,

Idk I am kinda sad that I didn't have more time to create more experiments, because I was mostly busy making the toolset for creating these experiments. However, on the bright side, the toolset is right now pretty mature and I was able to create these couple of experiments in a matter of hours. Of course, the framework always needs improvements, but it's pretty solid and it can definitely be used to set up new experiments easily.
