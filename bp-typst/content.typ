#import "template.typ": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node, shapes
#import "@preview/cheq:0.3.0": checklist

#show: checklist

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

= Explanation of The Malleable Glyph

This chapter introduces the _malleable glyph_, explains its purpose, the current state of research surrounding it, #TODO[...]

== What Is a Malleable Glyph?

The concept of a malleable glyph was first introduced by #cite(<BibMalleableGlyph>, form: "prose"). They defined the malleable glyph as an image of a fixed size (e.g. $1 times 1$ inch), which carries encoded information in its appearance. For the purpose of their research, they gave this information a specific format: a single scalar value they call $x$, with $x in [0.0, 100.0]$. This scalar value is then used to dictate the appearance of the malleable glyph itself. To better understand how this works, let's look at some examples.

#figure(
  image("fig/glyphs/green-triangle.png"),
  caption: [A malleable glyph showing a green triangle that encodes the value of $x$ in the size of the glyph. A small value of $x$ renders a small triangle, and as $x -> 100.0$, the triangle fills the entire glyph.],
) <fig-green-triangle>

In @fig-green-triangle, we can see how a malleable glyph might look like. The triangle's size is linearly mapped to the value of $x$, interpolating from infinitesimal to edge-to-edge. The author of the glyph can choose to vary any aspect of the glyph in accordance with the parameter $x$. The size, color, shape, roundedness, spikiness, contrast, or even fractal structure #footnote[https://github.com/adamherout/mglyph] just to name a few. Another example of what's possible can be seen in @fig-fractal-tree.

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

#TODO[Quickly just introduce the challenge.]

Having so many different options with

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

== The `mglyph` Python Library

The manual creation of malleable glyphs might prove challenging. Generating and packaging thousands of images in a strict output format is inherently repetitive and labor-intensive. One of the primary motivations behind the `mglyph` Python library was therefore to abstract this complexity away from the glyph designer and make the process of creating and distributing glyphs as frictionless as possible.

The library is published on PyPI #footnote[https://pypi.org/project/mglyph/] under the name `mglyph`. It is open-source, and its source code is available on GitHub #footnote[https://github.com/adamherout/mglyph]. In practice, it serves as a framework for designing, generating, and distributing malleable glyphs. Its core is implemented in Python, a widely adopted language with which many developers are already familiar. Python's syntax also offers the flexibility required for a library of this kind.

When one wishes to create their own malleable glyphs, it's best to start at the tutorial that's linked in the repository's `README.md` file. The tutorial showcases what's possible, illustrating many interesting ideas and techniques. After reading the tutorial, one can tweak some parameters to see how the glyphs react.

At the core of any glyph is a _drawer_ function. In `mglyph`, `Drawer` is a callable type alias with the following signature:

```py
type Drawer = Callable[[float, Canvas], None]
```

One possible implementation may take the following form:

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
  caption: [Diagram of the inner workings of the ```py show()``` function from the `mglyph` library. For every $x$ that gets passed into ```py show()```, it instantiates a new `mg.Canvas`, and passes it down to the ```py drawer()``` so that it can draw the glyph based on the passed argument `x`.],
  placement: top,
)


= Machine Learning Fundamentals for Glyph Decoding

we explain all the important details that are necessary for decoding glyphs... There are an infinite amount of ways that we can train a NN to decode glyphs. For example we could train a neural network on pairs like it has been done by Mohaned Anene

This chapter provides the technical "how-to" of your project, serving as a manual for your successor.

#TODO[
  -
]

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

== Trouble With The Straight Line

Explain the trouble I went through with getting the damn straight line to work. I first tried a clssical regression model, I always got these huge tails on both ends of the straight line. Honestly I have no idea why that was, and I wasn't able to make it work successfully.

The second attempt as binned regression.

= Experiments And Results

The whole point of this thesis was to find the limitations of computer vision in conjunction with the malleable glyphs. For running experiments, i created a series of Jupyter notebooks located at `notebooks/` named `experiment-*.ipynb`.

What's the point of an experiment? Well... we first ask questions and formulate a hypothesis. "Does augmentation help with the generalization of a NN?". A couple of hypotheses could be "We think that a neural network will generalize better when we show it augmented data because ...". Then, we test the hypothesis by constructing an experiment. We then run this experiment multiple times with different parameters (#TODO[insert link to hyperparams section]... these are explained in section blabla...) to confirm/reject the hypothesis. Based on the results, we might be asking some questions depending on what result we get, we modify the experiment a little, and we try again.

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

I also tried putting the parameters simply as global variables, but there was an issue with that approach. The task is reported to ClearML including all the hyperparameters... #TODO[clearml section?]
// this is also related to Papermill...
// TODO write a papermill section as well
