#set text(size: 10pt)


*Explain malleable glyphs: (maybe separate chapter)*
- what they are
- why research on them is important
- technical details on how they work (how can we encode a floating-point scalar value into a 1x1 in image?)
- show some examples of glyphs
- maybe touch on Q-methodology? Or just def mention Herout's article

*Explain machine learning: (separate chapter)*
- what machine learning is
- deep neural networks
- convolutional neural networks
- structure of neural network
 - maybe put like a diagram here of my NN (or maybe keep that later for the implementation?... or maybe just put a diagram of a generic CNN?)
- explain regression
- Explain binned regression
 - bins, logits, ...

*My implementation:*
- Python project using `uv` as build system
 - idk if it's relevant but i can explain why it's better than pip because dependencies are declared in a single file and `uv` makes sure the environment mirrors the declaration at all times (idempotency)
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
- augmentations:
 - AlbumentationsX
 - white noise, rotation, translation, random image background

*What I've found:*
- NNs are surprisingly good
 - even smaller networks are actually able to memorize the backgrounds from glyphs and thus generalize worse
 - so we need various augmentations and random backgrounds behind the glyphs to make it a challenge for the NN
- with the "template" experiment, the predictions lie on the y=x line perfectly
 - major hiccup was the "tail"... That one still pops up every once in a while, but augmentations help quite a bit
- There was also regression, but it had a huge "tail" that I didn't manage to remove
- I played around with the learning rate, tried different schedulers
- Explain that the NN has to be better than the worst-case, which is $"error" = 25 "units of x"$
- i will conduct more experiments based on the template task during the writing of the thesis...