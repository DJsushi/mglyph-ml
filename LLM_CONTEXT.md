# mglyph-ml: Comprehensive Repository Context for Thesis Planning

## Purpose of this document

This file is a long-form, NotebookLM-ready knowledge base for the repository. It is written to help with:

- Understanding what the project does end-to-end.
- Understanding why each component exists.
- Mapping experiments to thesis chapters and arguments.
- Capturing practical constraints, trade-offs, and open questions.

This document reflects the repository state as inspected on April 22, 2026.

## Project at a glance

The repository studies whether machine learning models can decode scalar values encoded by Malleable Glyphs (MGlyphs). In practical terms:

- A glyph rendering is generated from a scalar x in [0, 100].
- A model receives only the rendered image.
- The model predicts x.
- Performance is measured mostly by absolute error in x-space.

The project includes:

- Dataset generation tools for .mglyph archives.
- Neural architectures for both direct regression and binned regression.
- Reproducible experiment notebooks with parameter injection via Papermill.
- ClearML logging/reporting integration.
- Typst and LaTeX thesis scaffolds.

## Repository structure and role of each area

Top-level directories and their purpose:

- bp-typst: Current Typst thesis source and assets.
- bp-latex: Legacy/alternative LaTeX thesis template and assets.
- data: Generated datasets (.mglyph) and data docs.
- models: Stored trained models/checkpoints.
- notebooks: Main experiment workflow, from dataset building to model experiments.
- src/mglyph_ml: Reusable Python library code extracted from notebooks.
- test: Unit tests and test fixtures.

Top-level scripts:

- convert-notebook.sh: Runs notebooks with optional Papermill parameter injection and exports executed copies into notebooks/out.
- kill-jupyters.sh: Force-kills hanging ipykernel processes.

Project environment and dependencies:

- pyproject.toml specifies Python >= 3.13 and core packages (PyTorch, mglyph, albumentationsx, clearml, papermill, etc.).
- uv is the intended environment/package workflow.
- .env.template defines MGML_DEVICE (for example cuda:0).

## Core domain: Malleable Glyphs in this project

What Malleable Glyph means here:

- A scalar x in [0, 100] is encoded into visual shape properties of a glyph.
- The mapping is deterministic in generator code but can be made visually harder by random style/background choices.
- The ML model is tested on its ability to recover x from image only.

Why this is interesting:

- It tests machine perception of subtle shape encodings.
- It exposes model sensitivity to nuisance factors (background, color, slight transforms).
- It offers a controlled testbed to compare direct regression and binned approaches.

## Data format and dataset pipeline

### .mglyph dataset format

The project uses .mglyph as a ZIP container with:

- manifest.json
- png samples

Manifest model (from code):

- DatasetManifest:
  - name
  - creation_time
  - samples: map split_id -> list of ManifestSample
- ManifestSample:
  - x (label)
  - filename
  - metadata (dict)

Splits are keyed as strings like 0 and 1, but the code supports arbitrary split keys.

### Dataset generation code path

Main implementation is in src/mglyph_ml/dataset/export.py.

Pipeline:

- create_dataset creates a builder.
- add_sample adds (drawer, x, metadata, split).
- export renders each sample into PNG and writes manifest + files into ZIP.
- export_dataset convenience function:
  - Samples x uniformly in [0, 100] for train and test splits.
  - Explicitly includes boundary values 0 and 100.
  - Sorts x values.
  - Writes split 0 and split 1.

Rendering details:

- mglyph render is called at 512x512 during export.
- PNG compression is applied via PIL.
- Canvas parameters are configurable.

### Dataset loading path

Main implementation is in src/mglyph_ml/dataset/utils.py.

Behavior:

- Reads the whole .mglyph archive into memory.
- Parses manifest.
- Filters selected split.
- Optional index filtering and deterministic shuffling.
- Decodes PNG with OpenCV in a ThreadPoolExecutor (max_workers=32).
- Optional downscale to desired size, typically 64x64 for training speed.

Output object per sample:

- imid
- image (numpy array)
- label (float x)
- metadata (dict)

## Training data object

src/mglyph_ml/dataset/glyph_dataset.py defines GlyphDataset:

- Stores image arrays and labels.
- Applies optional transform callable (typically Albumentations -> tensor).
- Returns image tensor and label tensor.
- Includes utility to fetch random samples.

Important detail:

- Labels are currently kept as raw x values (not normalized in this class).

## Model families in the repository

### 1) Binned regression model

Implemented in src/mglyph_ml/nn/glyph_regressor_binned.py as BinnedGlyphRegressor.

Design:

- CNN feature extractor with three pooling stages.
- Classifier head outputs logits over bins.
- Number of bins = num_divisions + 3 (with extra outer bins).
- Bin centers include values beyond [0, 100] and final prediction is clamped to [0, 100].

Key methods:

- labels_to_bins: maps raw labels to bin indices.
- logits_to_labels: softmax-weighted expected value over bin centers.

Rationale:

- Reframes regression as discretized distribution prediction.
- Can stabilize learning and reduce some failure modes compared to direct regression.

### 2) Direct regression models

Implemented in:

- src/mglyph_ml/nn/glyph_regressor_gen1.py
- src/mglyph_ml/nn/glyph_regressor_gen2.py
- src/mglyph_ml/nn/glyph_regressor_gen3.py

Evolution trend:

- gen1: earlier architecture with larger kernels and adaptive pooling.
- gen2: streamlined CNN + MLP regressor.
- gen3: stronger backbone, dropout, and sigmoid output head.

Use case:

- Compare direct scalar prediction against binned strategy.

## Training and evaluation utilities

### Training loop utility

src/mglyph_ml/nn/training.py provides:

- train_one_epoch
- training_loop
- report_values for ClearML

Notable characteristics:

- Tracks timing breakdown (data loading, host-to-device, forward, backward, optimizer step).
- Uses MAE-like error metric in training logs (absolute difference in output space).
- Subsamples up to 10,000 train samples each epoch via random sampler in training_loop.

### Evaluation utility

src/mglyph_ml/nn/evaluation.py provides evaluate_glyph_regressor:

- Runs model in eval mode with no-grad.
- Computes average loss and average absolute error.
- Restores train mode after evaluation.

## Visualization support

Two visualization modules are present:

- src/mglyph_ml/experiment/visualization.py
- src/mglyph_ml/nn/util.py

Common plots include:

- Truth vs prediction scatter with y=x reference.
- Loss vs x scatter to inspect where errors concentrate.
- Training progress figure montage.

These plots are crucial for diagnosing the recurring tail behavior described in thesis notes and notebook comments.

## Experiment workflow philosophy

Documented in src/mglyph_ml/experiment/README.md:

- Version experiments as major.minor.patch.
- major: substantial conceptual shift.
- minor: related variation.
- patch: bugfix-level changes.

Parameter management strategy:

- Dataclass-based run configs.
- RunConfigBase utility supports:
  - clear_globals for clean Papermill injection in interactive contexts.
  - from_globals to instantiate config from injected variables.

Batch experiment execution:

- notebooks/experiment-commands.ipynb builds many convert-notebook.sh command lines.
- Papermill executes notebook variants with different seeds/gammas/gap settings.

## Notebooks: what each one contributes

### build-dataset.ipynb

Purpose:

- Explains .mglyph structure and manifest format.
- Implements concrete glyph drawers (simple star, varying star).
- Demonstrates dataset export.

Important idea:

- Glyph rendering can include random colors, random backgrounds, and geometric variability to prevent trivial memorization.

### experiment-base.ipynb

Purpose:

- Canonical template for most experiments.

Contains:

- Parameterized config cell (Papermill hook).
- ClearML initialization and parameter logging.
- Split-based data loading.
- Augmentation pipeline.
- Binned model initialization.
- Step-based training loop.
- Evaluation plots and logging.

### experiment-better-augment.ipynb

Purpose:

- Stronger augmentation to improve generalization.

Adds/changes:

- Hue shift.
- Gaussian noise.
- Larger affine perturbation range than base setup.

### experiment-hard-sample.ipynb

Purpose:

- Hard sample mining and edge-focused sampling.

Implements:

- Warmup random phase then hard mining phase.
- EMA sample scores from per-sample losses.
- Candidate sampling + hardest selection.
- Explicit edge/near-edge quota logic for x near boundaries.
- Optional edge-weighted objective.
- Diagnostics for selection statistics and score behavior.

### experiment-gap.ipynb

Purpose:

- Variant of base pattern (repository snapshot suggests heavy overlap with base/stability setup).

### experiment-regression.ipynb

Purpose:

- Direct regression experiment using GlyphRegressorGen3.

Notes:

- Uses MSE and MAE-style reporting.
- Includes actual-vs-predicted and loss-vs-x visualization.
- Appears to include legacy references (for example load_images_and_labels) that may not match current library API without adaptation.

### experiment-commands.ipynb

Purpose:

- Generates shell command batches to run many notebook variants.

Includes:

- Gap-parameter sweep patterns.
- Gamma sweep formulas and seed sweeps.

### worst-case.ipynb

Purpose:

- Synthetic random predictor baseline.

Value:

- Establishes lower-bound performance expectations for MAE/RMSE.
- Supports thesis argument that trained models must decisively beat random behavior.

## Logging, tracking, and reproducibility

Tracking stack:

- ClearML Tasks for metrics, parameters, and plots.
- Optional offline mode.

Reproducibility controls:

- Seed parameters propagated through data/augment/training paths.
- Papermill injection for explicit run configuration.
- Deterministic command construction in experiment-commands notebook.

Practical operational notes:

- MGML_DEVICE environment variable controls compute target.
- convert-notebook.sh supports run naming and parameterized execution.

## Tests and quality checks

Current tests are minimal but present.

Covered area:

- test/test_glyph_regressor_binned.py validates bin conversion and logits-to-label behavior.

Testing maturity:

- Unit tests mostly focus on binned conversion correctness.
- Broader integration/regression testing is still an opportunity area.

## CI and thesis publishing

GitHub Actions workflow .github/workflows/thesis-pdf.yml:

- Compiles bp-typst/thesis.typ to PDF.
- Publishes thesis.pdf to GitHub Pages artifact.

README states thesis is auto-published from the main line, while workflow currently triggers on master. This branch-name mismatch should be reconciled if publication reliability matters.

## Important technical insights already encoded in the repo

From code and notebook narrative, the project has converged on several practical insights:

- Augmentation is not optional for robust learning on glyphs.
- Learning-rate scheduling is a major sensitivity axis; gamma sweeps are justified.
- Binned regression is treated as a strong baseline/primary strategy.
- Error must be interpreted in x-space for thesis readability.
- Performance near edges and in sparse/hard regions deserves special handling (edge focus + hard mining).
- Random baseline should be explicitly presented to anchor evaluation claims.

## Risks, caveats, and cleanup opportunities

Current repository risks to mention transparently in thesis:

- Some notebooks look partially legacy relative to src APIs (for example experiment-regression references).
- src/mglyph_ml/visualization.py references objects/modules not visible in current tree snapshot (possible stale module).
- Branch naming mismatch between README text and CI trigger.
- Dataset loading currently preloads archives/images into memory, which can constrain scale.

These are not blockers for thesis value, but they are good to document as engineering limitations.

## Suggested thesis narrative mapping

A high-coherence chapter mapping that fits this repository:

- Chapter 1: Introduction and motivation.
- Chapter 2: Background.
- Chapter 3: Problem formulation and methodology.
- Chapter 4: Implementation and experiments.
- Chapter 5: Results, discussion, and conclusion.

How to map repo assets into that structure:

- Chapter 1:
  - Problem statement: decode scalar x from MGlyph image.
  - Why this matters for glyph-based quantitative visualization.
  - Research objectives and hypotheses.
- Chapter 2:
  - MGlyph encoding concept.
  - CNN/regression fundamentals.
  - Binned regression idea and expected advantages.
  - Experiment management tools (Papermill/ClearML) in reproducible ML.
- Chapter 3:
  - Dataset formalization (.mglyph, manifest, splits).
  - Data generation process and variability controls.
  - Augmentation strategy and rationale.
  - Model families and objective functions.
  - Evaluation metrics (MAE x-space, MSE, plots).
- Chapter 4:
  - Baseline experiment (experiment-base).
  - Augmentation ablations (experiment-better-augment).
  - Optimization sweeps (gamma/seed from experiment-commands).
  - Hard sample mining and edge-focus strategy.
  - Regression branch (gen3) versus binned branch.
  - Random predictor baseline (worst-case notebook).
- Chapter 5:
  - Synthesis of findings.
  - Failure modes (tail, edge behaviors, memorization risks).
  - Practical lessons and limitations.
  - Future work.

## Ready-to-use thesis planning prompts for NotebookLM

You can ask NotebookLM prompts like:

- Build a chapter-by-chapter thesis outline from this repository context, prioritizing a strong methodology section.
- Derive 3 alternative research questions and map each to concrete experiments already in notebooks.
- Convert the experiment list into a formal evaluation matrix: hypothesis, variables, controls, metrics, expected outcomes.
- Produce a writing plan for Chapter 4 with subsection order that mirrors the actual code workflow.
- Identify where direct regression and binned regression should be contrasted and how to present fair comparison criteria.
- Suggest figures/tables to include, using available plot types in the notebooks.

## Candidate figures and tables for the thesis

Figures:

- Dataset format diagram (.mglyph contents and manifest).
- Example glyph rendering progression across x.
- Actual vs predicted scatter with y=x line.
- Loss vs x scatter (to reveal tails/hard zones).
- Sampling diagnostics from hard mining notebook.
- Baseline random predictor plots from worst-case notebook.

Tables:

- RunConfig parameter table (base experiment).
- Experiment matrix (notebook, what changed, why, expected effect).
- Model architecture summary (gen1/gen2/gen3/binned).
- Metrics summary across key runs.

## Suggested next practical steps in the repository

If you want this repository to be thesis-ready with minimal friction:

- Freeze one stable experiment notebook as the canonical reproducible pipeline.
- Resolve any legacy/stale imports in regression and visualization notebooks/modules.
- Add a single experiment index table (file + goal + status + key result) in README or a docs file.
- Align README deployment branch text with CI branch trigger.
- Export representative plots and collect them under a thesis-figures directory.

## File pointers used to build this context

Primary references inspected while writing this summary:

- README.md
- pyproject.toml
- convert-notebook.sh
- .env.template
- .github/workflows/thesis-pdf.yml
- src/mglyph_ml/dataset/export.py
- src/mglyph_ml/dataset/manifest.py
- src/mglyph_ml/dataset/utils.py
- src/mglyph_ml/dataset/glyph_dataset.py
- src/mglyph_ml/nn/glyph_regressor_binned.py
- src/mglyph_ml/nn/glyph_regressor_gen1.py
- src/mglyph_ml/nn/glyph_regressor_gen2.py
- src/mglyph_ml/nn/glyph_regressor_gen3.py
- src/mglyph_ml/nn/training.py
- src/mglyph_ml/nn/evaluation.py
- src/mglyph_ml/experiment/run_config.py
- src/mglyph_ml/experiment/README.md
- notebooks/build-dataset.ipynb
- notebooks/experiment-base.ipynb
- notebooks/experiment-better-augment.ipynb
- notebooks/experiment-hard-sample.ipynb
- notebooks/experiment-gap.ipynb
- notebooks/experiment-regression.ipynb
- notebooks/experiment-commands.ipynb
- notebooks/worst-case.ipynb
- test/test_glyph_regressor_binned.py
- test/README.md

---

If needed, this document can be split into separate NotebookLM inputs (Architecture, Experiments, and Thesis Plan) to improve retrieval quality for longer conversations.
