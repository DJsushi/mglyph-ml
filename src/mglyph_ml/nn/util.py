import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import nn

from mglyph_ml.dataset.glyph_dataset import GlyphDataset
from mglyph_ml.nn.glyph_regressor_binned import BinnedGlyphRegressor


def plot_actual_vs_predicted_labels(
    model: BinnedGlyphRegressor,
    dataset: GlyphDataset,
    device: str,
    sample_count: int = 1000,
    seed: int | None = None,
    title: str = "Actual vs Predicted (Binned Regression)",
) -> Figure:
    """Plot actual labels against model predictions for random samples from a dataset."""

    sample_count = min(sample_count, len(dataset))
    rng = random.Random(seed) if seed is not None else random
    sample_indices = rng.sample(range(len(dataset)), sample_count)

    was_training = model.training
    model.eval()
    actuals: list[float] = []
    predictions: list[float] = []

    with torch.inference_mode():
        for idx in sample_indices:
            img_tensor, label = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).float().to(device, non_blocking=True)

            logits = model(img_tensor)
            pred = model.logits_to_labels(logits).item()

            actuals.append(float(label))
            predictions.append(float(pred))

    actuals_array = np.array(actuals)
    predictions_array = np.array(predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actuals_array, predictions_array, alpha=0.6, s=20, label="Samples")

    lo = min(actuals_array.min(), predictions_array.min())
    hi = max(actuals_array.max(), predictions_array.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")

    ax.set_xlabel("Actual label (raw x)")
    ax.set_ylabel("Predicted label (raw x)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if was_training:
        model.train()

    return fig


def combine_training_progress_figures(
    figures: list[tuple[int, Figure]],
    ncols: int = 4,
    figsize_per_panel: tuple[float, float] = (4.5, 3.5),
    title: str = "Training Progress",
) -> Figure:
    """Combine captured matplotlib figures into one montage figure."""

    if not figures:
        raise ValueError("figures must not be empty")

    nrows = math.ceil(len(figures) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )
    axes_array = np.atleast_1d(axes).reshape(-1)

    for axis, (step, source_fig) in zip(axes_array, figures):
        canvas = FigureCanvasAgg(source_fig)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())
        axis.imshow(image)
        axis.set_title(f"Step {step}")
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in axes_array[len(figures):]:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
