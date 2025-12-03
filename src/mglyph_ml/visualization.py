import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader

from mglyph_ml.data.glyph_dataset import GlyphDataset
from mglyph_ml.glyph_importer import GlyphImporter
from mglyph_ml.nn.training import evaluate_glyph_regressor, train_one_epoch


def visualize_test_predictions(
    model: nn.Module,
    importers: list[GlyphImporter],
    device: str,
    num_samples: int = 9,
    figsize: tuple[int, int] = (6, 6),
    augmentation_seed: int = 69,
) -> None:
    """
    Visualize model predictions on random test samples.

    Args:
        model: The trained regression model to evaluate
        importers: List of GlyphImporter objects for the test set
        device: Device to run inference on ('cuda' or 'cpu')
        num_samples: Number of samples to visualize (default: 9 for 3x3 grid)
        figsize: Figure size for the plot (default: (6, 6))
        augmentation_seed: Random seed for reproducible augmentation (default: 69)

    Displays:
        - Grid of test images with true values, predictions, and errors
        - Summary statistics of the predictions
    """
    model.eval()
    model = model.to(device)

    # Create datasets - one normalized for prediction, one unnormalized for display
    dataset_test = GlyphDataset(*importers, augmentation_seed=augmentation_seed)
    temp_dataset_viz = GlyphDataset(
        *importers, normalize=False, augmentation_seed=augmentation_seed
    )

    # Get random samples from test set
    temp_dataset_viz.reset_transform()
    sample_indices = random.sample(range(len(temp_dataset_viz)), num_samples)

    # Calculate grid dimensions
    ncols = 3
    nrows = (num_samples + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Flatten for easy indexing

    errors = []

    with torch.no_grad():
        for index, sample_idx in enumerate(sample_indices):
            # Get sample from non-normalized dataset for display
            img_tensor_display, true_label = temp_dataset_viz[sample_idx]

            # Get same sample from normalized dataset for prediction
            dataset_test.reset_transform()
            img_tensor_pred, _ = dataset_test[sample_idx]

            # Make prediction
            img_batch = img_tensor_pred.unsqueeze(0).to(device)
            pred_label = model(img_batch).item()

            # Convert to x units (0-100)
            true_x = true_label * 100
            pred_x = pred_label * 100
            error = abs(pred_x - true_x)
            errors.append(error)

            # Convert image for display
            img = (
                img_tensor_display.numpy().clip(0, 1).transpose(1, 2, 0)
            )  # [C, H, W] -> [H, W, C]

            # Display
            axes[index].imshow(img)
            axes[index].set_title(
                f"True: {true_x:.2f}\n" f"Pred: {pred_x:.2f}\n" f"Err: {error:.2f}",
                fontsize=9,
            )

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_samples(
    plot_title: str,
    dataset: GlyphDataset,
    num_samples: int = 9,
    figsize: tuple[int, int] = (6, 6),
) -> Figure:
    """
    Visualize random samples from the test dataset (without predictions).

    Args:
        plot_title: Title for the plot
        dataset: GlyphDataset to visualize samples from
        num_samples: Number of samples to visualize (default: 9 for 3x3 grid)
        figsize: Figure size for the plot (default: (6, 6))

    Returns:
        matplotlib.pyplot.Figure: The figure instance that can be displayed later

    Displays:
        - Grid of test images with their labels
    """
    # Get random samples
    dataset.reset_transform()
    sample_indices = random.sample(range(len(dataset)), num_samples)

    # Calculate grid dimensions
    ncols = 3
    nrows = (num_samples + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Flatten for easy indexing

    for index, sample_idx in enumerate(sample_indices):
        image, label = dataset[sample_idx]

        # Convert image for display
        img = image.numpy().clip(0, 1).transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]

        # Display the image
        axes[index].imshow(img)
        axes[index].set_title(f"{label:.2f}")

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    fig.suptitle(plot_title)
    fig.tight_layout()

    return fig
