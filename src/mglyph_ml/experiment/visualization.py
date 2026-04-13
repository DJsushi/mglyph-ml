import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def show_truth_vs_pred_graph(
    n_samples: int,
    model,
    dataset,
    device: str,
    seed: int | None = None,
    ax=None,
):
    """Plot ground-truth vs predicted x and return (fig, ax).
    The caller controls rendering and closing.
    """
    model.eval()

    n_samples = min(n_samples, len(dataset))
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    rng = random.Random(seed) if seed is not None else random
    sample_indices = rng.sample(range(len(dataset)), n_samples)

    x_true = []
    x_pred = []

    with torch.inference_mode():
        for idx in sample_indices:
            img_tensor, label = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).float().to(device, non_blocking=True)
            label_value = float(label)

            logits = model(img_tensor)
            pred_value = float(model.logits_to_labels(logits).squeeze(0).item())

            x_true.append(label_value)
            x_pred.append(pred_value)

    x_true = np.array(x_true)
    x_pred = np.array(x_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.scatter(x_true, x_pred, alpha=0.5, s=12)

    min_x = float(min(np.min(x_true), np.min(x_pred)))
    max_x = float(max(np.max(x_true), np.max(x_pred)))
    ax.plot(
        [min_x, max_x],
        [min_x, max_x],
        linestyle="--",
        linewidth=1.2,
        color="black",
        label="Ideal (y=x)",
    )

    ax.set_xlabel("Ground truth x")
    ax.set_ylabel("Predicted x")
    ax.set_title(f"Ground Truth vs Predicted x (random {n_samples} training samples)")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    return fig, ax


def show_loss_vs_x_graph(
    n_samples: int,
    model,
    dataset,
    device: str,
    loss_fn,
    seed: int | None = None,
    ax=None,
):
    """Plot per-sample loss against true x and return plot objects + summary metrics.
    The caller controls rendering, logging, and closing.
    """
    model.eval()

    n_samples = min(n_samples, len(dataset))
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    rng = random.Random(seed) if seed is not None else random
    sample_indices = rng.sample(range(len(dataset)), n_samples)

    x_vals = []
    losses_per_sample = []

    with torch.inference_mode():
        for idx in sample_indices:
            img_tensor, label = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).float().to(device, non_blocking=True)
            label_value = float(label)

            logits = model(img_tensor)
            pred = model.logits_to_labels(logits).squeeze(0)

            label_tensor = torch.tensor([label_value], dtype=torch.float32, device=device)
            pred_tensor = pred.view(1)
            loss = loss_fn(pred_tensor, label_tensor).item()

            x_vals.append(label_value)
            losses_per_sample.append(loss)

    x_vals = np.array(x_vals)
    losses_per_sample = np.array(losses_per_sample)

    worst_5_losses = np.sort(losses_per_sample)[-5:] if len(losses_per_sample) >= 5 else losses_per_sample
    worst_5_loss_avg = float(np.mean(worst_5_losses))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    ax.scatter(x_vals, losses_per_sample, alpha=0.5, s=12)
    ax.set_xlabel("Actual label (raw x)")
    ax.set_ylabel("Loss (training objective)")
    ax.set_title(f"Loss vs. x (random {n_samples} training samples)")
    fig.tight_layout()

    return fig, ax, worst_5_loss_avg, worst_5_losses
