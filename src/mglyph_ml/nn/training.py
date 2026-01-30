import time

import torch
from clearml import Logger
from torch import nn
from torch.utils.data import DataLoader

from mglyph_ml.nn.evaluation import evaluate_glyph_regressor
from mglyph_ml.visualization import visualize_samples


def train_one_epoch(
    model: nn.Module,
    train_data_loader: DataLoader,
    device: str,
    criterion,  # unfortunately, has no supertype... There's torch.nn.modules._Loss, but it's private
    optimizer,  # same as above
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_error = 0.0
    num_batches = 0

    for index, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels = labels.view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy as the average absolute difference (y_hat - y)
        error = torch.mean(torch.abs(outputs - labels)).item()

        running_loss += loss.item()
        running_error += error
        num_batches += 1
        # Optionally print every N batches
        # if (i + 1) % 10 == 0:
        # print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {loss.item():.4f}")

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_error = running_error / num_batches if num_batches > 0 else 0.0
    return avg_loss, avg_error


def training_loop(
    model: nn.Module,
    data_loader_train: DataLoader,
    data_loader_gap: DataLoader,
    device: str,
    criterion,
    optimizer,
    num_epochs: int,
    logger: Logger | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Train a model with real-time visualization of training progress.

    Args:
        model: The neural network model to train
        data_loader_train: DataLoader for training data
        data_loader_test: DataLoader for test/validation data
        device: Device to train on ('cuda' or 'cpu')
        criterion: Loss function
        optimizer: Optimizer for training
        num_epochs: Maximum number of epochs to train (default: 10)
        early_stopping_threshold: Stop training if test error drops below this value in x units (default: 0.3)

    Returns:
        Tuple of (train_losses, train_errors, test_losses, test_errors) - all as lists
    """
    losses = []
    errors = []
    test_losses = []
    test_errors = []

    # prev_train_error = None
    # prev_test_error = None

    # Train for specified number of epochs
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        if logger is not None:
            fig1 = visualize_samples(plot_title="Training samples", dataset=data_loader_train.dataset)  # type: ignore
            logger.report_matplotlib_figure(
                title="Training samples", series="idk", figure=fig1, report_image=True, iteration=epoch
            )
            fig2 = visualize_samples(plot_title="Test samples", dataset=data_loader_gap.dataset)  # type: ignore
            logger.report_matplotlib_figure(
                title="Test samples", series="idk", figure=fig2, report_image=True, iteration=epoch
            )

        loss, error = train_one_epoch(model, data_loader_train, device, criterion, optimizer)
        losses.append(loss)
        errors.append(error)

        gap_loss, gap_error = evaluate_glyph_regressor(model, data_loader_gap, device, criterion)
        test_losses.append(gap_loss)
        test_errors.append(gap_error)

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch}/{num_epochs} - Train: {error:.2f} | Test: {gap_error:.2f} | Time: {epoch_time:.1f}s"
        )

        if logger is not None:
            logger.report_scalar(title="Loss", series="Train", value=loss, iteration=epoch)
            logger.report_scalar(
                title="Loss",
                series="Test",
                value=gap_loss,
                iteration=epoch,
            )
            logger.report_scalar(
                title="Error (x units)",
                series="Train",
                value=error,
                iteration=epoch,
            )
            logger.report_scalar(
                title="Error (x units)",
                series="Test",
                value=gap_error,
                iteration=epoch,
            )
            logger.report_scalar(
                title="Epoch Time (s)",
                series="Time",
                value=epoch_time,
                iteration=epoch,
            )

    return losses, errors, test_losses, test_errors
