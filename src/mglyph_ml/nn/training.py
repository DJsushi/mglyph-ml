import random

from clearml import Logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mglyph_ml.data.glyph_dataset import GlyphDataset
from mglyph_ml.glyph_importer import GlyphImporter


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
        inputs = inputs.to(device)
        labels = labels.to(device)
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


def train_model(
    model: nn.Module,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    device: str,
    criterion,
    optimizer,
    num_epochs: int = 10,
    early_stopping_threshold: float = 0.3,
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
        reset_test_transform: Whether to reset test dataset transform before evaluation (default: True)

    Returns:
        Tuple of (train_losses, train_errors, test_losses, test_errors) - all as lists
    """
    losses = []
    errors = []
    test_losses = []
    test_errors = []

    # Train for specified number of epochs
    for epoch in range(1, num_epochs + 1):
        loss, error = train_one_epoch(
            model, data_loader_train, device, criterion, optimizer
        )
        error *= 100.0  # Convert normalized error (0-1) to actual x units (0-100)
        losses.append(loss)
        errors.append(error)

        test_loss, test_error = evaluate_glyph_regressor(
            model, data_loader_test, device, criterion
        )
        test_error *= 100.0  # Convert normalized error (0-1) to actual x units (0-100)
        test_losses.append(test_loss)
        test_errors.append(test_error)

        if logger is not None:
            logger.report_scalar(
                title="Loss", series="Train", value=loss, iteration=epoch
            )
            logger.report_scalar(
                title="Loss",
                series="Test",
                value=test_loss,
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
                value=test_error,
                iteration=epoch,
            )

        # Early stopping: stop if error is good enough
        if test_error < early_stopping_threshold:
            print(
                f"Early stopping at epoch {epoch+1}: test error {test_error:.4f} x units is below threshold"
            )
            break

    return losses, errors, test_losses, test_errors


def evaluate_glyph_regressor(
    model: nn.Module, data_loader: DataLoader, device: str, criterion
) -> tuple[float, float]:
    """
    Takes a glyph regressor, temporarily disables gradient calculation, and calculates the average
    loss on the given dataset (DataLoader). Processes in batches on GPU for efficiency.

    Returns a tuple containing the average loss and average error (mean absolute error)
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate error as the average absolute difference (y_hat - y)
            error = torch.mean(torch.abs(outputs - labels)).item()

            running_loss += loss.item()
            running_error += error
            num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_error = running_error / num_batches if num_batches > 0 else 0.0
    model.train()  # Set model back to training mode
    return avg_loss, avg_error
