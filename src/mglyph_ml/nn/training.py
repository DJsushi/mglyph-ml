import random
import time

from clearml import Logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mglyph_ml.dataset.glyph_dataset import GlyphDataset
from mglyph_ml.dataset.glyph_importer import GlyphImporter
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


def train_model(
    model: nn.Module,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    device: str,
    criterion,
    optimizer,
    num_epochs: int = 10,
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
            fig2 = visualize_samples(plot_title="Test samples", dataset=data_loader_test.dataset)  # type: ignore
            logger.report_matplotlib_figure(
                title="Test samples", series="idk", figure=fig2, report_image=True, iteration=epoch
            )

        loss, error = train_one_epoch(model, data_loader_train, device, criterion, optimizer)

        error *= 100.0  # Convert normalized error (0-1) to actual x units (0-100)
        losses.append(loss)
        errors.append(error)

        test_loss, test_error = evaluate_glyph_regressor(model, data_loader_test, device, criterion)
        test_error *= 100.0  # Convert normalized error (0-1) to actual x units (0-100)
        test_losses.append(test_loss)
        test_errors.append(test_error)

        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch}/{num_epochs} - Train: {error:.2f} | Test: {test_error:.2f} | Time: {epoch_time:.1f}s")

        if logger is not None:
            logger.report_scalar(title="Loss", series="Train", value=loss, iteration=epoch)
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
            logger.report_scalar(
                title="Epoch Time (s)",
                series="Time",
                value=epoch_time,
                iteration=epoch,
            )
        
        # Early stopping: stop if both train and test errors improved by less than 0.1
        # if prev_train_error is not None and prev_test_error is not None:
        #     train_improvement = prev_train_error - error
        #     test_improvement = prev_test_error - test_error
            
        #     if train_improvement < 0.05 and test_improvement < 0.05:
        #         print(f"Early stopping at epoch {epoch}: minimal improvement "
        #               f"(train: {train_improvement:.3f}, test: {test_improvement:.3f} x units)")
        #         break
        
        # prev_train_error = error
        # prev_test_error = test_error

    return losses, errors, test_losses, test_errors


def evaluate_glyph_regressor(model: nn.Module, data_loader: DataLoader, device: str, criterion) -> tuple[float, float]:
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
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
