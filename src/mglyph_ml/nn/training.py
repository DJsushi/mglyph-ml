import time
from dataclasses import dataclass

import torch
from clearml import Logger
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler

from mglyph_ml.dataset.glyph_dataset import GlyphDataset
from mglyph_ml.nn.evaluation import evaluate_glyph_regressor
from mglyph_ml.visualization import visualize_samples


@dataclass
class Timings:
    data_loading: float
    h2d: float
    forward: float
    backward: float
    step: float
    batch_total: float


def train_one_epoch(
    model: nn.Module,
    train_data_loader: DataLoader,
    device: str,
    criterion,  # unfortunately, has no supertype... There's torch.nn.modules._Loss, but it's private
    optimizer,  # same as above
) -> tuple[float, float, Timings]:
    model.train()
    running_loss = 0.0
    running_error = 0.0
    num_batches = 0
    data_loading_time = 0.0
    h2d_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    step_time = 0.0
    total_batch_time = 0.0

    use_cuda = "cuda" in device and torch.cuda.is_available()
    h2d_start = h2d_end = fwd_start = fwd_end = bwd_start = bwd_end = step_start = step_end = None
    if use_cuda:
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)

    batch_start = time.time()
    for index, data in enumerate(train_data_loader):
        # Measure time spent loading/augmenting data
        data_load_end = time.time()
        data_loading_time += data_load_end - batch_start

        inputs, labels = data
        if use_cuda and h2d_start is not None:
            h2d_start.record()
        inputs: torch.Tensor = inputs.to(device, non_blocking=True)
        labels: torch.Tensor = labels.to(device, non_blocking=True)
        if use_cuda and h2d_end is not None:
            h2d_end.record()
        # Use FP32 for compute stability (prevents NaNs from pure FP16 training)
        inputs = inputs.float()
        labels = labels.float().view(-1, 1)

        optimizer.zero_grad()
        if use_cuda and fwd_start is not None:
            fwd_start.record()
        outputs: torch.Tensor = model(inputs)
        loss = criterion(outputs.float(), labels.float())
        if use_cuda and fwd_end is not None and bwd_start is not None:
            fwd_end.record()
            bwd_start.record()
        loss.backward()
        if use_cuda and bwd_end is not None and step_start is not None:
            bwd_end.record()
            step_start.record()
        optimizer.step()
        if use_cuda and step_end is not None:
            step_end.record()

        # Calculate accuracy as the average absolute difference (y_hat - y)
        error = torch.mean(torch.abs(outputs.float() - labels.float())).item()

        running_loss += loss.item()
        running_error += error
        num_batches += 1
        # Optionally print every N batches
        # if (i + 1) % 10 == 0:
        # print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {loss.item():.4f}")

        if use_cuda:
            torch.cuda.synchronize()
            if h2d_start is not None and h2d_end is not None:
                h2d_time += h2d_start.elapsed_time(h2d_end) / 1000.0
            if fwd_start is not None and fwd_end is not None:
                forward_time += fwd_start.elapsed_time(fwd_end) / 1000.0
            if bwd_start is not None and bwd_end is not None:
                backward_time += bwd_start.elapsed_time(bwd_end) / 1000.0
            if step_start is not None and step_end is not None:
                step_time += step_start.elapsed_time(step_end) / 1000.0

        total_batch_time += time.time() - batch_start
        batch_start = time.time()

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    avg_error = running_error / num_batches if num_batches > 0 else 0.0
    timing = Timings(
        data_loading=data_loading_time,
        h2d=h2d_time,
        forward=forward_time,
        backward=backward_time,
        step=step_time,
        batch_total=total_batch_time,
    )
    return avg_loss, avg_error, timing


def training_loop(
    model: nn.Module,
    dataset_train: GlyphDataset,
    dataset_gap: GlyphDataset,
    device: str,
    criterion,
    optimizer,
    num_epochs: int,
    logger: Logger | None = None,
    batch_size: int = 32,
    data_loader_num_workers: int = 0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Train a model with real-time visualization of training progress.

    Args:
        model: The neural network model to train
        dataset_train: Training dataset
        dataset_gap: Validation/gap dataset
        device: Device to train on ('cuda' or 'cpu')
        criterion: Loss function
        optimizer: Optimizer for training
        num_epochs: Maximum number of epochs to train
        logger: ClearML logger for monitoring (optional)
        batch_size: Batch size for dataloaders
        data_loader_num_workers: Number of workers for dataloaders
        generator: Random generator for reproducibility

    Returns:
        Tuple of (train_losses, train_errors, test_losses, test_errors) - all as lists
    """
    generator = torch.manual_seed(69)

    losses = []
    errors = []
    test_losses = []
    test_errors = []

    # prev_train_error = None
    # prev_test_error = None

    # Train for specified number of epochs
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # Create a new random subset of 10,000 samples for this epoch
        indices_train = torch.randperm(len(dataset_train), generator=generator)[:10_000].tolist()
        sampler_train = SubsetRandomSampler(indices_train, generator=generator)

        # Create dataloaders for this epoch
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=sampler_train,
            num_workers=data_loader_num_workers,
            pin_memory=True,
        )
        data_loader_gap = DataLoader(
            dataset_gap,
            batch_size=batch_size,
            num_workers=data_loader_num_workers,
            pin_memory=True,
        )

        if logger is not None:
            fig1 = visualize_samples(plot_title="Training samples", data_loader=data_loader_train)
            logger.report_matplotlib_figure(
                title="Training samples", series="idk", figure=fig1, report_image=True, iteration=epoch
            )
            fig2 = visualize_samples(plot_title="Test samples", data_loader=data_loader_gap)
            logger.report_matplotlib_figure(
                title="Test samples", series="idk", figure=fig2, report_image=True, iteration=epoch
            )
            plt.close()

        loss_train, error_train, timing = train_one_epoch(
            model, data_loader_train, device, criterion, optimizer
        )
        losses.append(loss_train)
        errors.append(error_train)

        loss_gap, error_gap = evaluate_glyph_regressor(model, data_loader_gap, device, criterion)
        test_losses.append(loss_gap)
        test_errors.append(error_gap)

        epoch_time = time.time() - epoch_start_time

        error_x_train = error_train * 100.0
        error_x_gap = error_gap * 100.0
        print(
            f"Epoch {epoch}/{num_epochs} - Train (x units): {error_x_train:.2f} | "
            f"Gap (x units): {error_x_gap:.2f} | Time: {epoch_time:.1f}s | "
            f"Data load(total): {timing.data_loading:.1f}s | H2D(total): {timing.h2d:.1f}s | "
            f"Fwd(total): {timing.forward:.1f}s | Bwd(total): {timing.backward:.1f}s | "
            f"Step(total): {timing.step:.1f}s | Batch(total): {timing.batch_total:.1f}s"
        )

        if logger is not None:
            report_values(logger, epoch, loss_train, loss_gap, error_x_train, error_x_gap, epoch_time)

    return losses, errors, test_losses, test_errors


def report_values(
    logger: Logger,
    epoch: int,
    loss_train: float,
    loss_gap: float,
    error_x_train: float,
    error_x_gap: float,
    epoch_time: float,
):
    logger.report_scalar(title="Loss", series="Train", value=loss_train, iteration=epoch)
    logger.report_scalar(
        title="Loss",
        series="Gap",
        value=loss_gap,
        iteration=epoch,
    )
    logger.report_scalar(
        title="Error (x units)",
        series="Train",
        value=error_x_train,
        iteration=epoch,
    )
    logger.report_scalar(
        title="Error (x units)",
        series="Gap",
        value=error_x_gap,
        iteration=epoch,
    )
    logger.report_scalar(
        title="Epoch Time (s)",
        series="Time",
        value=epoch_time,
        iteration=epoch,
    )
