import random
from pathlib import Path

import numpy as np
import torch
from clearml import Task
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from mglyph_ml.nn.evaluation import evaluate_glyph_regressor
from mglyph_ml.nn.glyph_regressor_gen2 import GlyphRegressor
from mglyph_ml.nn.training import training_loop


def train_and_test_model(
    device: str,
    dataset_train: Dataset,
    dataset_gap: Dataset,
    dataset_test: Dataset,
    seed: int,
    data_loader_num_workers: int,
    batch_size: int,
    quick: bool,
    max_epochs: int,
    model_save_path: Path | None = None,
) -> None:
    print(f"Device used for training: {device}")

    # Ensure reproducible weight initialization and dataloader shuffling
    random.seed(seed)
    np.random.seed(seed)
    generator = torch.manual_seed(seed)

    logger = Task.current_task().logger

    if quick:
        indices_debug = list(range(0, len(dataset_train), 32))
        dataset_train = Subset(dataset_train, indices_debug)
        indices_debug = list(range(0, len(dataset_gap), 32))
        dataset_gap = Subset(dataset_gap, indices_debug)
        indices_debug = list(range(0, len(dataset_test), 32))
        dataset_test = Subset(dataset_test, indices_debug)

    # Use multiple workers for faster data loading and pin_memory for faster GPU transfer
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=data_loader_num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
    )
    data_loader_gap = DataLoader(
        dataset_gap,
        batch_size=batch_size,
        num_workers=data_loader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=data_loader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = GlyphRegressor()
    model = model.to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model with visualization using validation set
    training_loop(
        model=model,
        data_loader_train=data_loader_train,
        data_loader_gap=data_loader_gap,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=max_epochs,
        logger=logger,
    )

    # Final evaluation on held-out test set
    print("\nEvaluating on held-out test set...")

    _, test_error = evaluate_glyph_regressor(model, data_loader_test, device, criterion)
    test_error_x = test_error * 100.0
    logger.report_scalar(title="Final Test Error (x units)", series="Test", value=test_error_x, iteration=0)
    print(f"Final test error over the entire interval: {test_error_x:.2f} x units")

    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
