import random
from pathlib import Path

import numpy as np
import torch
from clearml import Task
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler

from mglyph_ml.dataset.glyph_dataset import GlyphDataset
from mglyph_ml.nn.evaluation import evaluate_glyph_regressor
from mglyph_ml.nn.glyph_regressor_gen2 import GlyphRegressor
from mglyph_ml.nn.training import training_loop


def train_and_test_model(
    device: str,
    dataset_train: GlyphDataset,
    dataset_gap: GlyphDataset,
    dataset_test: GlyphDataset,
    seed: int,
    data_loader_num_workers: int,
    batch_size: int,
    max_epochs: int,
    model_save_path: Path | None = None,
) -> None:
    print(f"Device used for training: {device}")

    # Ensure reproducible weight initialization and dataloader shuffling
    random.seed(seed)
    np.random.seed(seed)

    logger = Task.current_task().logger

    model = GlyphRegressor()
    model = model.to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model with visualization using validation set
    # Dataloaders are created inside the training loop to rotate training samples each epoch
    training_loop(
        model=model,
        dataset_train=dataset_train,
        dataset_gap=dataset_gap,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=max_epochs,
        logger=logger,
        batch_size=batch_size,
        data_loader_num_workers=data_loader_num_workers,
    )

    # Final evaluation on held-out test set
    print("\nEvaluating on held-out test set...")

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=data_loader_num_workers,
        pin_memory=True,
    )

    _, test_error = evaluate_glyph_regressor(model, data_loader_test, device, criterion)
    test_error_x = test_error * 100.0
    logger.report_scalar(title="Final Test Error (x units)", series="Test", value=test_error_x, iteration=0)
    print(f"Final test error over the entire interval: {test_error_x:.2f} x units")

    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
