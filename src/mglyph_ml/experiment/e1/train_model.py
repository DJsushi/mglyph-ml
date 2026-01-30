from pathlib import Path


def train_model(
    dataset_path: Path,
    seed: int,
    max_augment_rotation_degrees: float,
    max_augment_translation_percent: float,
    quick: bool,
    max_iterations: int,
):
    import random

    import numpy as np
    import torch
    from clearml import Task
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, Subset

    from mglyph_ml.dataset.glyph_dataset import GlyphDataset
    from mglyph_ml.nn.glyph_regressor_gen2 import GlyphRegressor
    from mglyph_ml.nn.training import train_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device used for training: {device}")

    # Ensure reproducible weight initialization and dataloader shuffling
    random.seed(seed)
    np.random.seed(seed)
    generator = torch.manual_seed(seed)

    logger = Task.current_task().logger

    dataset_train: Dataset = GlyphDataset(
        path=dataset_path,
        split="uni",
        augmentation_seed=seed,
        max_augment_rotation_degrees=max_augment_rotation_degrees,
        max_augment_translation_percent=max_augment_translation_percent,
    )
    dataset_val: Dataset = GlyphDataset(path=dataset_path, split="val", augment=False)
    dataset_test: Dataset = GlyphDataset(path=dataset_path, split="test", augment=False)

    if quick:
        indices_debug = list(range(0, len(dataset_train), 16))
        dataset_train = Subset(dataset_train, indices_debug)

    # Use multiple workers for faster data loading and pin_memory for faster GPU transfer
    num_workers = 32  # Adjust based on your CPU cores
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=128,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
    )
    data_loader = DataLoader(
        dataset, batch_size=128, num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    data_loader_gap = DataLoader(
        dataset_gap, batch_size=128, num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    model = GlyphRegressor()
    model = model.to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model with visualization using validation set
    losses, errors, gap_losses, gap_errors = train_model(
        model=model,
        data_loader_train=data_loader_train,
        data_loader_test=data_loader_val,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=max_iterations,
        logger=logger,
    )

    # Final evaluation on held-out test set
    print("\nEvaluating on held-out test set...")
    from mglyph_ml.nn.training import evaluate_glyph_regressor

    test_loss, gap_error = evaluate_glyph_regressor(model, data_loader_gap, device, criterion)
    gap_error *= 100.0
    logger.report_scalar(title="Final Test Error (x units)", series="Test", value=gap_error, iteration=0)
    print(f"Final test error: {gap_error:.2f} x units")

    torch.save(model.state_dict(), "models/experiment-1.pt")

    return model
