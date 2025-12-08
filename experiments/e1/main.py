from decimal import Decimal
from enum import Enum
import random

import mglyph as mg
import numpy as np
import torch
from clearml import Task, TaskTypes
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from prepare_data import prepare_data
from mglyph_ml.dataset.export import create_dataset
from mglyph_ml.dataset.manifest import ManifestSample
import mglyph_ml.lib as lib
from mglyph_ml.dataset.glyph_dataset import GlyphDataset
from mglyph_ml.nn.glyph_regressor_gen2 import GlyphRegressor
from mglyph_ml.nn.training import train_model
from mglyph_ml.visualization import visualize_samples
from clearml.automation.controller import PipelineDecorator
from clearml.automation.optimization import HyperParameterOptimizer


@PipelineDecorator.pipeline(name="Pipeline 1", project="Project 1", version="0.0.1")
def main():
    dataset_path = prepare_data(dataset_name="Dataset 1", start_x=30.0, end_x=60.0, seed=420)
    print(f"The dataset path is {dataset_path}")


def not_yet_in_pipeline():
    # HYPERPARAMETERS
    params = {
        "start_x": 40.0,  # where the training dataset should end and test dataset should begin
        "end_x": 60.0,  # where the test dataset should end and training dataset should begin
        "quick": False,  # whether to speedrun the training for testing purposes
        "seed": 420,
        "max_iterations": 50,
        "max_augment_rotation_degrees": 5,
        "max_augment_translation_percent": 0.05,
    }


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)


    dataset_train: Dataset = GlyphDataset(
        path="data/experiment-1.dataset",
        split="train",
        augmentation_seed=params["seed"],
        max_augment_rotation_degrees=params["max_augment_rotation_degrees"],
        max_augment_translation_percent=params["max_augment_translation_percent"],
    )
    dataset_test: Dataset = GlyphDataset(path="data/experiment-1.dataset", split="test", augment=False)

    if params["quick"]:
        indices_debug = list(range(0, len(dataset_train), 16))
        dataset_train = Subset(dataset_train, indices_debug)

    # Create a seeded generator for reproducible shuffling
    train_generator = torch.Generator()
    train_generator.manual_seed(params["seed"])

    data_loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, generator=train_generator)
    data_loader_test = DataLoader(dataset_test, batch_size=128)

    model = GlyphRegressor()
    model = model.to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model with visualization
    losses, errors, test_losses, test_errors = train_model(
        model=model,
        data_loader_train=data_loader_train,
        data_loader_test=data_loader_test,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=params["max_iterations"],
        early_stopping_threshold=0.3,
        logger=logger,
    )

if __name__ == "__main__":
    PipelineDecorator.run_locally()
    main()
