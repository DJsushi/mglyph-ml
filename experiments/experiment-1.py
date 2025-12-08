from enum import Enum
import random

import mglyph as mg
import numpy as np
import torch
from clearml import Task
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from mglyph_ml.dataset.export import create_dataset
from mglyph_ml.dataset.manifest import ManifestSample
import mglyph_ml.lib as lib
from mglyph_ml.dataset.glyph_dataset import GlyphDataset
from mglyph_ml.nn.glyph_regressor_gen2 import GlyphRegressor
from mglyph_ml.nn.training import train_model
from mglyph_ml.visualization import visualize_samples

task: Task = Task.init(project_name="mglyph-ml", task_name="Experiment 1 - working", output_uri=True)
logger = task.get_logger()

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
task.connect(params)

np_gen = np.random.default_rng(params["seed"])
random.seed(params["seed"])


# first, we build the dataset for the experiment
def square(x: float, canvas: mg.Canvas):
    canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
    canvas.rect(canvas.top_left, canvas.bottom_right, color="purple")


def triangle(x: float, canvas: mg.Canvas):
    canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
    canvas.polygon([canvas.bottom_left, canvas.bottom_right, canvas.top_center], color="cyan")


def circle(x: float, canvas: mg.Canvas):
    canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
    canvas.circle(canvas.center, canvas.xsize / 2, color="yellow")


class ManifestSampleShape(Enum):
    SQUARE = "s"
    TRIANGLE = "t"
    CIRCLE = "c"


class ShapedManifestSample(ManifestSample):
    shape: ManifestSampleShape


ds = create_dataset(name="experiment-1")
for x in np.concatenate([np.linspace(0.0, params["start_x"], 400), np.linspace(params["end_x"], 100.0, 400)]):
    ds.add_sample(square, x, split="train", metadata={"shape": ManifestSampleShape.SQUARE})
    ds.add_sample(triangle, x, split="train", metadata={"shape": ManifestSampleShape.TRIANGLE})
    ds.add_sample(circle, x, split="train", metadata={"shape": ManifestSampleShape.CIRCLE})
for x in np.linspace(params["start_x"], params["end_x"], 200):
    ds.add_sample(square, x, split="test", metadata={"shape": ManifestSampleShape.SQUARE})
    ds.add_sample(triangle, x, split="test", metadata={"shape": ManifestSampleShape.TRIANGLE})
    ds.add_sample(circle, x, split="test", metadata={"shape": ManifestSampleShape.CIRCLE})
ds.export(path="data/experiment-1.dataset")


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
