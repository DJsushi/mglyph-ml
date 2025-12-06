import random

import mglyph as mg
import numpy as np
import torch
from clearml import Task
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

import mglyph_ml.lib as lib
from mglyph_ml.dataset.glyph_dataset import GlyphDataset
from mglyph_ml.dataset.glyph_importer import GlyphImporter
from mglyph_ml.nn.glyph_regressor_gen2 import GlyphRegressor
from mglyph_ml.nn.training import train_model
from mglyph_ml.visualization import visualize_samples

task: Task = Task.init(
    project_name="mglyph-ml", task_name="Experiment 1 - all shapes - smaller augment", output_uri=True
)
logger = task.get_logger()

# HYPERPARAMETERS
params = {
    "start_x": 40.0,  # where the training dataset should end and test dataset should begin
    "end_x": 60.0,  # where the test dataset should end and training dataset should begin
    "quick": True,  # whether to speedrun the training for testing purposes
    "seed": 69,
    "max_iterations": 10,
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


lib.export_glyph(
    square,
    name="train-square",
    glyph_set="experiment-1",
    xvalues=list(np_gen.uniform(low=0.0, high=params["start_x"], size=400))
    + list(np_gen.uniform(low=params["end_x"], high=100.0, size=400)),
)
lib.export_glyph(
    square,
    name="test-square",
    glyph_set="experiment-1",
    xvalues=list(np_gen.uniform(low=params["start_x"], high=params["end_x"], size=200)),
)


def triangle(x: float, canvas: mg.Canvas):
    canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
    canvas.polygon(
        [canvas.bottom_left, canvas.bottom_right, canvas.top_center], color="cyan"
    )


lib.export_glyph(
    triangle,
    name="train-triangle",
    glyph_set="experiment-1",
    xvalues=list(np_gen.uniform(low=0.0, high=params["start_x"], size=400))
    + list(np_gen.uniform(low=params["end_x"], high=100.0, size=400)),
)
lib.export_glyph(
    triangle,
    name="test-triangle",
    glyph_set="experiment-1",
    xvalues=list(np_gen.uniform(low=params["start_x"], high=params["end_x"], size=200)),
)


def circle(x: float, canvas: mg.Canvas):
    canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
    canvas.circle(canvas.center, canvas.xsize / 2, color="yellow")


lib.export_glyph(
    circle,
    name="train-circle",
    glyph_set="experiment-1",
    xvalues=list(np_gen.uniform(low=0.0, high=params["start_x"], size=400))
    + list(np_gen.uniform(low=params["end_x"], high=100.0, size=400)),
)
lib.export_glyph(
    circle,
    name="test-circle",
    glyph_set="experiment-1",
    xvalues=list(np_gen.uniform(low=params["start_x"], high=params["end_x"], size=200)),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


glyphs_train = ["train-square.mglyph", "train-triangle.mglyph", "train-circle.mglyph"]
importers_train = [
    GlyphImporter(f"data/glyphs-experiment-1/{glyph}") for glyph in glyphs_train
]
dataset_train: Dataset = GlyphDataset(
    *importers_train,
    augmentation_seed=params["seed"],
    max_augment_rotation_degrees=params["max_augment_rotation_degrees"],
    max_augment_translation_percent=params["max_augment_translation_percent"],
)

glyphs_test = ["test-square.mglyph", "test-triangle.mglyph", "test-circle.mglyph"]
importers_test = [
    GlyphImporter(f"data/glyphs-experiment-1/{glyph}") for glyph in glyphs_test
]
dataset_test: Dataset = GlyphDataset(
    *importers_test, augment=False
)  # Changed from importers_train to importers_test


fig1 = visualize_samples(plot_title="Training samples", dataset=dataset_train)
logger.report_matplotlib_figure(
    title="Training samples", series="Beginning", figure=fig1, report_image=True
)
fig2 = visualize_samples(plot_title="Test samples", dataset=dataset_test)
logger.report_matplotlib_figure(
    title="Test samples", series="Beginning", figure=fig2, report_image=True
)

if params["quick"]:
    indices_debug = list(range(0, len(dataset_train), 16))
    dataset_train = Subset(dataset_train, indices_debug)

data_loader_train = DataLoader(dataset_train, batch_size=128)
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

print(losses)
print(errors)
print(test_losses)
print(test_errors)