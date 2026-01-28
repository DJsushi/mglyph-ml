import random
from pathlib import Path

import mglyph as mg
import numpy as np
from util import ManifestSampleShape

from mglyph_ml.dataset.export import create_dataset


def prepare_data(gap_start_x: float, gap_end_x: float, seed: int, dataset_name: str | None = None, samples_per_x: int = 50):
    # Generate dataset name from parameters if not provided
    if dataset_name is None:
        dataset_name = f"dataset_x{gap_start_x:.1f}-{gap_end_x:.1f}_seed{seed}"

    path = Path(f"data/{dataset_name}.dataset")

    # Check if dataset already exists
    if path.exists():
        print(f"Dataset '{dataset_name}' already exists at {path}, skipping creation.")
        return path

    print(f"Creating new dataset '{dataset_name}'...")

    # This might come in handy later when creating a randomized dataset
    np_gen = np.random.default_rng(seed)
    random.seed(seed)

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

    ds = create_dataset(name=dataset_name)

    # Generate training samples: samples_per_x samples per x unit
    # Uniform distribution in [0, start_x) range
    if gap_start_x > 0.0:
        num_samples_start = int(gap_start_x * samples_per_x)
        train_start_samples = np_gen.uniform(0.0, gap_start_x, num_samples_start)
    else:
        train_start_samples = np.array([])

    # Uniform distribution in (end_x, 100] range (only if end_x < 100)
    if gap_end_x < 100.0:
        num_samples_end = int((100.0 - gap_end_x) * samples_per_x)
        train_end_samples = np_gen.uniform(gap_end_x, 100.0, num_samples_end)
    else:
        train_end_samples = np.array([])

    train_x_values = np.concatenate([train_start_samples, train_end_samples])

    for x in train_x_values:
        ds.add_sample(square, x, split="train", metadata={"shape": ManifestSampleShape.SQUARE})
        ds.add_sample(triangle, x, split="train", metadata={"shape": ManifestSampleShape.TRIANGLE})
        ds.add_sample(circle, x, split="train", metadata={"shape": ManifestSampleShape.CIRCLE})

    # Generate validation and test samples: samples_per_x samples per x unit total, uniform distribution in [start_x, end_x] range
    # Split into 60% validation, 40% test
    if gap_start_x < gap_end_x:
        total_samples = int((gap_end_x - gap_start_x) * samples_per_x)
        num_val_samples = int(total_samples * 0.6)

        # Generate all samples first, then split
        all_gap_samples = np_gen.uniform(gap_start_x, gap_end_x, total_samples)
        val_x_values = all_gap_samples[:num_val_samples]
        test_x_values = all_gap_samples[num_val_samples:]
    else:
        val_x_values = np.array([])
        test_x_values = np.array([])

    # Add validation samples
    for x in val_x_values:
        ds.add_sample(square, x, split="val", metadata={"shape": ManifestSampleShape.SQUARE})
        ds.add_sample(triangle, x, split="val", metadata={"shape": ManifestSampleShape.TRIANGLE})
        ds.add_sample(circle, x, split="val", metadata={"shape": ManifestSampleShape.CIRCLE})

    # Add test samples
    for x in test_x_values:
        ds.add_sample(square, x, split="test", metadata={"shape": ManifestSampleShape.SQUARE})
        ds.add_sample(triangle, x, split="test", metadata={"shape": ManifestSampleShape.TRIANGLE})
        ds.add_sample(circle, x, split="test", metadata={"shape": ManifestSampleShape.CIRCLE})

    ds.export(path)

    # TODO: this might come in handy later in case I decide to add proper dataset functionality
    # if upload_to_clearml:
    #     project_name = dataset_project or "mglyph-ml"
    #     parent_list = [parent_dataset_id] if parent_dataset_id else None
    #     clearml_ds = Dataset.create(
    #         dataset_name=dataset_name,
    #         dataset_project=project_name,
    #         parent_datasets=parent_list,
    #     )
    #     clearml_ds.add_files(path)
    #     clearml_ds.upload()
    #     clearml_ds.finalize()
    #     return clearml_ds.id

    return path
