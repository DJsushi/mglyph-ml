import os
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from clearml import Task
from dotenv import load_dotenv

from mglyph_ml.dataset.glyph_dataset import GlyphDataset
from mglyph_ml.dataset.manifest import DatasetManifest
from mglyph_ml.experiment.e1.train_model import train_and_test_model
from mglyph_ml.experiment.e1.util import load_image_into_ndarray


@dataclass
class ExperimentConfig:
    task_name: str
    task_tag: str
    dataset_path: Path
    gap_start_x: float | None = None
    gap_end_x: float | None = None
    quick: bool = True
    seed: int = 420
    max_iterations: int = 2
    max_augment_rotation_degrees: float = 5.0
    max_augment_translation_percent: float = 0.05
    data_loader_num_workers: int = 32
    offline: bool = False


def run_experiment(config: ExperimentConfig) -> None:
    """Run a single experiment with the specified parameters."""

    Task.set_offline(config.offline)
    task: Task = Task.init(project_name="mglyph-ml", task_name=config.task_name, reuse_last_task_id=False)
    task.add_tags(config.task_tag)
    task.connect(config)

    with open(config.dataset_path, "rb") as f:
        temp_archive = ZipFile(BytesIO(f.read()))

    manifest_data = temp_archive.read("manifest.json")
    manifest = DatasetManifest.model_validate_json(manifest_data)

    samples_0 = manifest.samples["0"]  # this is where the training and validation data comes from
    samples_1 = manifest.samples["1"]  # this is where the test data comes from

    # Create index mappings for each subset
    indices_train = [
        i
        for i, sample in enumerate(samples_0)
        if sample.x < config.gap_start_x or sample.x >= config.gap_end_x
    ]
    indices_gap = [
        i
        for i, sample in enumerate(samples_0)
        if sample.x >= config.gap_start_x and sample.x < config.gap_end_x
    ]
    indices_test = list(range(len(samples_1)))

    random.shuffle(indices_train)
    random.shuffle(indices_gap)
    random.shuffle(indices_test)

    indices_train = indices_train[: len(indices_train)]
    indices_gap = indices_gap[: len(indices_gap)]
    indices_test = indices_test[: len(indices_test)]

    affine = A.Affine(
        rotate=(-5, 5),
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        fit_output=False,
        keep_ratio=True,
        border_mode=cv2.BORDER_CONSTANT,
        fill=255,
        p=1.0,
    )
    normalize = A.Normalize(normalization="min_max")
    to_tensor = ToTensorV2()
    pipeline = A.Compose([affine, normalize, to_tensor], seed=420)
    normalize_pipeline = A.Compose([normalize, to_tensor])

    def affine_and_normalize(image: np.ndarray) -> torch.Tensor:
        return pipeline(image=image)["image"]

    def just_normalize(image: np.ndarray) -> torch.Tensor:
        return normalize_pipeline(image=image)["image"]

    with ThreadPoolExecutor(max_workers=32) as executor:
        images_train = list(
            executor.map(
                lambda i: affine_and_normalize(load_image_into_ndarray(temp_archive, samples_0[i].filename)),
                indices_train,
            )
        )
        images_gap = list(
            executor.map(
                lambda i: just_normalize(load_image_into_ndarray(temp_archive, samples_0[i].filename)),
                indices_gap,
            )
        )
        images_test = list(
            executor.map(
                lambda i: just_normalize(load_image_into_ndarray(temp_archive, samples_1[i].filename)),
                indices_test,
            )
        )
        labels_train = list(executor.map(lambda i: samples_0[i].x, indices_train))
        labels_gap = list(executor.map(lambda i: samples_0[i].x, indices_gap))
        labels_test = list(executor.map(lambda i: samples_1[i].x, indices_test))

    temp_archive.close()

    dataset_train = GlyphDataset(images=images_train, labels=labels_train)
    dataset_gap = GlyphDataset(images=images_gap, labels=labels_gap)
    dataset_test = GlyphDataset(images=images_test, labels=labels_test)

    print(
        f"DATASET IMAGE COUNT | train={len(images_train)} | gap={len(images_gap)} | test={len(images_test)}"
    )

    device = os.environ["MGML_DEVICE"]

    try:
        train_and_test_model(
            device=device,
            dataset_train=dataset_train,
            dataset_gap=dataset_gap,
            dataset_test=dataset_test,
            seed=420,
            data_loader_num_workers=config.data_loader_num_workers,
            batch_size=256,
            max_epochs=config.max_iterations,
            model_save_path=Path("models/exp1.pt"),
        )
    finally:
        task.close()


if __name__ == "__main__":
    load_dotenv()

    config = ExperimentConfig(
        task_name="Experiment 1.2.1",
        task_tag="exp-1.2.1",
        dataset_path=Path("data/uni.mglyph"),
        gap_start_x=10.0,
        gap_end_x=90.0,
        quick=False,
        seed=420,
        max_iterations=10,
        max_augment_rotation_degrees=5,
        max_augment_translation_percent=10,
        data_loader_num_workers=8,
        offline=False,
    )

    run_experiment(config)
