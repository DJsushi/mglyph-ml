import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import cv2
import numpy as np
from matplotlib import pyplot as plt

from mglyph_ml.dataset.glyph_dataset import GlyphDataset
from mglyph_ml.dataset.manifest import DatasetManifest


@dataclass
class LoadedSample:
    imid: str
    image: np.ndarray
    label: float
    metadata: dict


def load_split_samples(
    dataset_path: Path,
    split: str,
    indices_filter: Callable[[int], bool] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    desired_size: int | tuple[int, int] | None = None,
    keep_alpha: bool = False,
) -> list[LoadedSample]:
    """
    Loads all the images and labels from the dataset from a certain specified split. It also supports specifying
    a desired size of the loaded images, for faster training. The `indices_filter` function is used to specify
    a predicate that filters out unwanted indices.
    """
    assert not (seed is not None and shuffle == False)

    if indices_filter is None:
        indices_filter = lambda _: True

    # loading everything... this cell takes the longest time
    # Load the entire zip file into memory
    with open(dataset_path, "rb") as f:
        temp_archive = ZipFile(BytesIO(f.read()))

    manifest_data = temp_archive.read("manifest.json")
    manifest = DatasetManifest.model_validate_json(manifest_data)

    samples = manifest.samples[split]

    indices = [i for i in range(len(samples)) if indices_filter(i)]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    samples = [samples[i] for i in indices]

    if isinstance(desired_size, int):
        max_width = desired_size
        max_height = desired_size
    elif desired_size is not None:
        max_width, max_height = desired_size
    else:
        max_width = None
        max_height = None

    # Load all images from memory using OpenCV (faster than PIL, directly to numpy)
    imread_flag = cv2.IMREAD_UNCHANGED if keep_alpha else cv2.IMREAD_COLOR

    def load_image_cv2(sample):
        img_bytes = temp_archive.read(sample.filename)
        img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), imread_flag)

        if img_array is None:
            raise ValueError(f"Failed to decode image: {sample.filename}")

        if keep_alpha:
            if img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        if max_width is not None and max_height is not None:
            height, width = img_array.shape[:2]

            # Resize each axis independently (no aspect ratio preservation).
            if height > max_height or width > max_width:
                new_width = min(width, max_width)
                new_height = min(height, max_height)
                img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return img_array

    with ThreadPoolExecutor(max_workers=32) as executor:
        images = list(executor.map(load_image_cv2, samples))

    logging.debug(f'Sample count in split before filtering "{split}": {len(samples)}')
    logging.debug(f'Sample count in split after filtering"{split}": {len(indices)}')

    temp_archive.close()

    loaded_samples = [
        LoadedSample(
            imid=sample.filename,
            image=image,
            label=sample.x,
            metadata=sample.metadata,
        )
        for sample, image in zip(samples, images)
    ]

    return loaded_samples


def show_datasets(
    *datasets: GlyphDataset,
    n_samples: int = 6,
    x_range: tuple[float, float] = (0.0, 100.0),
) -> None:
    """
    Given any amount of GlyphDatasets, it draws a plot containing `n_samples` samples from each
    of the datasets. If `x_range` is provided, only samples whose labels fall inside the range are shown.
    """
    assert len(datasets) >= 1

    x_min, x_max = x_range
    if x_min > x_max:
        raise ValueError("x_range must be ordered as (min, max)")

    _, axes = plt.subplots(len(datasets), n_samples, figsize=(2 * n_samples, 2 * len(datasets) + 1))
    axes = np.atleast_2d(axes)

    for row, dataset in enumerate(datasets):
        dataset_name = getattr(dataset, "name", f"Dataset {row + 1}")

        # indices_in_range = [
        #     index
        #     for index in range(len(dataset))
        #     if x_min <= float(dataset[index][1]) <= x_max
        # ]
        indices_in_range = range(len(dataset))

        if not indices_in_range:
            raise ValueError(f"No samples from {dataset_name} fall within x_range={x_range}")

        if len(indices_in_range) >= n_samples:
            sample_positions = np.linspace(0, len(indices_in_range) - 1, n_samples, dtype=int)
            selected_indices = [indices_in_range[position] for position in sample_positions]
        else:
            selected_indices = indices_in_range

        for i in range(n_samples):
            if i < len(selected_indices):
                img_tensor, label = dataset[selected_indices[i]]
            else:
                axes[row, i].axis("off")
                continue

            img = img_tensor.permute(1, 2, 0).numpy()

            axes[row, i].imshow(img)
            axes[row, i].set_title(f"{label:.5f}")
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

        axes[row, 0].set_ylabel(dataset_name, rotation=90, fontsize=11, labelpad=12)

    plt.tight_layout()
    plt.show()
