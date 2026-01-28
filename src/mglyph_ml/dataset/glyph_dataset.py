from copy import deepcopy
from functools import cached_property
import math
import os
from pathlib import Path
import random
from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO
from typing import Literal, Union
import zipfile

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image

from mglyph_ml.dataset.manifest import DatasetManifest, ManifestSample

type GlyphSample = tuple[Tensor, Tensor]


class GlyphDataset(Dataset):
    """
    This class is used to feed glyphs into the neural network.

    It has been designed in a way to provide a variety of datasets so that it's very easy for the user
    of the dataset to only include glyphs that they really want in the dataset.

    All glyphs are eagerly loaded into RAM at construction time so samples never hit disk during
    training. Use smaller splits or more memory if the archive is very large.
    """

    def __init__(
        self,
        path: str | Path,
        split: str,
        augment: bool = True,
        normalize: bool = True,
        augmentation_seed: int | None = None,
        max_augment_rotation_degrees: float = 5.0,
        max_augment_translation_percent: float = 0.05,
        preload_format: Literal["decoded", "encoded"] = "decoded",
    ):
        self.__path = Path(path) if isinstance(path, str) else path
        self.__max_augment_rotation_degrees = max_augment_rotation_degrees
        self.__max_augment_translation_percent = max_augment_translation_percent
        self.__augmentation_seed = augmentation_seed

        if preload_format not in ("encoded", "decoded"):
            raise ValueError(f"preload_format must be 'encoded' or 'decoded', got '{preload_format}'")
        self.__preload_format = preload_format

        # Load manifest from a temporary archive just to get the metadata
        with zipfile.ZipFile(self.__path, "r") as temp_archive:
            manifest_data = temp_archive.read("manifest.json")
            self.__manifest = DatasetManifest.model_validate_json(manifest_data)

        # Select the appropriate samples based on split
        if split not in self.__manifest.samples:
            available_splits = list(self.__manifest.samples.keys())
            raise ValueError(f"Invalid split: '{split}'. Available splits: {available_splits}")

        self.__samples = self.__manifest.samples[split]
        self.__preloaded_images: list[np.ndarray | bytes] = []
        self.__labels: list[float] = []
        self.__preload_data()

        self.__set_up_augmentation(augment, normalize)

    def __len__(self) -> int:
        return len(self.__samples)

    def __getitem__(self, index: int) -> GlyphSample:
        sample = self.__samples[index]
        label = self.__labels[index]

        if self.__preload_format == "encoded":
            image_bytes = self.__preloaded_images[index]
            assert isinstance(image_bytes, bytes)
            image_np = self.__decode_image_bytes(image_bytes)
        else:
            image_np = self.__preloaded_images[index]
            assert isinstance(image_np, np.ndarray)

        image_augmented: np.ndarray = self.__transform(image=image_np.copy())["image"]
        image_tensor = Tensor(image_augmented).permute(2, 0, 1)  # permute [H, W, C] -> [C, H, W]
        # TODO: this type of normalization is actually somehow worse than dividing by 255.0
        # normalize the image by dividing by 255.0
        # if self.__normalize:
        #     image_tensor /= 255.0
        return image_tensor, torch.tensor(label, dtype=torch.float32)

    def __preload_data(self) -> None:
        with zipfile.ZipFile(self.__path, "r") as archive:
            for sample in self.__samples:
                image_bytes = archive.read(sample.filename)

                if self.__preload_format == "encoded":
                    self.__preloaded_images.append(image_bytes)
                else:
                    image_np = self.__decode_image_bytes(image_bytes)
                    self.__preloaded_images.append(image_np)

                label = Decimal(sample.x) / Decimal(100.0)
                self.__labels.append(float(label))

    def __decode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decode an image from raw bytes into an RGB numpy array."""
        image = Image.open(BytesIO(image_bytes))

        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")

        return np.asarray(image)

    def get_random_samples(self, n: int) -> list[GlyphSample]:
        indices = random.sample(range(len(self)), n)
        return [self[index] for index in indices]

    @cached_property
    def glyph_size(self) -> tuple[int, int]:
        """Get the size of glyphs in this dataset."""
        if len(self.__samples) == 0:
            raise ValueError("Dataset is empty")

        if self.__preload_format == "encoded":
            image_bytes = self.__preloaded_images[0]
            assert isinstance(image_bytes, bytes)
            image_np = self.__decode_image_bytes(image_bytes)
            return image_np.shape[1], image_np.shape[0]
        else:
            first_image = self.__preloaded_images[0]
            assert isinstance(first_image, np.ndarray)
            return first_image.shape[1], first_image.shape[0]

    def close(self) -> None:
        """Release cached images and labels."""
        self.__preloaded_images = []
        self.__labels = []

    def __set_up_augmentation(self, augment: bool, normalize: bool) -> None:
        step1 = A.Affine(
            rotate=(
                -self.__max_augment_rotation_degrees,
                self.__max_augment_rotation_degrees,
            ),
            translate_percent=(
                -self.__max_augment_translation_percent,
                self.__max_augment_translation_percent,
            ),
            fit_output=False,
            keep_ratio=True,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            p=float(augment),
        )
        step2 = A.Normalize(normalization="min_max", p=float(normalize))
        self.__original_transform = A.Compose(
            [step1, step2], seed=self.__augmentation_seed  # temporarily removed step2
        )
        self.reset_transform()

    def reset_transform(self):
        self.__transform = deepcopy(self.__original_transform)
