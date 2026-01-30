import math
import os
import random
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

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
        images: list[np.ndarray],  # (N, C, H, W), uint8
        labels: list[float],  # (N,), float32
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.__images = images
        self.__labels = labels

    def __len__(self) -> int:
        return len(self.__images)

    def __getitem__(self, index: int) -> GlyphSample:
        label = self.__labels[index]

        image_np = self.__preloaded_images[index]
        assert isinstance(image_np, np.ndarray)

        image_augmented: np.ndarray = self.__transform(image=image_np.copy())["image"]
        image_tensor = Tensor(image_augmented).permute(2, 0, 1)  # permute [H, W, C] -> [C, H, W]
        # TODO: this type of normalization is actually somehow worse than dividing by 255.0
        # normalize the image by dividing by 255.0
        # if self.__normalize:
        #     image_tensor /= 255.0
        return image_tensor, torch.tensor(label, dtype=torch.float32)

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
