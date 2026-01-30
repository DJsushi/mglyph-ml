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
        assert all(img.dtype == np.uint8 for img in images), "All images must have dtype uint8"

        self.__images = images
        self.__labels = labels
        self.__transform = transform

    def __len__(self) -> int:
        return len(self.__images)

    def __getitem__(self, index: int) -> GlyphSample:
        image = self.__images[index]
        label = self.__labels[index]

        if self.__transform is not None:
            image = self.__transform(image.copy())

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image_tensor, torch.tensor(label, dtype=torch.float32)

    def get_random_samples(self, n: int) -> list[GlyphSample]:
        indices = random.sample(range(len(self)), n)
        return [self[index] for index in indices]

    # @cached_property
    # def glyph_size(self) -> tuple[int, int]:
    #     """Get the size of glyphs in this dataset."""
    #     if len(self.__images) == 0:
    #         raise ValueError("Dataset is empty")

    #     first_image = self.__preloaded_images[0]
    #     assert isinstance(first_image, np.ndarray)
    #     return first_image.shape[1], first_image.shape[0]
