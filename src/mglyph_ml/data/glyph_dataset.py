from functools import cached_property
import math
import random
from dataclasses import dataclass
from decimal import Decimal

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from mglyph_ml.glyph_importer import GlyphImporter

type GlyphSample = tuple[Tensor, Tensor]


class GlyphDataset(Dataset):
    """
    This class is used to feed glyphs into the neural network.

    It has been designed in a way to provide a variety of datasets so that it's very easy for the user
    of the dataset to only include glyphs that they really want in the dataset.
    """

    def __init__(
        self,
        *glyph_importers: GlyphImporter,
        augment: bool = True,
        normalize: bool = True,
    ):
        self.__glyph_importers = glyph_importers
        self.__set_up_augmentation(augment, normalize)
        # calculate the number of samples just once and cache it
        self.__len = sum([importer.count for importer in self.__glyph_importers])
        self.__normalize = normalize

    def __len__(self) -> int:
        return self.__len

    def __getitem__(self, index: int) -> tuple:
        importer, start_index = self.__get_glyph_importer_based_on_index(index)
        index -= start_index
        label = importer.get_glyph_xvalue_by_index(index)
        label /= Decimal(100.0)
        image_pil = importer.get_glyph_at_index_as_pil_image(index)
        image_np = np.asarray(image_pil)  # outputs [H, W, C]
        image_augmented: np.ndarray = self.__transform(image=image_np)["image"]
        image_tensor = Tensor(image_augmented).permute(2, 0, 1)  # permute [H, W, C] -> [C, H, W]
        # TODO: this type of normalization is actually somehow worse than dividing by 255.0
        # normalize the image by dividing by 255.0
        # if self.__normalize:
        #     image_tensor /= 255.0
        return image_tensor, torch.tensor(float(label), dtype=torch.float32)
    
    def __get_glyph_importer_based_on_index(self, index: int) -> tuple[GlyphImporter, int]:
        cumulative_count = 0
        for importer in self.__glyph_importers:
            cumulative_count += importer.count
            if index < cumulative_count:
                return importer, cumulative_count
        raise IndexError("Index out of range")

    def get_random_samples(self, n: int) -> list[GlyphSample]:
        indices = random.sample(range(len(self)), n)
        return [self[index] for index in indices]

    def __set_up_augmentation(self, augment: bool, normalize: bool) -> None:
        step1 = A.Affine(
            rotate=(-15, 15),
            translate_percent=(-0.10, 0.10),
            fit_output=False,
            keep_ratio=True,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            p=float(augment),
        )
        step2 = A.Normalize(normalization="image", p=float(normalize))
        self.__transform = A.Compose([step1, step2])
