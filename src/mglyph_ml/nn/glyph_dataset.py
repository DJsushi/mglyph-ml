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
        glyph_importer: GlyphImporter,
        augment: bool = True,
        normalize: bool = True,
    ):
        self.__glyph_provider = glyph_importer
        self.__set_up_augmentation(augment, normalize)

    def __len__(self) -> int:
        return self.__glyph_provider.count

    def __getitem__(self, index: int) -> tuple:
        label = self.__glyph_provider.get_glyph_xvalue_by_index(index)
        label /= Decimal(100.0)
        image_pil = self.__glyph_provider.get_glyph_at_index_as_pil_image(index)
        image_np = np.asarray(image_pil)
        image_augmented: np.ndarray = self.__transform(image=image_np)["image"]
        image_tensor = Tensor(image_augmented).permute(2, 1, 0)
        return image_tensor, torch.tensor(float(label), dtype=torch.float32)

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
