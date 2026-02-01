import random
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

type GlyphSample = tuple[Tensor, Tensor]


class GlyphDataset(Dataset):
    """
    This class is used to feed glyphs into the neural network.
    """

    def __init__(
        self,
        images: list[np.ndarray],  # (N, C, H, W), uint8
        labels: list[float],  # (N,), float32
        transform: Callable[
            [np.ndarray], torch.Tensor
        ],  # Input: (H, W, C) uint8 [0, 255] -> Output: (C, H, W) float32 normalized
    ):
        assert all(img.dtype == np.uint8 for img in images), "All images must have dtype uint8"

        self.__images = images
        self.__labels = [label / 100.0 for label in labels]
        self.__transform = transform

    def __len__(self) -> int:
        return len(self.__images)

    def __getitem__(self, index: int) -> GlyphSample:
        image = self.__images[index]
        label = self.__labels[index]

        image_tensor = self.__transform(image)

        return image_tensor, torch.tensor(label)

    def get_random_samples(self, n: int) -> list[GlyphSample]:
        indices = random.sample(range(len(self)), n)
        return [self[index] for index in indices]
