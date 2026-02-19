import random
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

type GlyphSample = tuple[Tensor, Tensor]


class GlyphDataset(Dataset):
    """
    This class is used to feed glyphs into the neural network.
    """

    def __init__(
        self,
        images: list[np.ndarray],  # PIL images
        labels: list[float],
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ):
        self.__images = images
        self.__labels = [label / 100.0 for label in labels]
        self.__transform = transform

    def __len__(self) -> int:
        return len(self.__images)

    def __getitem__(self, index: int) -> GlyphSample:
        image = self.__images[index]
        label = self.__labels[index]

        if self.__transform is not None:
            image_tensor = self.__transform(image)
        else:
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image_tensor, torch.tensor(label)

    def get_random_samples(self, n: int) -> list[GlyphSample]:
        indices = random.sample(range(len(self)), n)
        return [self[index] for index in indices]
