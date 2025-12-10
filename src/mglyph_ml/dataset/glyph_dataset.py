from copy import deepcopy
from functools import cached_property
import math
import os
from pathlib import Path
import random
from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO
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
    
    This dataset is multiprocessing-safe: each worker process will open its own ZIP file handle.
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
    ):
        self.__path = Path(path) if isinstance(path, str) else path
        self.__max_augment_rotation_degrees = max_augment_rotation_degrees
        self.__max_augment_translation_percent = max_augment_translation_percent
        self.__augmentation_seed = augmentation_seed
        
        # Don't open archive yet - will be opened lazily per worker process
        self.__archive = None
        self.__worker_id = None
        
        # Load manifest from a temporary archive just to get the metadata
        with zipfile.ZipFile(self.__path, "r") as temp_archive:
            manifest_data = temp_archive.read("manifest.json")
            self.__manifest = DatasetManifest.model_validate_json(manifest_data)
        
        # Select the appropriate samples based on split
        if split not in self.__manifest.samples:
            available_splits = list(self.__manifest.samples.keys())
            raise ValueError(f"Invalid split: '{split}'. Available splits: {available_splits}")
        
        self.__samples = self.__manifest.samples[split]
        
        self.__set_up_augmentation(augment, normalize)

    def __len__(self) -> int:
        return len(self.__samples)
    
    def _get_archive(self):
        """
        Get or create the ZIP archive handle for the current worker.
        Each worker process needs its own handle for thread-safety.
        """
        # Check if we're in a new worker process
        current_worker = torch.utils.data.get_worker_info()
        worker_id = current_worker.id if current_worker is not None else None
        
        # If worker changed or archive not yet opened, (re)open it
        if self.__archive is None or self.__worker_id != worker_id:
            if self.__archive is not None:
                self.__archive.close()
            self.__archive = zipfile.ZipFile(self.__path, "r")
            self.__worker_id = worker_id
        
        return self.__archive

    def __getitem__(self, index: int) -> tuple:
        sample = self.__samples[index]
        label = Decimal(sample.x) / Decimal(100.0)
        
        # Load image from archive
        image_pil = self.__get_image_as_pil(sample.filename)
        image_np = np.asarray(image_pil)  # outputs [H, W, C]
        image_augmented: np.ndarray = self.__transform(image=image_np)["image"]
        image_tensor = Tensor(image_augmented).permute(
            2, 0, 1
        )  # permute [H, W, C] -> [C, H, W]
        # TODO: this type of normalization is actually somehow worse than dividing by 255.0
        # normalize the image by dividing by 255.0
        # if self.__normalize:
        #     image_tensor /= 255.0
        return image_tensor, torch.tensor(float(label), dtype=torch.float32)

    def __get_image_as_pil(self, filename: str) -> Image.Image:
        """
        Loads an image from the archive as a PIL Image.
        
        If the image has a transparent background, it will be pasted onto a completely white image and the result
        will be returned.
        """
        archive = self._get_archive()
        image_bytes = archive.read(filename)
        image = Image.open(BytesIO(image_bytes))

        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")

        return image

    def get_random_samples(self, n: int) -> list[GlyphSample]:
        indices = random.sample(range(len(self)), n)
        return [self[index] for index in indices]
    
    @cached_property
    def glyph_size(self) -> tuple[int, int]:
        """Get the size of glyphs in this dataset."""
        if len(self.__samples) == 0:
            raise ValueError("Dataset is empty")
        first_sample = self.__samples[0]
        pil_image = self.__get_image_as_pil(first_sample.filename)
        return pil_image.width, pil_image.height
    
    def close(self) -> None:
        """Close the dataset archive."""
        if self.__archive is not None:
            self.__archive.close()
            self.__archive = None

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
