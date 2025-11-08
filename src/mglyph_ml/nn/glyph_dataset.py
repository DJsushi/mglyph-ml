from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from PIL import Image
from torch import Tensor
from mglyph_ml.glyph_importer import GlyphImporter
from torch.utils.data import Dataset
import cv2
import albumentations as A


@dataclass
class GlyphSample:
    image: Tensor
    label: Decimal


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
        return self.__glyph_provider.size

    def __getitem__(self, index: int) -> GlyphSample:
        label = self.__glyph_provider.get_glyph_xvalue_by_index(index)
        image_pil = self.__glyph_provider.get_glyph_at_index_as_pil_image(index)
        augmented = self.__transform(image_pil)["image"]
        image_tensor = Tensor(augmented)
        return GlyphSample(image=image_tensor, label=label)

    def __set_up_augmentation(self, augment: bool, normalize: bool) -> None:
        step1 = A.Affine(
            rotate=(-5, 5),
            translate_percent=(-0.20, 0.20),
            fit_output=False,
            keep_ratio=True,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            p=float(augment),
        )
        step2 = A.Compose([A.Normalize(normalization="image", p=float(normalize))])
        self.__transform = A.Compose([step1, step2])
