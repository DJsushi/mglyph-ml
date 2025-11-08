from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from PIL import Image
from torch import Tensor
from mglyph_ml.glyph_provider import GlyphProvider
from torch.utils.data import Dataset
import cv2
import albumentations as A


@dataclass
class GlyphSample:
    image: Tensor
    label: Decimal


class GlyphDataset(Dataset):
    '''
    This class is used to feed glyphs into the neural network.

    It has been designed in a way to provide a variety of datasets so that it's very easy for the user
    of the dataset to only include glyphs that they really want in the dataset.
    '''
    def __init__(
        self,
        glyph_provider: GlyphProvider,
        augment: bool = True
    ):
        self.__glyph_provider = glyph_provider
        self.__set_up_augmentation(augment)

    def __len__(self) -> int:
        return self.__glyph_provider.size

    def __getitem__(self, index: int) -> GlyphSample:
        image_pil = self.__glyph_provider.get_glyph_at_index_as_pil_image(index)
        # If image has alpha channel, paste onto white background
        
        img_np = np.array(img_pil)  # H x W x C, dtype=uint8, values 0-255
        # Apply augmentation on HWC uint8 image
        if self.augment and self.transform is not None:
            augmented = self.transform(image=img_np)
            img_np = augmented["image"]
        # Convert to CHW float tensor in [0,1] for the model
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        # Optionally normalize to zero-mean/unit-std using mean/std (channels first)
        if self.normalize:
            img_tensor = (img_tensor - self.mean[:, None, None]) / self.std[
                :, None, None
            ]
        label = label / 100.0
        return img_tensor, label
        

    def __set_up_augmentation(self, augment: bool) -> None:
        self.augment: bool = augment
        if self.augment:
            self.transform = A.Compose(
                [
                    A.Affine(
                        rotate=(-5, 5),
                        translate_percent=(-0.20, 0.20),
                        fit_output=False,
                        keep_ratio=True,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=255,
                    ),
                ]
            )
        else:
            self.transform = None


    def __build_dataset(self, glyph_provider: GlyphProvider, ranges: list[tuple[Decimal, Decimal]]) -> Iterable[]:
        dataset = []
        for range_ in ranges:
            start = range_[0]
            end = range_[1]
            # iterate through all decimals from start to end
            



    def _ensure_archive(self):
        if self.archive is None:
            self.archive = zipfile.ZipFile(self.zip_path, "r")
