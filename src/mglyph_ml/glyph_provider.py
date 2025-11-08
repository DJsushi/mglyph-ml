from decimal import Decimal
from functools import cached_property
import zipfile
from io import BytesIO

from PIL import Image

from mglyph_ml.manifest_parsing import Manifest, ManifestImage


class GlyphProvider:
    def __init__(self, archive_path: str) -> None:
        self.__archive: zipfile.ZipFile = zipfile.ZipFile(archive_path, "r")
        manifest = self.__archive.read("metadata.json")
        self.__manifest = Manifest.model_validate_json(manifest)

    def __get_glyph_path(self, label: Decimal) -> str:
        return self.__manifest.get_glyph_filename(label)

    def get_glyph_as_bytes(self, label: Decimal) -> bytes:
        return self.__archive.read(self.__get_glyph_path(label))

    def get_glyph_as_pil_image(self, label: Decimal) -> Image.Image:
        '''
        Outputs the image with the specified label as a PIL Image.

        If the image has a transparent background, it will be pasted onto a completely white image and the result
        will be returned.
        '''
        glyph_bytes = self.get_glyph_as_bytes(label)
        image = Image.open(BytesIO(glyph_bytes))
        
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")

        return image

    def get_glyph_at_index_as_pil_image(self, index: int) -> Image.Image:
        image: ManifestImage = self.__manifest.images[index]
        return self.get_glyph_as_pil_image(Decimal(image.x))

    @cached_property
    def glyph_size(self) -> tuple[int, int]:
        first_image = self.__manifest.images[0]
        pil_image = self.get_glyph_as_pil_image(Decimal(first_image.x))
        return pil_image.width, pil_image.height

    @cached_property
    def step_size(self) -> Decimal:
        print(len(self.__manifest.images))
        return Decimal(100) / Decimal(len(self.__manifest.images) - 1)

    @cached_property
    def size(self) -> int:
        return len(self.__manifest.images)

    def close(self) -> None:
        self.__archive.close()
