from decimal import Decimal
from functools import cached_property
import zipfile
from io import BytesIO

from PIL import Image

from mglyph_ml.manifest_parsing import Manifest


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
        glyph_bytes = self.get_glyph_as_bytes(label)
        return Image.open(BytesIO(glyph_bytes))
    
    @cached_property
    def glyph_size(self) -> tuple[int, int]:
        first_image = self.__manifest.images[0]
        pil_image = self.get_glyph_as_pil_image(Decimal(first_image.x))
        return pil_image.width, pil_image.height

    def close(self) -> None:
        self.__archive.close()
