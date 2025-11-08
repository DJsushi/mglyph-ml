import decimal
import zipfile
from io import BytesIO

from PIL import Image

from mglyph_ml.manifest_parsing import Manifest


class GlyphProvider:
    def __init__(self, archive_path: str):
        self.__archive: zipfile.ZipFile = zipfile.ZipFile(archive_path, "r")
        manifest = self.__archive.read("metadata.json")
        self.__manifest = Manifest.model_validate_json(manifest)

    def __get_glyph_path(self, label: decimal.Decimal) -> str:
        return self.__manifest.get_glyph_filename(label)

    def get_glyph_as_bytes(self, label: decimal.Decimal) -> bytes:
        return self.__archive.read(self.__get_glyph_path(label))

    def get_glyph_as_pil_image(self, label: decimal.Decimal) -> Image.Image:
        glyph_bytes = self.get_glyph_as_bytes(label)
        return Image.open(BytesIO(glyph_bytes))

    def close(self):
        self.__archive.close()
