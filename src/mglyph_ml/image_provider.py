import zipfile
import decimal


class GlyphProvider:
    def __init__(self, archive_path: str):
        self.archive_path = archive_path
        self.archive = zipfile.ZipFile(self.archive_path, 'r')

    def get_glyph_bytes(self, label: decimal.Decimal) -> bytes:
        return self.archive.read(f"{label}.png")
