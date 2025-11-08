from decimal import Decimal

import pytest

from mglyph_ml.glyph_importer import GlyphImporter


@pytest.fixture
def glyph_provider():
    provider = GlyphImporter("test/data/square.mglyph")
    yield provider
    provider.close()

def test_get_glyph_bytes(glyph_provider: GlyphImporter):
    label = Decimal(1.0)
    glyph_bytes = glyph_provider.get_glyph_as_bytes(label)
    assert isinstance(glyph_bytes, bytes)
    assert len(glyph_bytes) > 0

def test_get_glyph_as_pil_image(glyph_provider: GlyphImporter):
    label = Decimal(1.0)
    image = glyph_provider.get_glyph_as_pil_image(label)
    assert image is not None
    assert image.width == 512 and image.height == 512
    assert image.mode in ["RGB"]

def test_glyph_size(glyph_provider: GlyphImporter):
    size = glyph_provider.glyph_size
    assert size == (512, 512)

def test_white_background_where_transparency_was(glyph_provider: GlyphImporter) -> None:
    image = glyph_provider.get_glyph_at_index_as_pil_image(0)
    assert image.mode == "RGB"
    pixel = image.getpixel((0, 0))
    assert pixel == (255, 255, 255)
