from decimal import Decimal

import pytest

from mglyph_ml.image_provider import GlyphProvider


@pytest.fixture
def glyph_provider():
    provider = GlyphProvider("test/data/square.mglyph")
    yield provider
    provider.close()

def test_get_glyph_bytes(glyph_provider: GlyphProvider):
    label = Decimal(1.0)
    glyph_bytes = glyph_provider.get_glyph_as_bytes(label)
    assert isinstance(glyph_bytes, bytes)
    assert len(glyph_bytes) > 0

def test_get_glyph_as_pil_image(glyph_provider):
    label = 1.0
    image = glyph_provider.get_glyph_as_pil_image(label)
    assert image is not None
    assert image.format == "PNG"
    assert image.size[0] > 0 and image.size[1] > 0
    assert image.mode in ["RGB", "RGBA", "L"]

def test_glyph_size(glyph_provider):
    size = glyph_provider.glyph_size
    assert size == (512, 512)