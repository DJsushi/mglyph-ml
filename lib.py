from typing import Callable

import mglyph as mg


def export_glyph(drawer: Callable[[float, mg.Canvas], None], name: str, glyph_set: str):
    mg.export(drawer, name=name, short_name=name, version="1.0.0", path=f"data/glyphs-{glyph_set}/{name}.mglyph")
