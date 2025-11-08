from typing import Callable

import mglyph as mg


def export_glyph(
    drawer: Callable[[float, mg.Canvas], None],
    name: str,
    glyph_set: str,
    xvalues: list[float] = [x / 1000 * 100 for x in range(1001)],
):
    mg.export(
        drawer,  # type: ignore
        name=name,
        short_name=name,
        version="1.0.0",
        path=f"data/glyphs-{glyph_set}/{name}.mglyph",
        xvalues=xvalues,
    )
