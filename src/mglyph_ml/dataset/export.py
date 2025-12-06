"""
Logic for dataset creation and exporting.
"""

import json
import sys
import zipfile
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from typing import Callable

from mglyph import Canvas, CanvasParameters, render

from mglyph_ml.dataset.manifest import DatasetManifest, ManifestSample

# i envision the usage like this:
# dataset = create_dataset(name='', creation_time='', ...) # returns a DatasetBuilder
# dataset.add_training_data(square_drawer, )
# dataset.add_testing_data()
# dataset.export(path='')

Drawer = Callable[[float, Canvas], None]


class _DatasetBuilder:
    def __init__(self, name: str, creation_time: datetime):
        self._name: str = name
        self._creation_time: datetime = creation_time
        self._training_data: list[tuple[Drawer, list[Decimal]]] = []
        self._testing_data: list[tuple[Drawer, list[Decimal]]] = []
        self._number_of_samples: int = 0

    def add_training_data(self, drawer: Drawer, xvalues: list[Decimal]):
        self._training_data.append((drawer, xvalues))
        self._number_of_samples += len(xvalues)

    def add_testing_data(self, drawer: Drawer, xvalues: list[Decimal]):
        self._testing_data.append((drawer, xvalues))
        self._number_of_samples += len(xvalues)

    def export(
        self,
        path: str,
        canvas_parameters: CanvasParameters = CanvasParameters(
            canvas_round_corner=False
        ),
    ) -> None:
        global_id = 0
        order = len(str(self._number_of_samples - 1))

        train_samples = []
        for drawer, xvalues in self._training_data:
            for x in xvalues:
                train_samples.append(
                    ManifestSample(x=Decimal(x), filename=f"{global_id:0{order}d}.png")
                )
                global_id += 1
        test_samples = []
        for drawer, xvalues in self._testing_data:
            for x in xvalues:
                test_samples.append(
                    ManifestSample(x=Decimal(x), filename=f"{global_id:0{order}d}.png")
                )
                global_id += 1

        manifest = DatasetManifest(
            name=self._name,
            creation_time=self._creation_time,
            train_samples=train_samples,
            test_samples=test_samples,
        )

        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("manifest.json", manifest.model_dump_json(indent=2))

            # for index, x in enumerate(xvalues):
            #     image = render(drawer, resolution, x, canvas_parameters, compress="pil")
            #     data = BytesIO()
            #     image["pil"].save(data, format="PNG", compress_level=5)
            #     data.seek(0)
            #     zf.writestr(f"{index:0{number_of_digits}d}.png", data.read())
            #     if not silent:
            #         progress_bar.value = index + 1

        zip_buffer.seek(0)
        if path is not None:
            with open(f"{path}", "wb") as f:
                f.write(zip_buffer.getvalue())

        if path is None:
            return zip_buffer


def create_dataset(
    name: str, creation_time: datetime = datetime.now()
) -> _DatasetBuilder:
    return _DatasetBuilder(name=name, creation_time=creation_time)
