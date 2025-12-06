"""
Logic for dataset creation and exporting.
"""

import dataclasses
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
# dataset.add_training_sample(drawer, x, optional_metadata)
# dataset.add_testing_sample(drawer, x, optional_metadata)
# dataset.export(path='')

type Drawer = Callable[[float, Canvas], None]


@dataclasses.dataclass
class _Sample:
    drawer: Drawer
    x: Decimal
    metadata: dict


class _DatasetBuilder:
    def __init__(self, name: str, creation_time: datetime):
        self._name: str = name
        self._creation_time: datetime = creation_time
        self._training_samples: list[_Sample] = []
        # self._testing_data: list[tuple[Drawer, list[Decimal]]] = []
        self._number_of_samples: int = 0

    def add_training_sample(self, drawer: Drawer, x: Decimal, metadata: dict = {}):
        self._training_samples.append(_Sample(drawer, x, metadata))

    # def add_testing_data(self, drawer: Drawer, xvalues: list[Decimal]):
    #     self._testing_data.append((drawer, xvalues))
    #     self._number_of_samples += len(xvalues)

    def export(
        self,
        path: str,
        canvas_parameters: CanvasParameters = CanvasParameters(
            canvas_round_corner=False
        ),
    ) -> None:
        global_id = 0
        order = len(str(len(self._training_samples) - 1))

        manifest_train_samples = []
        for sample in self._training_samples:
            manifest_train_samples.append(
                ManifestSample(x=sample.x, filename=f"{global_id:0{order}d}.png", metadata=sample.metadata)
            )
            global_id += 1
        # test_samples = []
        # for _, xvalues in self._testing_data:
        #     for x in xvalues:
        #         test_samples.append(
        #             ManifestSample(x=x, filename=f"{global_id:0{order}d}.png")
        #         )
        #         global_id += 1

        manifest = DatasetManifest(
            name=self._name,
            creation_time=self._creation_time,
            train_samples=manifest_train_samples,
            test_samples=[],
        )

        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("manifest.json", manifest.model_dump_json(indent=2))

            for sample, manifest_sample in zip(self._training_samples, manifest_train_samples): # + self._testing_samples
                image = render(sample.drawer, (512, 512), float(sample.x), canvas_parameters, compress="pil") # type: ignore
                data = BytesIO()
                image["pil"].save(data, format="PNG", compress_level=5) # type: ignore
                data.seek(0)
                zf.writestr(manifest_sample.filename, data.read())

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
