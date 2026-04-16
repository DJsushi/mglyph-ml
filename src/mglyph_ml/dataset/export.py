"""
Logic for dataset creation and exporting.
"""

import dataclasses
import json
import random
import sys
import zipfile
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import numpy as np
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
    x: float
    metadata: dict
    split: str


class _DatasetBuilder:
    def __init__(self, name: str, creation_time: datetime):
        self._name: str = name
        self._creation_time: datetime = creation_time
        self._samples: list[_Sample] = []
        self._number_of_samples: int = 0

    def add_sample(self, drawer: Drawer, x: float, split: str, metadata: dict = {}):
        """Add a sample to the specified split (e.g., 'train', 'test', 'val')."""
        self._samples.append(_Sample(drawer, x, metadata, split))

    def export(
        self,
        path: Path,
        canvas_parameters: CanvasParameters = CanvasParameters(canvas_round_corner=False),
    ) -> Optional[BytesIO]:
        global_id = 0
        order = len(str(len(self._samples) - 1))

        # Group samples by split
        samples_by_split: dict[str, list[ManifestSample]] = {}
        all_manifest_samples = []

        for sample in self._samples:
            manifest_sample = ManifestSample(
                x=sample.x,
                filename=f"{global_id:0{order}d}.png",
                metadata=sample.metadata,
            )

            if sample.split not in samples_by_split:
                samples_by_split[sample.split] = []
            samples_by_split[sample.split].append(manifest_sample)
            all_manifest_samples.append(manifest_sample)
            global_id += 1

        manifest = DatasetManifest(
            name=self._name,
            creation_time=self._creation_time,
            samples=samples_by_split,
        )

        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("manifest.json", manifest.model_dump_json(indent=2))

            for sample, manifest_sample in zip(
                self._samples,
                all_manifest_samples,
            ):
                image = render(sample.drawer, (512, 512), float(sample.x), canvas_parameters, compress="pil")  # type: ignore
                data = BytesIO()
                image["pil"].save(data, format="PNG", compress_level=5)  # type: ignore
                data.seek(0)
                zf.writestr(manifest_sample.filename, data.read())

        zip_buffer.seek(0)
        if path is not None:
            with open(f"{path}", "wb") as f:
                f.write(zip_buffer.getvalue())

        if path is None:
            return zip_buffer


def create_dataset(name: str, creation_time: datetime = datetime.now()) -> _DatasetBuilder:
    return _DatasetBuilder(name=name, creation_time=creation_time)


def export_dataset(
    name: str, path: Path, drawer: Drawer, seed: int | None = None, n_samples: int = 10_000
) -> None:
    # Ensure the output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file so generation is always fresh and deterministic.
    if path.exists():
        path.unlink()
        print(f"Removed existing {path}")

    np_gen = np.random.default_rng(seed)
    random.seed(seed)

    dataset = create_dataset(name=name)

    # we subtract 2 because we manually add 2 extra samples (0.0 and 100.0)
    xvalues_train = np_gen.uniform(0.0, 100.0, n_samples - 2)
    xvalues_train = np.append(xvalues_train, [0.0, 100.0])
    xvalues_train.sort()

    xvalues_test = np_gen.uniform(0.0, 100.0, n_samples - 2)
    xvalues_test = np.append(xvalues_test, [0.0, 100.0])
    xvalues_test.sort()

    for x in xvalues_train:
        dataset.add_sample(drawer, x, split="0")

    for x in xvalues_test:
        dataset.add_sample(drawer, x, split="1")

    dataset.export(path)
    print(f"Generated: {path}")
