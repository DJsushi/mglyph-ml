from decimal import Decimal

from clearml import TaskTypes
from clearml.automation import PipelineDecorator


@PipelineDecorator.component(name="Prepare Dataset", cache=True, task_type=TaskTypes.data_processing.value)
def prepare_data(dataset_name: str, start_x: float, end_x: float, seed: int):
    import random
    from pathlib import Path

    import mglyph as mg
    import numpy as np
    from util import ManifestSampleShape

    from mglyph_ml.dataset.export import create_dataset

    # This might come in handy later when creating a randomized dataset
    np_gen = np.random.default_rng(seed)
    random.seed(seed)

    # first, we build the dataset for the experiment
    def square(x: float, canvas: mg.Canvas):
        canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
        canvas.rect(canvas.top_left, canvas.bottom_right, color="purple")

    def triangle(x: float, canvas: mg.Canvas):
        canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
        canvas.polygon([canvas.bottom_left, canvas.bottom_right, canvas.top_center], color="cyan")

    def circle(x: float, canvas: mg.Canvas):
        canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
        canvas.circle(canvas.center, canvas.xsize / 2, color="yellow")

    ds = create_dataset(name=dataset_name)
    for x in np.concatenate([np.linspace(0.0, start_x, 400), np.linspace(end_x, 100.0, 400)]):
        ds.add_sample(square, x, split="train", metadata={"shape": ManifestSampleShape.SQUARE})
        ds.add_sample(triangle, x, split="train", metadata={"shape": ManifestSampleShape.TRIANGLE})
        ds.add_sample(circle, x, split="train", metadata={"shape": ManifestSampleShape.CIRCLE})
    for x in np.linspace(start_x, end_x, 200):
        ds.add_sample(square, x, split="test", metadata={"shape": ManifestSampleShape.SQUARE})
        ds.add_sample(triangle, x, split="test", metadata={"shape": ManifestSampleShape.TRIANGLE})
        ds.add_sample(circle, x, split="test", metadata={"shape": ManifestSampleShape.CIRCLE})

    path = Path(f"data/{dataset_name}.dataset")
    ds.export(path)

    return path
