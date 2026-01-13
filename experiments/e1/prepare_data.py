from decimal import Decimal


def prepare_data(dataset_name: str, start_x: float, end_x: float, seed: int):
    import random
    from pathlib import Path

    import mglyph as mg
    from clearml import Dataset
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
    
    # Generate training samples: 50 samples per x unit
    # Uniform distribution in [0, start_x) range
    if start_x > 0.0:
        num_samples_start = int(start_x * 50)
        train_start_samples = np_gen.uniform(0.0, start_x, num_samples_start)
    else:
        train_start_samples = np.array([])
    
    # Uniform distribution in (end_x, 100] range (only if end_x < 100)
    if end_x < 100.0:
        num_samples_end = int((100.0 - end_x) * 50)
        train_end_samples = np_gen.uniform(end_x, 100.0, num_samples_end)
    else:
        train_end_samples = np.array([])
    
    train_x_values = np.concatenate([train_start_samples, train_end_samples])
    
    for x in train_x_values:
        ds.add_sample(square, x, split="train", metadata={"shape": ManifestSampleShape.SQUARE})
        ds.add_sample(triangle, x, split="train", metadata={"shape": ManifestSampleShape.TRIANGLE})
        ds.add_sample(circle, x, split="train", metadata={"shape": ManifestSampleShape.CIRCLE})
    
    # Generate test samples: 50 samples per x unit, uniform distribution in [start_x, end_x] range
    if start_x < end_x:
        num_test_samples = int((end_x - start_x) * 50)
        test_x_values = np_gen.uniform(start_x, end_x, num_test_samples)
    else:
        test_x_values = np.array([])
    
    for x in test_x_values:
        ds.add_sample(square, x, split="test", metadata={"shape": ManifestSampleShape.SQUARE})
        ds.add_sample(triangle, x, split="test", metadata={"shape": ManifestSampleShape.TRIANGLE})
        ds.add_sample(circle, x, split="test", metadata={"shape": ManifestSampleShape.CIRCLE})

    path = Path(f"data/{dataset_name}.dataset")
    ds.export(path)

    # TODO: this might come in handy later in case I decide to add proper dataset functionality
    # if upload_to_clearml:
    #     project_name = dataset_project or "mglyph-ml"
    #     parent_list = [parent_dataset_id] if parent_dataset_id else None
    #     clearml_ds = Dataset.create(
    #         dataset_name=dataset_name,
    #         dataset_project=project_name,
    #         parent_datasets=parent_list,
    #     )
    #     clearml_ds.add_files(path)
    #     clearml_ds.upload()
    #     clearml_ds.finalize()
    #     return clearml_ds.id

    return path
