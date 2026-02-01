import argparse
import random
from pathlib import Path

import mglyph as mg
import numpy as np

from mglyph_ml.dataset.export import create_dataset
from mglyph_ml.experiment.e1.util import ManifestSampleShape


def gen_universal_dataset(seed: int, dataset_name: str, samples_per_x: int):

    path = Path(f"data/{dataset_name}.mglyph")

    # Check if dataset already exists
    if path.exists():
        print(f"Dataset '{dataset_name}' already exists at {path}, skipping creation.")
        return path

    print(f"Creating new dataset '{dataset_name}'...")

    # This might come in handy later when creating a randomized dataset
    np_gen = np.random.default_rng(seed)
    random.seed(seed)

    # first, we build the dataset for the experiment
    def square(x: float, canvas: mg.Canvas):
        canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
        canvas.rect(canvas.top_left, canvas.bottom_right, color="purple")

    def triangle(x: float, canvas: mg.Canvas):
        canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
        canvas.polygon([canvas.bottom_left, canvas.bottom_right, canvas.top_center], color="gray")

    def circle(x: float, canvas: mg.Canvas):
        canvas.tr.scale(mg.lerp(x, 0.05, 0.95))
        canvas.circle(canvas.center, canvas.xsize / 2, color="yellow")

    ds = create_dataset(name=dataset_name)

    total_samples_shape = samples_per_x * 100

    # xvalues_square = np_gen.uniform(0.0, 100.0, total_samples_shape)
    xvalues_triangle_train = np_gen.uniform(0.0, 100.0, total_samples_shape)
    # xvalues_circle = np_gen.uniform(0.0, 100.0, total_samples_shape)
    # xvalues_square.sort()
    xvalues_triangle_train.sort()
    # xvalues_circle.sort()

    xvalues_triangle_test = np_gen.uniform(0.0, 100.0, total_samples_shape)
    xvalues_triangle_test.sort()

    # for x in xvalues_square:
    #     ds.add_sample(square, x, split="uni", metadata={"shape": ManifestSampleShape.SQUARE})

    for x in xvalues_triangle_train:
        ds.add_sample(triangle, x, split="0", metadata={"shape": ManifestSampleShape.TRIANGLE})

    for x in xvalues_triangle_test:
        ds.add_sample(triangle, x, split="1", metadata={"shape": ManifestSampleShape.TRIANGLE})

    # for x in xvalues_circle:
    #     ds.add_sample(circle, x, split="uni", metadata={"shape": ManifestSampleShape.CIRCLE})

    ds.export(path)

    return path


def main():
    parser = argparse.ArgumentParser(description="Generate dataset once for reuse across experiments")
    parser.add_argument("--seed", type=int, default=420, help="Random seed (default: 420)")
    parser.add_argument("--dataset-name", type=str, help="Custom dataset name")
    parser.add_argument(
        "--samples-per-x", type=int, default=50, help="Number of samples per x value (default: 50)"
    )

    args = parser.parse_args()

    print(
        f"Generating dataset with name: {args.dataset_name}, seed: {args.seed}, samples_per_x: {args.samples_per_x}"
    )
    dataset_path = gen_universal_dataset(
        seed=args.seed,
        dataset_name=args.dataset_name,
        samples_per_x=args.samples_per_x,
    )
    print(f"Dataset generated at: {dataset_path}")
    print(f"Reuse this dataset by passing: --dataset-name {dataset_path}")


if __name__ == "__main__":
    main()
