import argparse
from dataclasses import dataclass

from clearml import Task
from clearml.backend_api.services import projects
from prepare_data import prepare_data
from train_model import train_model


@dataclass
class ExperimentConfig:
    task_name: str
    task_tag: str
    start_x: float
    end_x: float
    quick: bool = True
    seed: int = 420
    max_iterations: int = 2
    max_augment_rotation_degrees: float = 5.0
    max_augment_translation_percent: float = 0.05
    good_enough_x_error: float = 0.1


def run_experiment(config: ExperimentConfig) -> None:
    """Run a single experiment with the specified parameters."""
    task: Task = Task.init(project_name="mglyph-ml", task_name=config.task_name)
    task.add_tags(config.task_tag)

    task.connect(config)

    if config.start_x is None or config.end_x is None:
        raise ValueError("Parameters 'start_x' and 'end_x' must be provided.")

    print("Starting the preparation of the dataset...")
    dataset_path = prepare_data(start_x=config.start_x, end_x=config.end_x, seed=config.seed)
    print(f"Dataset prepared. The dataset path is: {dataset_path}")

    train_model(
        dataset_path=dataset_path,
        seed=config.seed,
        max_augment_rotation_degrees=config.max_augment_rotation_degrees,
        max_augment_translation_percent=config.max_augment_translation_percent,
        quick=config.quick,
        max_iterations=config.max_iterations,
        good_enough_x_error=config.good_enough_x_error,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")
    parser.add_argument("task_name", type=str, help="Name of the task")
    parser.add_argument("--start-x", type=float, required=True, help="Start x parameter")
    parser.add_argument("--end-x", type=float, required=True, help="End x parameter")
    parser.add_argument("--quick", action="store_true", help="Run quick training (default: False)")
    parser.add_argument("--no-quick", action="store_false", dest="quick", help="Disable quick training mode")
    parser.add_argument("--seed", type=int, default=420, help="Random seed (default: 420)")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max iterations (default: 2)")
    parser.add_argument(
        "--max-augment-rotation-degrees",
        type=float,
        default=5.0,
        help="Max augmentation rotation in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--max-augment-translation-percent",
        type=float,
        default=0.05,
        help="Max augmentation translation percent (default: 0.05)",
    )
    parser.add_argument(
        "--stop-x-error",
        type=float,
        default=1.0,
        help="When test x error reaches this threshold, stop training (default: 2)",
    )
    parser.add_argument("--task-tag", type=str)

    args = parser.parse_args()

    config = ExperimentConfig(
        task_name=args.task_name,
        task_tag=args.task_tag,
        start_x=args.start_x,
        end_x=args.end_x,
        quick=args.quick,
        seed=args.seed,
        max_iterations=args.max_iterations,
        max_augment_rotation_degrees=args.max_augment_rotation_degrees,
        max_augment_translation_percent=args.max_augment_translation_percent,
    )

    run_experiment(config)
