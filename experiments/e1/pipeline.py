from clearml import Task
from clearml.automation.controller import PipelineDecorator
from prepare_data import prepare_data
from train_model import train_model


@PipelineDecorator.pipeline(name="pipeline-1", project="mglyph-ml", version="0.0.1")
def main():
    task: Task = Task.current_task()

    params = {
        "start_x": 40.0,  # where the training dataset should end and test dataset should begin
        "end_x": 60.0,  # where the test dataset should end and training dataset should begin
        "quick": True,  # whether to speedrun the training for testing purposes
        "seed": 420,
        "max_iterations": 5,
        "max_augment_rotation_degrees": 5,
        "max_augment_translation_percent": 0.05,
    }
    task.connect(params)

    dataset_path = prepare_data(
        dataset_name="Dataset 1", start_x=params["start_x"], end_x=params["end_x"], seed=params["seed"]
    )
    print(f"The dataset path is: {dataset_path}")

    train_model(
        dataset_path=dataset_path,
        seed=params["seed"],
        max_augment_rotation_degrees=params["max_augment_rotation_degrees"],
        max_augment_translation_percent=params["max_augment_translation_percent"],
        quick=params["quick"],
        max_iterations=params["max_iterations"],
    )


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    main()
