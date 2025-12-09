from clearml import Task, TaskTypes
from clearml.automation.controller import PipelineDecorator
from pipeline import main


task: Task = Task.init(
    project_name="mglyph-ml",
    task_name="Hyper-Parameter Exploration",
    task_type=TaskTypes.testing,
    # reuse_last_task_id=False,
)

# Define the parameter sets you want to explore
parameter_sets = [
    {
        "start_x": 10.0,
        "end_x": 90.0,
    },
    {
        "start_x": 30.0,
        "end_x": 70.0,
    },
    {
        "start_x": 45.0,
        "end_x": 55.0,
    },
]

# Run the pipeline with each parameter set
for i, params in enumerate(parameter_sets):
    print(f"\nRunning pipeline with parameter set {i + 1}/{len(parameter_sets)}")
    print(f"Parameters: {params}")

    # Execute the pipeline with the current parameters
    main(start_x=params["start_x"], end_x=params["end_x"])

print("\nHyper-parameter exploration complete!")
