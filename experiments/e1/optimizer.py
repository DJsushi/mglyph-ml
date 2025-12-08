from clearml import Task, TaskTypes
from clearml.automation import DiscreteParameterRange, GridSearch, HyperParameterOptimizer

task: Task = Task.init(
    project_name="mglyph-ml",
    task_name="Hyper-Parameter Exploration",
    task_type=TaskTypes.optimizer,
    # reuse_last_task_id=False,
)

# The hyperparameter optimizer is a task that has its own hyper-parameters
args = {
    # The ID of the task that we want to optimize.
    # The optimizer will run this specified task many times with different hyper-parameters and
    # report back the results
    "template_task_id": None
}
args = task.connect(args)

if not args["template_task_id"]:
    raise Exception("No task id provided! Exiting...")

optimizer = HyperParameterOptimizer(
    base_task_id=args["template_task_id"],
    hyper_parameters=[
        DiscreteParameterRange("start_x", [20.0, 30.0]),
        DiscreteParameterRange("end_x", [70.0, 80.0]),
    ],
    objective_metric_title="error",
    objective_metric_series="train",
    optimizer_class=GridSearch,
    max_number_of_concurrent_tasks=1,
)

optimizer.start_locally()
optimizer.wait()
