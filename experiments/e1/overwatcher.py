"""
The experiment overwatcher is just a script that watches over the creation of different experiments.
It simply runs experiment tasks one by one as subprocesses.
"""

import subprocess
import sys
from dataclasses import dataclass


@dataclass
class ParameterSet:
    start_x: float
    end_x: float
    max_iterations: float


parameter_sets = [
    ParameterSet(start_x=10.0, end_x=90.0, max_iterations=20),
    ParameterSet(start_x=30.0, end_x=70.0, max_iterations=20),
    ParameterSet(start_x=45.0, end_x=55.0, max_iterations=20),
]

# Run an experiment for each parameter set
for i, params in enumerate(parameter_sets):
    print(f"\nRunning experiment with parameter set {i + 1}/{len(parameter_sets)}")

    task_name = f"EXP 1: s={params.start_x}; e={params.end_x}"

    cmd = [
        sys.executable,
        "experiments/e1/experiment.py",
        task_name,
        "--start-x",
        str(params.start_x),
        "--end-x",
        str(params.end_x),
        "--max-iterations",
        str(params.max_iterations),
        "--task-tag",
        "experiment-1"
    ]

    result = subprocess.run(cmd, check=True)

    if result.returncode != 0:
        print(f"Experiment {i + 1} failed with return code {result.returncode}")
        sys.exit(1)

print("\nAll experiments completed successfully!")
