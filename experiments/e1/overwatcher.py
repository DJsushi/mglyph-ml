"""
The experiment overwatcher is just a script that watches over the creation of different experiments.
It simply runs experiment tasks one by one as subprocesses.
"""

import subprocess
import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class ParameterSet:
    start_x: float
    end_x: float
    max_iterations: float


parameter_sets: list[ParameterSet] = []

# Generate parameter sets with varying gap sizes
for gap_size in range(10, 100, 10):
    for start in range(0, 101 - gap_size, 10):
        end = start + gap_size
        parameter_sets.append(ParameterSet(start_x=float(start), end_x=float(end), max_iterations=20))

print(parameter_sets)

# Run an experiment for each parameter set
for i, params in enumerate(parameter_sets):
    print(f"\nRunning experiment with parameter set {i + 1}/{len(parameter_sets)}")

    task_name = f"EXP 2: s={params.start_x}; e={params.end_x}; i=20"

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
        "experiment-2",
    ]

    result = subprocess.run(cmd, check=True)

    if result.returncode != 0:
        print(f"Experiment {i + 1} failed with return code {result.returncode}")
        sys.exit(1)

print("\nAll experiments completed successfully!")
