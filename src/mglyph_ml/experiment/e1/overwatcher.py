"""
The experiment overwatcher is just a script that watches over the creation of different experiments.
It runs the same experiment with different numbers of workers to benchmark performance.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from mglyph_ml.experiment.e1.experiment import ExperimentConfig, run_experiment

load_dotenv()

# Run the same experiment with different numbers of workers
worker_counts = [8, 16, 32, 40, 48]

for i, num_workers in enumerate(worker_counts):
    print(f"\nRunning experiment {i + 1}/{len(worker_counts)} with num_workers={num_workers}")

    config = ExperimentConfig(
        task_name=f"Experiment 1 - num_workers={num_workers}",
        task_tag="exp-1-workers-benchmark",
        dataset_path=Path("data/uni.mglyph"),
        gap_start_x=30.0,
        gap_end_x=70.0,
        quick=False,
        seed=420,
        max_iterations=3,
        data_loader_num_workers=num_workers,
        offline=False,
    )

    try:
        run_experiment(config)
        print(f"Experiment with num_workers={num_workers} completed successfully!")
    except Exception as e:
        print(f"Experiment with num_workers={num_workers} failed with error: {e}")
        sys.exit(1)

print("\nAll experiments completed successfully!")
