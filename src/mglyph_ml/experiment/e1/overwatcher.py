"""
The experiment overwatcher is just a script that watches over the creation of different experiments.
It runs the same experiment with different numbers of workers to benchmark performance.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from mglyph_ml.experiment.e1.experiment import ExperimentConfig, run_experiment

load_dotenv()

gaps: list[tuple[float, float]] = [
    # (2.0, 98.0),
    # (1.0, 99.0),
    # (3.0, 97.0),
    # (5.0, 95.0),
    (10.0, 90.0),
    (0.0, 10.0),
    (90.0, 100.0),
    (0.0, 0.0),
]

for index, gap in enumerate(gaps):
    config = ExperimentConfig(
        task_name=f"Experiment 1.2.2,x[{gap[0]},{gap[1]}]",
        task_tag="exp-1.2.2",
        dataset_path=Path("data/uni.mglyph"),
        gap_start_x=gap[0],
        gap_end_x=gap[1],
        quick=False,
        seed=420,
        # max_iterations=2 if index == 0 else 20,
        max_iterations=20,
        data_loader_num_workers=16,
        offline=False,
    )

    try:
        run_experiment(config)
    except Exception as e:
        print(f"Experiment {index} failed: {e}")

print("\nAll experiments completed successfully!")
