from typing import cast

import torch
from torch import nn


class BinnedGlyphRegressor(nn.Module):
    """
    Regresses x via classification over fixed-width bins.

    Labels in this project are normalized to [0, 1] (x / 100.0), so binning is
    done in x-space and converted back to normalized space when needed.
    """

    def __init__(
        self,
        image_resolution: tuple[int, int] = (64, 64),
        bin_size_x: float = 1.0,
        x_min: float = 0.0,
        x_max: float = 100.0,
    ):
        super().__init__()

        if bin_size_x <= 0:
            raise ValueError("bin_size_x must be > 0")
        if x_max <= x_min:
            raise ValueError("x_max must be greater than x_min")
        if image_resolution[0] <= 0 or image_resolution[1] <= 0:
            raise ValueError("image_resolution values must be > 0")

        # Include both ends (for 0..100 with size 1 -> 101 bins).
        span = x_max - x_min
        self.bin_size_x = float(bin_size_x)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.num_bins = int(round(span / self.bin_size_x)) + 1
        self.image_resolution = image_resolution

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        flattened_features = 128 * (self.image_resolution[0] // 8) * (self.image_resolution[1] // 8)
        if flattened_features <= 0:
            raise ValueError("image_resolution is too small for 3 pooling stages")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_bins),
        )

        bin_centers_x = torch.linspace(self.x_min, self.x_max, steps=self.num_bins)
        self.register_buffer("bin_centers_x", bin_centers_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    def labels_to_bins(self, labels_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized labels in [0, 1] to class indices for CE loss."""
        labels_x = labels_normalized.float() * 100.0
        bin_indices = torch.round((labels_x - self.x_min) / self.bin_size_x).long()
        return torch.clamp(bin_indices, 0, self.num_bins - 1)

    def bins_to_labels(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """Convert predicted class indices back to normalized labels in [0, 1]."""
        clamped = torch.clamp(bin_indices.long(), 0, self.num_bins - 1)
        bin_centers_x = cast(torch.Tensor, self.bin_centers_x)
        pred_x = bin_centers_x[clamped]
        return pred_x / 100.0

    def logits_to_labels(self, logits: torch.Tensor, strategy: str = "argmax") -> torch.Tensor:
        """
        Convert logits to normalized regression output in [0, 1].

        strategy='argmax': nearest bin center
        strategy='expectation': probability-weighted average over bin centers
        """
        if strategy == "argmax":
            pred_bins = torch.argmax(logits, dim=1)
            return self.bins_to_labels(pred_bins)

        if strategy == "expectation":
            probs = torch.softmax(logits, dim=1)
            bin_centers_x = cast(torch.Tensor, self.bin_centers_x)
            pred_x = probs @ bin_centers_x
            return pred_x / 100.0

        raise ValueError(f"Unknown strategy: {strategy}")
