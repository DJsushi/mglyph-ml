from typing import Literal, cast

import torch
from torch import nn

from mglyph_ml.nn.base import GlyphPredictorBase


def _scaled_channels(base_channels: int, width_mult: float) -> int:
    """Scale channels with width multiplier and round to a hardware-friendly multiple."""
    scaled = int(round(base_channels * width_mult / 8.0) * 8)
    return max(8, scaled)


class BinnedGlyphRegressor(GlyphPredictorBase):
    """
    Regresses x via classification over fixed-width bins.
    """

    def __init__(
        self,
        image_resolution: tuple[int, int] = (64, 64),
        num_divisions: int = 5,
        width_mult: float = 1,
        dropout: float = 0.00,
        use_new_centroid_distribution: bool = True,
    ):
        super().__init__()

        if num_divisions <= 0:
            raise ValueError("num_bins must be > 0")
        if image_resolution[0] <= 0 or image_resolution[1] <= 0:
            raise ValueError("image_resolution values must be > 0")
        if width_mult <= 0.0:
            raise ValueError("width_mult must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

        c1 = _scaled_channels(32, width_mult)
        c2 = _scaled_channels(64, width_mult)
        c3 = _scaled_channels(128, width_mult)

        self.num_bins = num_divisions + 3
        self.bin_size_x = 100.0 / self.num_bins
        self.image_resolution = image_resolution
        self.width_mult = width_mult
        self.dropout = dropout

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        flattened_features = c3 * (self.image_resolution[0] // 8) * (self.image_resolution[1] // 8)
        if flattened_features <= 0:
            raise ValueError("image_resolution is too small for 3 pooling stages")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, self.num_bins),
        )

        centroid_distance = 100 / num_divisions
        centroid_count = num_divisions + 3

        if use_new_centroid_distribution:
            centroids_x = torch.linspace(-centroid_distance, 100 + centroid_distance, centroid_count)
        else:
            centroids_x = torch.linspace(0, 100, centroid_count)

        self.register_buffer("centroids_x", centroids_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    def labels_to_bins(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert labels in [0, 100] to class indices for CE loss."""
        bin_indices = torch.floor(labels / self.bin_size_x).long()
        # we gotta clamp here just to make sure rounding errors at label=100.0 don't mess something up
        return torch.clamp(bin_indices, 0, self.num_bins - 1)

    def logits_to_labels(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to regression output in [0, 100].
        """
        centroids_x = cast(torch.Tensor, self.centroids_x)
        probs = torch.softmax(logits, dim=1)
        return torch.clamp(torch.sum(probs * centroids_x, dim=1), 0.0, 100.0)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict labels in [0, 100].

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Tensor of shape (batch_size,) with predictions in [0, 100]
        """
        logits = self(x)
        return self.logits_to_labels(logits)
