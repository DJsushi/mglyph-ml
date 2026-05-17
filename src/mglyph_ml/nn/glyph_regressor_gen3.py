from torch import nn
import torch

from mglyph_ml.nn.base import GlyphPredictorBase


class GlyphRegressorGen3(GlyphPredictorBase):
    """
    Stronger plain regressor (no binning) for normalized labels in [0, 1].

    This keeps a similar backbone to the binned model but predicts a single
    scalar directly.
    """

    def __init__(self, image_resolution: tuple[int, int] = (64, 64)):
        super().__init__()

        if image_resolution[0] <= 0 or image_resolution[1] <= 0:
            raise ValueError("image_resolution values must be > 0")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        flattened_features = 128 * (image_resolution[0] // 8) * (image_resolution[1] // 8)
        if flattened_features <= 0:
            raise ValueError("image_resolution is too small for 3 pooling stages")

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict normalized labels in [0, 1].

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Tensor of shape (batch_size,) with predictions in [0, 1]
        """
        return self(x).view(-1)
