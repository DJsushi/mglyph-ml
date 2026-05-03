"""Base classes for glyph regressors."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class GlyphPredictorBase(nn.Module, ABC):
    """Abstract base class for all glyph regressors.

    All regressors must implement the predict() method which should
    return predictions in a consistent format.
    """

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict glyph labels from input images.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Tensor of shape (batch_size,) with predictions in the model's
            native range (e.g., [0, 1] or [0, 100]).
        """
        pass
