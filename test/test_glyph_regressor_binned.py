import pytest
import torch

from mglyph_ml.nn.glyph_regressor_binned import BinnedGlyphRegressor


@pytest.fixture
def regressor_provider():
    provider = BinnedGlyphRegressor(num_divisions=5)
    yield provider


def test_labels_to_bins(regressor_provider: BinnedGlyphRegressor):
    # def labels_to_bins(self, labels: torch.Tensor) -> torch.Tensor:
    labels = torch.Tensor([0.0, 10.0, 19.999, 20.0, 20.01, 39.99, 40.0, 99.999, 100.0])
    bins = regressor_provider.labels_to_bins(labels)
    assert torch.equal(bins, torch.Tensor([1, 1, 1, 2, 2, 2, 3, 5, 6]))


def test_logits_to_labels(regressor_provider: BinnedGlyphRegressor):
    logits = torch.Tensor(
        [
            [0.0, 1.0, 5.0, 2.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0, 0.0, 60.0, 60.0, 0.0],
        ]
    )
    # bin centers are [-20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
    labels = regressor_provider.logits_to_labels(logits)
    assert torch.allclose(labels, torch.Tensor([26.3725, 90.0]), atol=1e-3)
