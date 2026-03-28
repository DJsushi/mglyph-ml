from torch import nn


class GlyphRegressor(nn.Module):
    def __init__(self, image_resolution: tuple[int, int]):
        super().__init__()

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

        flattened_features = 128 * (image_resolution[0] // 8) * (image_resolution[1] // 8)
        if flattened_features <= 0:
            raise ValueError("image_resolution is too small for 3 pooling stages")

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
