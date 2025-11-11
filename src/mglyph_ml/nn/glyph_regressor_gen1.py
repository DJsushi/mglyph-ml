import torch
from torch import nn


class GlyphRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (batch, 3, 512, 512)
        # First conv: small kernel (3x3), shallow depth (4)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=4, kernel_size=9, padding=4
        )  # (3,512,512) -> (4,512,512)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (4,512,512) -> (4,256,256)

        # Second conv: medium kernel (5x5), medium depth (8)
        self.conv2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=9, padding=4
        )  # (4,256,256) -> (8,256,256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (8,256,256) -> (8,128,128)

        # Third conv: large kernel (7x7), deeper (16)
        self.conv3 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=9, padding=4
        )  # (8,128,128) -> (16,128,128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (16,128,128) -> (16,64,64)

        # Reduce spatial size before FC to lower memory
        self.adaptivepool = nn.AdaptiveAvgPool2d((8, 8))  # (16,64,64) -> (16,8,8)

        # Flatten: 16*8*8 = 1024
        self.fc1 = nn.Linear(16 * 8 * 8, 16 * 8 * 8)
        self.fc2 = nn.Linear(16 * 8 * 8, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # First conv block
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        # Second conv block
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        # Third conv block
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        # Final pooling and fully connected layers
        x = self.adaptivepool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
