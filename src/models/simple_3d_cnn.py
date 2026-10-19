import torch.nn as nn


class Simple3DSegmentationCNN(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_classes=3
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv3d(
                in_channels,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv3d(
                32,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, x):
        return self.network(x)