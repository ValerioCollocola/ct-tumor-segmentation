import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        base_channels=8
    ):
        super().__init__()

        # -----------------------------
        # Encoder
        # -----------------------------

        self.encoder1 = DoubleConv3D(
            in_channels,
            base_channels
        )

        self.pool1 = nn.MaxPool3d(
            kernel_size=2
        )

        self.encoder2 = DoubleConv3D(
            base_channels,
            base_channels * 2
        )

        self.pool2 = nn.MaxPool3d(
            kernel_size=2
        )

        # -----------------------------
        # Bottleneck
        # -----------------------------

        self.bottleneck = DoubleConv3D(
            base_channels * 2,
            base_channels * 4
        )

        # -----------------------------
        # Decoder
        # -----------------------------

        self.upconv2 = nn.ConvTranspose3d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = DoubleConv3D(
            base_channels * 4,
            base_channels * 2
        )

        self.upconv1 = nn.ConvTranspose3d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = DoubleConv3D(
            base_channels * 2,
            base_channels
        )

        # -----------------------------
        # Output layer
        # -----------------------------

        self.output_layer = nn.Conv3d(
            base_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        enc1 = self.encoder1(x)

        x = self.pool1(enc1)

        enc2 = self.encoder2(x)

        x = self.pool2(enc2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder livello 2
        x = self.upconv2(x)

        x = torch.cat(
            (x, enc2),
            dim=1
        )

        x = self.decoder2(x)

        # Decoder livello 1
        x = self.upconv1(x)

        x = torch.cat(
            (x, enc1),
            dim=1
        )

        x = self.decoder1(x)

        # Output
        x = self.output_layer(x)

        return x