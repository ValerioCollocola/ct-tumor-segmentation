import torch

from src.models.unet3d import UNet3D


# -----------------------------
# Batch 3D fittizio
# -----------------------------

images = torch.randn(
    1,
    1,
    96,
    96,
    64
)


# -----------------------------
# 3D U-Net
# -----------------------------

model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)


# -----------------------------
# Forward pass
# -----------------------------

outputs = model(images)


# -----------------------------
# Output
# -----------------------------

print("Input shape:")
print(images.shape)

print("Output shape:")
print(outputs.shape)


# -----------------------------
# Controllo
# -----------------------------

assert outputs.shape == (
    1,
    3,
    96,
    96,
    64
)