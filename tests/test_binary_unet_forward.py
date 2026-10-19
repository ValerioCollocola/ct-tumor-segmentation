import torch

from src.models.unet3d import UNet3D


images = torch.randn(
    1,
    1,
    96,
    96,
    64
)

model = UNet3D(
    in_channels=1,
    num_classes=2,
    base_channels=8
)

outputs = model(images)


print("Input shape:")
print(images.shape)

print("Binary UNet output shape:")
print(outputs.shape)


assert outputs.shape == (
    1,
    2,
    96,
    96,
    64
)