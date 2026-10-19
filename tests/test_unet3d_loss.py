import torch

from src.models.unet3d import UNet3D
from src.models.losses import combined_loss


images = torch.randn(
    1,
    1,
    96,
    96,
    64
)

labels = torch.randint(
    low=0,
    high=3,
    size=(
        1,
        96,
        96,
        64
    ),
    dtype=torch.long
)

model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)

outputs = model(images)

total_loss, ce_loss, dice = combined_loss(
    outputs,
    labels,
    num_classes=3
)

print("Output shape:")
print(outputs.shape)

print("Cross Entropy Loss:")
print(ce_loss.item())

print("Dice Loss:")
print(dice.item())

print("Combined Loss:")
print(total_loss.item())

assert outputs.shape == (
    1,
    3,
    96,
    96,
    64
)

assert torch.isfinite(total_loss)
assert total_loss.item() > 0