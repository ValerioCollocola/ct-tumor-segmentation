import torch

from src.models.unet3d import UNet3D
from src.models.losses import binary_combined_loss


images = torch.randn(
    1,
    1,
    96,
    96,
    64
)

labels = torch.randint(
    low=0,
    high=2,
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
    num_classes=2,
    base_channels=8
)

outputs = model(
    images
)

total_loss, ce_loss, dice_loss = (
    binary_combined_loss(
        outputs,
        labels
    )
)

print("Binary Cross Entropy Loss:")
print(ce_loss.item())

print("Binary Tumor Dice Loss:")
print(dice_loss.item())

print("Binary Combined Loss:")
print(total_loss.item())


assert torch.isfinite(
    total_loss
)

assert total_loss.item() > 0