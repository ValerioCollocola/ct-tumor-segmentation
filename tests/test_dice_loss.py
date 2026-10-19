import torch

from src.models.simple_3d_cnn import Simple3DSegmentationCNN
from src.models.losses import dice_loss


images = torch.randn(
    2,
    1,
    96,
    96,
    64
)

labels = torch.randint(
    low=0,
    high=3,
    size=(
        2,
        96,
        96,
        64
    ),
    dtype=torch.long
)

model = Simple3DSegmentationCNN(
    in_channels=1,
    num_classes=3
)

outputs = model(images)

loss = dice_loss(
    outputs,
    labels,
    num_classes=3
)

print("Dice Loss:")
print(loss.item())

assert loss.ndim == 0
assert torch.isfinite(loss)
assert 0 <= loss.item() <= 1