import torch

from src.models.simple_3d_cnn import Simple3DSegmentationCNN
from src.models.losses import combined_loss


# -----------------------------
# Batch fittizio
# -----------------------------

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


# -----------------------------
# Modello
# -----------------------------

model = Simple3DSegmentationCNN(
    in_channels=1,
    num_classes=3
)


# -----------------------------
# Forward pass
# -----------------------------

outputs = model(images)


# -----------------------------
# Combined loss
# -----------------------------

total_loss, ce_loss, d_loss = combined_loss(
    outputs,
    labels,
    num_classes=3
)


print("Cross Entropy Loss:")
print(ce_loss.item())

print("Dice Loss:")
print(d_loss.item())

print("Combined Loss:")
print(total_loss.item())


# -----------------------------
# Controlli
# -----------------------------

assert torch.isfinite(total_loss)
assert torch.isfinite(ce_loss)
assert torch.isfinite(d_loss)

assert total_loss.item() > 0