import torch
import torch.nn as nn

from src.models.simple_3d_cnn import Simple3DSegmentationCNN


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
# Loss function
# -----------------------------

criterion = nn.CrossEntropyLoss()

loss = criterion(
    outputs,
    labels
)


# -----------------------------
# Output
# -----------------------------

print("Model output shape:")
print(outputs.shape)

print("Label shape:")
print(labels.shape)

print("Loss:")
print(loss.item())


# -----------------------------
# Controlli
# -----------------------------

assert loss.ndim == 0
assert loss.item() > 0
assert torch.isfinite(loss)