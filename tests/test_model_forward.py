import torch

from src.models.simple_3d_cnn import Simple3DSegmentationCNN


# -----------------------------
# Batch 3D fittizio
# -----------------------------

images = torch.randn(
    2,
    1,
    96,
    96,
    64
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


print("Input shape:")
print(images.shape)

print("Output shape:")
print(outputs.shape)


# -----------------------------
# Controllo automatico
# -----------------------------

assert outputs.shape == (
    2,
    3,
    96,
    96,
    64
)