import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.dataset import TumorSegmentationDataset
from src.models.simple_3d_cnn import Simple3DSegmentationCNN
from src.training.validate import validate_model


# -----------------------------
# Dati fittizi
# -----------------------------

image_patches = [
    np.random.rand(
        96,
        96,
        64
    ).astype(np.float32)
    for _ in range(4)
]

label_patches = [
    np.random.randint(
        0,
        3,
        size=(
            96,
            96,
            64
        )
    ).astype(np.int64)
    for _ in range(4)
]


# -----------------------------
# Dataset
# -----------------------------

dataset = TumorSegmentationDataset(
    image_patches=image_patches,
    label_patches=label_patches
)


# -----------------------------
# DataLoader
# -----------------------------

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)


# -----------------------------
# Modello
# -----------------------------

model = Simple3DSegmentationCNN(
    in_channels=1,
    num_classes=3
)


# -----------------------------
# Validation
# -----------------------------

validation_loss = validate_model(
    model=model,
    dataloader=dataloader,
    device="cpu"
)


print("Validation Loss:")
print(validation_loss)


# -----------------------------
# Controlli
# -----------------------------

assert validation_loss > 0