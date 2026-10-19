import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.dataset import TumorSegmentationDataset
from src.models.simple_3d_cnn import Simple3DSegmentationCNN
from src.training.train import train_model


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
# Dataset e DataLoader
# -----------------------------

dataset = TumorSegmentationDataset(
    image_patches=image_patches,
    label_patches=label_patches
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# -----------------------------
# Modello
# -----------------------------

model = Simple3DSegmentationCNN(
    in_channels=1,
    num_classes=3
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# -----------------------------
# Training
# -----------------------------

history = train_model(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    num_epochs=3,
    device="cpu"
)


# -----------------------------
# Controlli
# -----------------------------

print("Training history:")
print(history)

assert len(history) == 3