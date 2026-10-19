import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.dataset import TumorSegmentationDataset
from src.models.simple_3d_cnn import Simple3DSegmentationCNN
from src.training.train import train_and_validate
from src.visualization.plot_training_history import plot_training_history

# -----------------------------
# Training data
# -----------------------------

train_images = [
    np.random.rand(
        96,
        96,
        64
    ).astype(np.float32)
    for _ in range(4)
]

train_labels = [
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
# Validation data
# -----------------------------

val_images = [
    np.random.rand(
        96,
        96,
        64
    ).astype(np.float32)
    for _ in range(2)
]

val_labels = [
    np.random.randint(
        0,
        3,
        size=(
            96,
            96,
            64
        )
    ).astype(np.int64)
    for _ in range(2)
]


# -----------------------------
# Dataset
# -----------------------------

train_dataset = TumorSegmentationDataset(
    image_patches=train_images,
    label_patches=train_labels
)

val_dataset = TumorSegmentationDataset(
    image_patches=val_images,
    label_patches=val_labels
)


# -----------------------------
# DataLoader
# -----------------------------

train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)


# -----------------------------
# Model
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
# Train + Validation
# -----------------------------

history = train_and_validate(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    num_epochs=3,
    device="cpu",
    checkpoint_path="outputs/checkpoints/best_test_model.pt"
)


# -----------------------------
# Output
# -----------------------------

print("Training history:")
print(history)


# -----------------------------
# Controlli
# -----------------------------

assert len(
    history["train_loss"]
) == 3

assert len(
    history["val_loss"]
) == 3

plot_training_history(
    history,
    output_path="outputs/training_validation_loss.png"
)