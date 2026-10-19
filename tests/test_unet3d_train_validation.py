import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.dataset import TumorSegmentationDataset
from src.models.unet3d import UNet3D
from src.training.train import train_and_validate
from src.training.checkpoint import load_checkpoint


# -----------------------------
# Dati sintetici di training
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
# Dati sintetici di validation
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
    batch_size=1,
    shuffle=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)


# -----------------------------
# 3D U-Net
# -----------------------------

model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# -----------------------------
# Training + Validation
# -----------------------------

checkpoint_path = (
    "outputs/checkpoints/"
    "best_unet3d_model.pt"
)

history = train_and_validate(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    num_epochs=3,
    device="cpu",
    checkpoint_path=checkpoint_path
)


# -----------------------------
# Output history
# -----------------------------

print("UNet training history:")
print(history)


# -----------------------------
# Ricarica il best checkpoint
# -----------------------------

new_model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)

new_optimizer = torch.optim.Adam(
    new_model.parameters(),
    lr=1e-3
)

new_model, new_optimizer, best_epoch, best_loss = load_checkpoint(
    model=new_model,
    optimizer=new_optimizer,
    checkpoint_path=checkpoint_path,
    device="cpu"
)


print("Best UNet epoch:")
print(best_epoch)

print("Best UNet validation loss:")
print(best_loss)


# -----------------------------
# Controlli
# -----------------------------

assert len(
    history["train_loss"]
) == 3

assert len(
    history["val_loss"]
) == 3

assert best_epoch >= 1
assert best_loss > 0