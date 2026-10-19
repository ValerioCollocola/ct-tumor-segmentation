import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import TumorSegmentationDataset


# -----------------------------
# Patch fittizie
# -----------------------------

image_patch_1 = np.zeros(
    (96, 96, 64),
    dtype=np.float32
)

image_patch_2 = np.ones(
    (96, 96, 64),
    dtype=np.float32
)

label_patch_1 = np.zeros(
    (96, 96, 64),
    dtype=np.int64
)

label_patch_2 = np.ones(
    (96, 96, 64),
    dtype=np.int64
)


# -----------------------------
# Dataset
# -----------------------------

dataset = TumorSegmentationDataset(
    image_patches=[
        image_patch_1,
        image_patch_2
    ],
    label_patches=[
        label_patch_1,
        label_patch_2
    ]
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
# Recupera un batch
# -----------------------------

images, labels = next(
    iter(dataloader)
)


print("Batch image shape:")
print(images.shape)

print("Batch label shape:")
print(labels.shape)

print("Batch image dtype:")
print(images.dtype)

print("Batch label dtype:")
print(labels.dtype)


# -----------------------------
# Controlli automatici
# -----------------------------

assert images.shape == (
    2,
    1,
    96,
    96,
    64
)

assert labels.shape == (
    2,
    96,
    96,
    64
)