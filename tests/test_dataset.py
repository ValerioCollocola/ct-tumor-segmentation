import numpy as np
import torch

from src.data.dataset import TumorSegmentationDataset


# -----------------------------
# Crea due patch fittizie
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
# Costruisce il dataset
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
# Controllo lunghezza dataset
# -----------------------------

print("Dataset length:")
print(len(dataset))

assert len(dataset) == 2


# -----------------------------
# Recupera il primo campione
# -----------------------------

image_tensor, label_tensor = dataset[0]


print("Image tensor shape:")
print(image_tensor.shape)

print("Label tensor shape:")
print(label_tensor.shape)

print("Image tensor dtype:")
print(image_tensor.dtype)

print("Label tensor dtype:")
print(label_tensor.dtype)


# -----------------------------
# Controlli automatici
# -----------------------------

assert image_tensor.shape == (
    1,
    96,
    96,
    64
)

assert label_tensor.shape == (
    96,
    96,
    64
)

assert image_tensor.dtype == torch.float32

assert label_tensor.dtype == torch.long