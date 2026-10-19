import numpy as np
import torch

from src.data.io import load_nifti
from src.data.dataset import TumorSegmentationDataset

from src.preprocessing.windowing import apply_ct_window
from src.preprocessing.normalization import normalize_intensity
from src.preprocessing.resampling import resample_volume
from src.preprocessing.patches import (
    find_label_center,
    extract_3d_patch
)

from src.models.unet3d import UNet3D
from src.models.losses import combined_loss


# --------------------------------------------------
# 1. Caricamento del caso reale
# --------------------------------------------------

image, image_affine = load_nifti(
    "data/sample/image/liver_0.nii.gz"
)

label, label_affine = load_nifti(
    "data/sample/label/liver_0.nii.gz"
)


# --------------------------------------------------
# 2. CT windowing
# --------------------------------------------------

windowed_image = apply_ct_window(
    image,
    center=50,
    width=150
)


# --------------------------------------------------
# 3. Intensity normalization
# --------------------------------------------------

normalized_image = normalize_intensity(
    windowed_image,
    min_value=-25,
    max_value=125
)


# --------------------------------------------------
# 4. Spacing originale
# --------------------------------------------------

current_spacing = np.linalg.norm(
    image_affine[:3, :3],
    axis=0
)

target_spacing = np.array([
    1.0,
    1.0,
    2.5
])


# --------------------------------------------------
# 5. Resampling CT e label
# --------------------------------------------------

resampled_image, _ = resample_volume(
    normalized_image,
    current_spacing=current_spacing,
    target_spacing=target_spacing,
    is_label=False
)

resampled_label, _ = resample_volume(
    label,
    current_spacing=current_spacing,
    target_spacing=target_spacing,
    is_label=True
)


# --------------------------------------------------
# 6. Trova il centro del tumore
# --------------------------------------------------

tumor_center = find_label_center(
    resampled_label,
    label_value=2
)


# --------------------------------------------------
# 7. Estrae patch CT + label
# --------------------------------------------------

patch_size = (
    96,
    96,
    64
)

image_patch = extract_3d_patch(
    resampled_image,
    center=tumor_center,
    patch_size=patch_size
)

label_patch = extract_3d_patch(
    resampled_label,
    center=tumor_center,
    patch_size=patch_size
)


# --------------------------------------------------
# 8. PyTorch Dataset
# --------------------------------------------------

dataset = TumorSegmentationDataset(
    image_patches=[
        image_patch
    ],
    label_patches=[
        label_patch
    ]
)

image_tensor, label_tensor = dataset[0]


# Aggiunge la batch dimension
image_tensor = image_tensor.unsqueeze(0)
label_tensor = label_tensor.unsqueeze(0)


# --------------------------------------------------
# 9. 3D U-Net
# --------------------------------------------------

model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)


# --------------------------------------------------
# 10. Forward pass
# --------------------------------------------------

outputs = model(
    image_tensor
)


# --------------------------------------------------
# 11. Loss sul caso reale
# --------------------------------------------------

loss, ce_loss, dice_loss = combined_loss(
    outputs,
    label_tensor,
    num_classes=3
)


# --------------------------------------------------
# 12. Risultati
# --------------------------------------------------

print("Original CT shape:")
print(image.shape)

print("Resampled CT shape:")
print(resampled_image.shape)

print("Tumor center:")
print(tumor_center)

print("Image patch shape:")
print(image_patch.shape)

print("Image patch range:")
print(
    image_patch.min(),
    image_patch.max()
)

print("Label values:")
print(
    np.unique(label_patch)
)

print("PyTorch input shape:")
print(image_tensor.shape)

print("UNet output shape:")
print(outputs.shape)

print("Cross Entropy Loss:")
print(ce_loss.item())

print("Dice Loss:")
print(dice_loss.item())

print("Combined Loss:")
print(loss.item())


# --------------------------------------------------
# 13. Controlli
# --------------------------------------------------

assert image_tensor.shape == (
    1,
    1,
    96,
    96,
    64
)

assert label_tensor.shape == (
    1,
    96,
    96,
    64
)

assert outputs.shape == (
    1,
    3,
    96,
    96,
    64
)

assert np.array_equal(
    np.unique(label_patch),
    np.array([0.0, 1.0, 2.0])
)

assert torch.isfinite(loss)