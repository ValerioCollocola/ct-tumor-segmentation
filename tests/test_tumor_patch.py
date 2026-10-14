import numpy as np

from src.data.io import load_nifti
from src.preprocessing.resampling import resample_volume
from src.preprocessing.patches import (
    extract_3d_patch,
    find_label_center
)


# -----------------------------
# Caricamento CT e label
# -----------------------------

image, image_affine = load_nifti(
    "data/sample/image/liver_0.nii.gz"
)

label, label_affine = load_nifti(
    "data/sample/label/liver_0.nii.gz"
)


# -----------------------------
# Spacing originale
# -----------------------------

current_spacing = np.linalg.norm(
    image_affine[:3, :3],
    axis=0
)

target_spacing = np.array([
    1.0,
    1.0,
    2.5
])


# -----------------------------
# Resampling
# -----------------------------

resampled_image, _ = resample_volume(
    image,
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


# -----------------------------
# Trova il centro del tumore
# -----------------------------

tumor_center = find_label_center(
    resampled_label,
    label_value=2
)

print("Tumor center:")
print(tumor_center)


# -----------------------------
# Estrazione patch
# -----------------------------

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


# -----------------------------
# Controlli
# -----------------------------

print("Image patch shape:")
print(image_patch.shape)

print("Label patch shape:")
print(label_patch.shape)

print("Label values inside patch:")
print(np.unique(label_patch))

print("Tumor voxels inside patch:")
print(np.count_nonzero(label_patch == 2))


assert image_patch.shape == patch_size
assert label_patch.shape == patch_size

assert 2 in np.unique(label_patch)