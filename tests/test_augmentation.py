import numpy as np

from src.data.io import load_nifti
from src.preprocessing.resampling import resample_volume
from src.preprocessing.patches import (
    extract_3d_patch,
    find_label_center
)
from src.preprocessing.augmentation import random_flip_3d


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
# Centro del tumore
# -----------------------------

tumor_center = find_label_center(
    resampled_label,
    label_value=2
)


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
# Data augmentation
# -----------------------------

augmented_image, augmented_label = random_flip_3d(
    image_patch,
    label_patch,
    axis=0
)


# -----------------------------
# Controlli
# -----------------------------

print("Original image patch shape:")
print(image_patch.shape)

print("Augmented image patch shape:")
print(augmented_image.shape)

print("Original label values:")
print(np.unique(label_patch))

print("Augmented label values:")
print(np.unique(augmented_label))

original_tumor_voxels = np.count_nonzero(
    label_patch == 2
)

augmented_tumor_voxels = np.count_nonzero(
    augmented_label == 2
)

print("Original tumor voxels:")
print(original_tumor_voxels)

print("Augmented tumor voxels:")
print(augmented_tumor_voxels)


assert augmented_image.shape == image_patch.shape
assert augmented_label.shape == label_patch.shape

assert np.array_equal(
    np.unique(augmented_label),
    np.unique(label_patch)
)

assert original_tumor_voxels == augmented_tumor_voxels