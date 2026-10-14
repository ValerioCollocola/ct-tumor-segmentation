import numpy as np

from src.data.io import load_nifti
from src.preprocessing.resampling import resample_volume
from src.preprocessing.patches import (
    extract_3d_patch,
    find_label_center
)
from src.visualization.plot_tumor_patch import plot_tumor_patch


image, image_affine = load_nifti(
    "data/sample/image/liver_0.nii.gz"
)

label, label_affine = load_nifti(
    "data/sample/label/liver_0.nii.gz"
)

current_spacing = np.linalg.norm(
    image_affine[:3, :3],
    axis=0
)

target_spacing = np.array([
    1.0,
    1.0,
    2.5
])

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

tumor_center = find_label_center(
    resampled_label,
    label_value=2
)

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

plot_tumor_patch(
    image_patch,
    label_patch,
    output_path="outputs/liver_0_tumor_patch.png"
)