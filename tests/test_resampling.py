import numpy as np

from src.data.io import load_nifti
from src.preprocessing.resampling import resample_volume


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

resampled_image, image_zoom_factors = resample_volume(
    image,
    current_spacing=current_spacing,
    target_spacing=target_spacing,
    is_label=False
)

resampled_label, label_zoom_factors = resample_volume(
    label,
    current_spacing=current_spacing,
    target_spacing=target_spacing,
    is_label=True
)

print("Original image shape:")
print(image.shape)

print("Current spacing:")
print(current_spacing)

print("Target spacing:")
print(target_spacing)

print("Zoom factors:")
print(image_zoom_factors)

print("Resampled image shape:")
print(resampled_image.shape)

print("Resampled label shape:")
print(resampled_label.shape)

print("Original label values:")
print(np.unique(label))

print("Resampled label values:")
print(np.unique(resampled_label))