import numpy as np
import torch

from src.dicom.io import (
    load_dicom_series,
    group_by_series
)

from src.dicom.reconstruction import (
    reconstruct_ct_volume
)

from src.preprocessing.windowing import (
    apply_ct_window
)

from src.preprocessing.normalization import (
    normalize_intensity
)

from src.preprocessing.resampling import (
    resample_volume
)

from src.models.unet3d import (
    UNet3D
)

from src.training.checkpoint import (
    load_checkpoint
)

from src.inference.sliding_window import (
    sliding_window_inference
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

dicom_directory = (
    "data/dicom/liver_portal_venous"
)

checkpoint_path = (
    "outputs/checkpoints/"
    "best_real_binary_unet_best_tumor_dice.pt"
)

target_spacing = (
    1.0,
    1.0,
    2.5
)

patch_size = (
    96,
    96,
    64
)

overlap = 0.5

device = "cpu"


# --------------------------------------------------
# 1. Load DICOM datasets
# --------------------------------------------------

datasets = load_dicom_series(
    dicom_directory
)

series_dict = group_by_series(
    datasets
)


# --------------------------------------------------
# 2. Select CT series
# --------------------------------------------------

ct_series = None

for series_uid, series_datasets in series_dict.items():

    modality = getattr(
        series_datasets[0],
        "Modality",
        None
    )

    if modality == "CT":

        ct_series = series_datasets
        break


if ct_series is None:

    raise RuntimeError(
        "No CT series found."
    )


# --------------------------------------------------
# 3. Reconstruct HU volume
# --------------------------------------------------

volume_hu, original_spacing, sorted_datasets = (
    reconstruct_ct_volume(
        ct_series
    )
)


print()
print("Original DICOM volume:")
print(volume_hu.shape)

print("Original spacing:")
print(original_spacing)


# --------------------------------------------------
# 4. Windowing
# --------------------------------------------------

window_center = 50
window_width = 150

window_min = (
    window_center
    - window_width / 2
)

window_max = (
    window_center
    + window_width / 2
)


windowed_volume = apply_ct_window(
    volume_hu,
    center=window_center,
    width=window_width
)


# --------------------------------------------------
# 5. Normalization
# --------------------------------------------------

normalized_volume = normalize_intensity(
    windowed_volume,
    min_value=window_min,
    max_value=window_max
)


# --------------------------------------------------
# 6. Resampling
# --------------------------------------------------

resampled_volume, zoom_factors = (
    resample_volume(
        normalized_volume,
        current_spacing=original_spacing,
        target_spacing=target_spacing,
        is_label=False
    )
)


print()
print("Resampled volume:")
print(resampled_volume.shape)

print("Zoom factors:")
print(zoom_factors)


# --------------------------------------------------
# 7. Load binary U-Net
# --------------------------------------------------

model = UNet3D(
    in_channels=1,
    num_classes=2,
    base_channels=8
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


model, optimizer, epoch, loss = load_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_path=checkpoint_path,
    device=device
)


print()
print("Loaded checkpoint epoch:")
print(epoch)

print("Checkpoint loss:")
print(loss)


# --------------------------------------------------
# 8. Sliding-window inference
# --------------------------------------------------

tumor_probability, prediction_mask = (
    sliding_window_inference(
        model=model,
        volume=resampled_volume,
        patch_size=patch_size,
        overlap=overlap,
        device=device
    )
)


# --------------------------------------------------
# 9. Summary
# --------------------------------------------------

print()
print("FULL-VOLUME DICOM INFERENCE")
print("=" * 60)


print()
print("Probability map shape:")
print(
    tumor_probability.shape
)


print()
print("Prediction mask shape:")
print(
    prediction_mask.shape
)


print()
print("Probability range:")
print(
    tumor_probability.min(),
    "to",
    tumor_probability.max()
)


print()
print("Mean tumor probability:")
print(
    tumor_probability.mean()
)


predicted_tumor_voxels = np.count_nonzero(
    prediction_mask
)


print()
print("Predicted tumor voxels:")
print(
    predicted_tumor_voxels
)


predicted_fraction = (
    predicted_tumor_voxels
    / prediction_mask.size
)


print()
print("Predicted tumor fraction:")
print(
    predicted_fraction
)


# --------------------------------------------------
# 10. Assertions
# --------------------------------------------------

assert tumor_probability.shape == (
    resampled_volume.shape
)

assert prediction_mask.shape == (
    resampled_volume.shape
)

assert tumor_probability.min() >= 0.0

assert tumor_probability.max() <= 1.0


print()
print(
    "Full-volume DICOM inference completed."
)