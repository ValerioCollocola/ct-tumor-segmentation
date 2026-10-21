from pathlib import Path

import numpy as np
import pydicom
import torch

from src.dicom.io import (
    load_dicom_series
)

from src.dicom.series import (
    sort_dicom_slices
)

from src.dicom.reconstruction import (
    reconstruct_ct_volume
)

from src.dicom.seg import (
    reconstruct_segmentation_volume
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

ct_directory = (
    "data/dicom/liver_arterial"
)

seg_directory = Path(
    "data/dicom/liver_seg"
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
# 1. Load arterial CT
# --------------------------------------------------

ct_datasets = load_dicom_series(
    ct_directory
)

ct_datasets = [
    dataset
    for dataset in ct_datasets
    if getattr(
        dataset,
        "Modality",
        None
    ) == "CT"
]

sorted_ct_datasets = sort_dicom_slices(
    ct_datasets
)


# --------------------------------------------------
# 2. Reconstruct CT in HU
# --------------------------------------------------

volume_hu, original_spacing, _ = (
    reconstruct_ct_volume(
        sorted_ct_datasets
    )
)


print()
print("ARTERIAL CT")
print("=" * 70)

print("Original shape:")
print(volume_hu.shape)

print("Original spacing:")
print(original_spacing)


# --------------------------------------------------
# 3. Load DICOM SEG
# --------------------------------------------------

seg_dataset = None

for file_path in seg_directory.rglob("*"):

    if not file_path.is_file():
        continue

    try:
        dataset = pydicom.dcmread(
            file_path
        )

    except Exception:
        continue

    if getattr(
        dataset,
        "Modality",
        None
    ) == "SEG":

        seg_dataset = dataset
        break


if seg_dataset is None:

    raise RuntimeError(
        "No DICOM SEG object found."
    )


# --------------------------------------------------
# 4. Reconstruct original liver mask
# --------------------------------------------------

liver_mask, matched_frames = (
    reconstruct_segmentation_volume(
        seg_dataset=seg_dataset,
        sorted_ct_datasets=sorted_ct_datasets,
        segment_number=1
    )
)


# --------------------------------------------------
# 5. Preprocess arterial CT
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


windowed = apply_ct_window(
    volume_hu,
    center=window_center,
    width=window_width
)


normalized = normalize_intensity(
    windowed,
    min_value=window_min,
    max_value=window_max
)


resampled_ct, ct_zoom_factors = (
    resample_volume(
        normalized,
        current_spacing=original_spacing,
        target_spacing=target_spacing,
        is_label=False
    )
)


# --------------------------------------------------
# 6. Resample liver mask
# --------------------------------------------------

resampled_liver_mask, mask_zoom_factors = (
    resample_volume(
        liver_mask,
        current_spacing=original_spacing,
        target_spacing=target_spacing,
        is_label=True
    )
)


resampled_liver_mask = (
    resampled_liver_mask > 0
).astype(
    np.uint8
)


print()
print("RESAMPLED DATA")
print("=" * 70)

print("CT shape:")
print(resampled_ct.shape)

print("Liver mask shape:")
print(resampled_liver_mask.shape)

print("CT zoom factors:")
print(ct_zoom_factors)

print("Mask zoom factors:")
print(mask_zoom_factors)


# --------------------------------------------------
# 7. Geometry sanity check
# --------------------------------------------------

if resampled_ct.shape != (
    resampled_liver_mask.shape
):

    raise RuntimeError(
        "Resampled CT and liver mask "
        "do not have the same shape."
    )


# --------------------------------------------------
# 8. Load binary U-Net
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


model, optimizer, epoch, loss = (
    load_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        device=device
    )
)


# --------------------------------------------------
# 9. Full-volume sliding-window inference
# --------------------------------------------------

tumor_probability, raw_prediction = (
    sliding_window_inference(
        model=model,
        volume=resampled_ct,
        patch_size=patch_size,
        overlap=overlap,
        device=device
    )
)


# --------------------------------------------------
# 10. Anatomical constraint
# --------------------------------------------------

constrained_probability = (
    tumor_probability
    * resampled_liver_mask
)


constrained_prediction = (
    raw_prediction
    * resampled_liver_mask
).astype(
    np.uint8
)


# --------------------------------------------------
# 11. Compare raw vs constrained
# --------------------------------------------------

raw_voxels = np.count_nonzero(
    raw_prediction
)

constrained_voxels = np.count_nonzero(
    constrained_prediction
)


outside_liver_voxels = (
    raw_voxels
    - constrained_voxels
)


print()
print("ANATOMICALLY CONSTRAINED INFERENCE")
print("=" * 70)


print()
print("Raw predicted tumor voxels:")
print(raw_voxels)


print()
print("Predicted tumor voxels inside liver:")
print(constrained_voxels)


print()
print("Predicted voxels removed outside liver:")
print(outside_liver_voxels)


if raw_voxels > 0:

    removed_fraction = (
        outside_liver_voxels
        / raw_voxels
    )

else:

    removed_fraction = 0.0


print()
print("Fraction removed:")
print(removed_fraction)


print()
print("Raw maximum probability:")
print(
    tumor_probability.max()
)


print()
print("Maximum probability inside liver:")
print(
    constrained_probability.max()
)


# --------------------------------------------------
# 12. Assertions
# --------------------------------------------------

assert (
    tumor_probability.shape
    == resampled_liver_mask.shape
)

assert (
    constrained_prediction.shape
    == resampled_liver_mask.shape
)

assert constrained_voxels <= raw_voxels


print()
print(
    "Arterial liver-constrained inference passed."
)