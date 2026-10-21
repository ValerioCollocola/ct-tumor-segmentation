from pathlib import Path

import numpy as np
import pydicom

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

from src.preprocessing.resampling import (
    resample_volume
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

target_spacing = (
    1.0,
    1.0,
    2.5
)


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
# 2. Get CT spacing
# --------------------------------------------------

_, original_spacing, _ = (
    reconstruct_ct_volume(
        sorted_ct_datasets
    )
)


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


print()
print("ORIGINAL LIVER MASK")
print("=" * 70)

print()
print("Shape:")
print(
    liver_mask.shape
)

print()
print("Spacing:")
print(
    original_spacing
)

print()
print("Unique values:")
print(
    np.unique(
        liver_mask
    )
)


# --------------------------------------------------
# 5. Resample mask
# --------------------------------------------------

resampled_liver_mask, zoom_factors = (
    resample_volume(
        liver_mask,
        current_spacing=original_spacing,
        target_spacing=target_spacing,
        is_label=True
    )
)


# force uint8 after nearest-neighbor interpolation
resampled_liver_mask = (
    resampled_liver_mask > 0
).astype(
    np.uint8
)


print()
print("RESAMPLED LIVER MASK")
print("=" * 70)

print()
print("Shape:")
print(
    resampled_liver_mask.shape
)

print()
print("Target spacing:")
print(
    target_spacing
)

print()
print("Zoom factors:")
print(
    zoom_factors
)

print()
print("Unique values:")
print(
    np.unique(
        resampled_liver_mask
    )
)


print()
print("Original liver voxels:")
print(
    np.count_nonzero(
        liver_mask
    )
)

print()
print("Resampled liver voxels:")
print(
    np.count_nonzero(
        resampled_liver_mask
    )
)


# --------------------------------------------------
# 6. Assertions
# --------------------------------------------------

assert set(
    np.unique(
        resampled_liver_mask
    )
).issubset(
    {0, 1}
)

assert resampled_liver_mask.ndim == 3

assert np.count_nonzero(
    resampled_liver_mask
) > 0


print()
print(
    "Liver mask resampling test passed."
)