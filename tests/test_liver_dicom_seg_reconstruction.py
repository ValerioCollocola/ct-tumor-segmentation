from pathlib import Path

import numpy as np
import pydicom

from src.dicom.io import (
    load_dicom_series
)

from src.dicom.series import (
    sort_dicom_slices
)

from src.dicom.seg import (
    reconstruct_segmentation_volume
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


print()
print("ARTERIAL CT")
print("=" * 70)

print()
print("Number of CT slices:")
print(
    len(sorted_ct_datasets)
)


# --------------------------------------------------
# 2. Load DICOM SEG
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


print()
print("DICOM SEG")
print("=" * 70)

print()
print("Frames:")
print(
    seg_dataset.NumberOfFrames
)


# --------------------------------------------------
# 3. Reconstruct full-volume liver mask
# --------------------------------------------------

liver_mask, matched_frames = (
    reconstruct_segmentation_volume(
        seg_dataset=seg_dataset,
        sorted_ct_datasets=sorted_ct_datasets,
        segment_number=1
    )
)


# --------------------------------------------------
# 4. Report
# --------------------------------------------------

print()
print("RECONSTRUCTED LIVER MASK")
print("=" * 70)


print()
print("Mask shape:")
print(
    liver_mask.shape
)


print()
print("Matched SEG frames:")
print(
    matched_frames
)


print()
print("Mask values:")
print(
    np.unique(
        liver_mask
    )
)


liver_voxels = np.count_nonzero(
    liver_mask
)


print()
print("Liver voxels:")
print(
    liver_voxels
)


slices_with_liver = np.where(
    np.any(
        liver_mask > 0,
        axis=(0, 1)
    )
)[0]


print()
print("Slices containing liver:")
print(
    len(
        slices_with_liver
    )
)


print()
print("First liver slice:")
print(
    slices_with_liver[0]
)


print()
print("Last liver slice:")
print(
    slices_with_liver[-1]
)


# --------------------------------------------------
# 5. Assertions
# --------------------------------------------------

assert liver_mask.shape == (
    512,
    512,
    len(sorted_ct_datasets)
)

assert matched_frames > 0

assert liver_voxels > 0

assert set(
    np.unique(
        liver_mask
    )
).issubset(
    {0, 1}
)


print()
print(
    "DICOM SEG reconstruction passed."
)