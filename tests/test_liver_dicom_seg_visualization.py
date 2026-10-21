from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pydicom

from src.dicom.io import (
    load_dicom_series
)

from src.dicom.series import (
    sort_dicom_slices
)

from src.dicom.intensity import (
    dicom_pixels_to_hu
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


# --------------------------------------------------
# 2. Reconstruct CT volume in HU
# --------------------------------------------------

ct_slices = []

for dataset in sorted_ct_datasets:

    hu_slice = dicom_pixels_to_hu(
        dataset
    )

    ct_slices.append(
        hu_slice
    )


ct_volume_hu = np.stack(
    ct_slices,
    axis=-1
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
# 4. Reconstruct full liver mask
# --------------------------------------------------

liver_mask, matched_frames = (
    reconstruct_segmentation_volume(
        seg_dataset=seg_dataset,
        sorted_ct_datasets=sorted_ct_datasets,
        segment_number=1
    )
)


# --------------------------------------------------
# 5. Find slice containing most liver
# --------------------------------------------------

liver_voxels_per_slice = np.sum(
    liver_mask,
    axis=(0, 1)
)


best_slice = int(
    np.argmax(
        liver_voxels_per_slice
    )
)


print()
print("LIVER SEG VISUALIZATION")
print("=" * 70)


print()
print("Best liver slice:")
print(
    best_slice
)


print()
print("Liver voxels in slice:")
print(
    liver_voxels_per_slice[
        best_slice
    ]
)


print()
print("Matched SEG frames:")
print(
    matched_frames
)


# --------------------------------------------------
# 6. Extract slice
# --------------------------------------------------

ct_slice = ct_volume_hu[
    :,
    :,
    best_slice
]

mask_slice = liver_mask[
    :,
    :,
    best_slice
]


# --------------------------------------------------
# 7. Liver / soft-tissue window
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


# --------------------------------------------------
# 8. Plot
# --------------------------------------------------

figure, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# CT only
axes[0].imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower",
    vmin=window_min,
    vmax=window_max
)

axes[0].set_title(
    f"Arterial CT - slice {best_slice}"
)


# Mask only
axes[1].imshow(
    mask_slice.T,
    cmap="gray",
    origin="lower"
)

axes[1].set_title(
    "DICOM SEG - Liver mask"
)


# Overlay
axes[2].imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower",
    vmin=window_min,
    vmax=window_max
)

axes[2].imshow(
    mask_slice.T,
    origin="lower",
    alpha=0.35,
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "CT + Liver DICOM SEG"
)


for axis in axes:
    axis.set_axis_off()


plt.tight_layout()
plt.show()