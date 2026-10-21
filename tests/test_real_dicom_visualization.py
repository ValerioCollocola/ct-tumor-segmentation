import matplotlib.pyplot as plt

from src.dicom.io import (
    load_dicom_series,
    group_by_series
)

from src.dicom.reconstruction import (
    reconstruct_ct_volume
)


dicom_directory = (
    "data/dicom/lidc_idri"
)


# --------------------------------------------------
# 1. Load DICOM files
# --------------------------------------------------

datasets = load_dicom_series(
    dicom_directory
)


# --------------------------------------------------
# 2. Group by series
# --------------------------------------------------

series_dict = group_by_series(
    datasets
)


# --------------------------------------------------
# 3. Select CT series
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
# 4. Reconstruct volume
# --------------------------------------------------

volume_hu, spacing, sorted_datasets = (
    reconstruct_ct_volume(
        ct_series
    )
)


# --------------------------------------------------
# 5. Select central slice
# --------------------------------------------------

slice_index = (
    volume_hu.shape[-1] // 2
)

ct_slice = volume_hu[
    :,
    :,
    slice_index
]


print()
print("Central slice index:")
print(slice_index)

print()
print("Slice HU range:")
print(
    ct_slice.min(),
    "to",
    ct_slice.max()
)

print()
print("Voxel spacing:")
print(spacing)


# --------------------------------------------------
# 6. Display using a soft-tissue window
# --------------------------------------------------

window_center = 40
window_width = 400

window_min = (
    window_center
    - window_width / 2
)

window_max = (
    window_center
    + window_width / 2
)


plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower",
    vmin=window_min,
    vmax=window_max
)

plt.title(
    f"Real DICOM CT - Slice {slice_index}"
)

plt.xlabel(
    "X"
)

plt.ylabel(
    "Y"
)

plt.tight_layout()

plt.show()