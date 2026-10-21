import matplotlib.pyplot as plt

from src.dicom.io import (
    load_dicom_series,
    group_by_series
)

from src.dicom.reconstruction import (
    reconstruct_ct_volume
)


dicom_directory = (
    "data/dicom/liver_portal_venous"
)


datasets = load_dicom_series(
    dicom_directory
)

series_dict = group_by_series(
    datasets
)


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


volume_hu, spacing, sorted_datasets = (
    reconstruct_ct_volume(
        ct_series
    )
)


axial_index = (
    volume_hu.shape[2] // 2
)

coronal_index = (
    volume_hu.shape[0] // 2
)

sagittal_index = (
    volume_hu.shape[1] // 2
)


axial = volume_hu[
    :,
    :,
    axial_index
]

coronal = volume_hu[
    coronal_index,
    :,
    :
]

sagittal = volume_hu[
    :,
    sagittal_index,
    :
]


# Liver / soft-tissue window
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


figure, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


axes[0].imshow(
    axial.T,
    cmap="gray",
    origin="lower",
    vmin=window_min,
    vmax=window_max
)

axes[0].set_title(
    f"Axial - slice {axial_index}"
)


axes[1].imshow(
    coronal.T,
    cmap="gray",
    origin="lower",
    aspect=(
        spacing[2]
        / spacing[1]
    ),
    vmin=window_min,
    vmax=window_max
)

axes[1].set_title(
    f"Coronal - row {coronal_index}"
)


axes[2].imshow(
    sagittal.T,
    cmap="gray",
    origin="lower",
    aspect=(
        spacing[2]
        / spacing[0]
    ),
    vmin=window_min,
    vmax=window_max
)

axes[2].set_title(
    f"Sagittal - column {sagittal_index}"
)


for axis in axes:
    axis.set_axis_off()


plt.tight_layout()
plt.show()