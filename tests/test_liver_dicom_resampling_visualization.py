import matplotlib.pyplot as plt

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


dicom_directory = (
    "data/dicom/liver_portal_venous"
)


# --------------------------------------------------
# 1. Load and select CT series
# --------------------------------------------------

datasets = load_dicom_series(
    dicom_directory
)

series_dict = group_by_series(
    datasets
)

ct_series = None

for _, series_datasets in series_dict.items():

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
# 2. Reconstruct in HU
# --------------------------------------------------

volume_hu, original_spacing, _ = (
    reconstruct_ct_volume(
        ct_series
    )
)


# --------------------------------------------------
# 3. Window + normalize
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


# --------------------------------------------------
# 4. Resample
# --------------------------------------------------

target_spacing = (
    1.0,
    1.0,
    2.5
)


resampled, zoom_factors = (
    resample_volume(
        normalized,
        current_spacing=original_spacing,
        target_spacing=target_spacing,
        is_label=False
    )
)


# --------------------------------------------------
# 5. Select central coronal planes
# --------------------------------------------------

original_coronal_index = (
    normalized.shape[0] // 2
)

resampled_coronal_index = (
    resampled.shape[0] // 2
)


original_coronal = normalized[
    original_coronal_index,
    :,
    :
]

resampled_coronal = resampled[
    resampled_coronal_index,
    :,
    :
]


# --------------------------------------------------
# 6. Plot
# --------------------------------------------------

figure, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)


axes[0].imshow(
    original_coronal.T,
    cmap="gray",
    origin="lower",
    aspect=(
        original_spacing[2]
        / original_spacing[1]
    )
)

axes[0].set_title(
    "Before resampling\n"
    f"{original_spacing}"
)


axes[1].imshow(
    resampled_coronal.T,
    cmap="gray",
    origin="lower",
    aspect=(
        target_spacing[2]
        / target_spacing[1]
    )
)

axes[1].set_title(
    "After resampling\n"
    f"{target_spacing}"
)


for axis in axes:
    axis.set_axis_off()


plt.tight_layout()
plt.show()