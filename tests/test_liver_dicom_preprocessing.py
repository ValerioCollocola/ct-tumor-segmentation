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
# 3. Reconstruct volume in HU
# --------------------------------------------------

volume_hu, original_spacing, sorted_datasets = (
    reconstruct_ct_volume(
        ct_series
    )
)


print()
print("ORIGINAL DICOM CT")
print("=" * 60)

print("Shape:")
print(volume_hu.shape)

print("Spacing:")
print(original_spacing)

print("HU range:")
print(
    volume_hu.min(),
    "to",
    volume_hu.max()
)


# --------------------------------------------------
# 4. Liver / soft-tissue windowing
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


print()
print("WINDOWED CT")
print("=" * 60)

print("Expected window:")
print(
    window_min,
    "to",
    window_max
)

print("Actual range:")
print(
    windowed_volume.min(),
    "to",
    windowed_volume.max()
)


# --------------------------------------------------
# 5. Intensity normalization
# --------------------------------------------------

normalized_volume = normalize_intensity(
    windowed_volume,
    min_value=window_min,
    max_value=window_max
)


print()
print("NORMALIZED CT")
print("=" * 60)

print("Range:")
print(
    normalized_volume.min(),
    "to",
    normalized_volume.max()
)


# --------------------------------------------------
# 6. Resampling
# --------------------------------------------------

target_spacing = (
    1.0,
    1.0,
    2.5
)


resampled_volume, zoom_factors = (
    resample_volume(
        normalized_volume,
        current_spacing=original_spacing,
        target_spacing=target_spacing,
        is_label=False
    )
)


print()
print("RESAMPLED CT")
print("=" * 60)

print("Original shape:")
print(volume_hu.shape)

print("Resampled shape:")
print(resampled_volume.shape)

print("Original spacing:")
print(original_spacing)

print("Target spacing:")
print(target_spacing)

print("Zoom factors:")
print(zoom_factors)

print("Range:")
print(
    resampled_volume.min(),
    "to",
    resampled_volume.max()
)


# --------------------------------------------------
# 7. Assertions
# --------------------------------------------------

assert volume_hu.ndim == 3

assert windowed_volume.shape == (
    volume_hu.shape
)

assert normalized_volume.shape == (
    volume_hu.shape
)

assert normalized_volume.min() >= 0.0

assert normalized_volume.max() <= 1.0

assert resampled_volume.ndim == 3

assert len(zoom_factors) == 3


print()
print(
    "Liver DICOM preprocessing test passed."
)