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


# --------------------------------------------------
# 1. Load DICOM datasets
# --------------------------------------------------

datasets = load_dicom_series(
    dicom_directory
)


print()
print("Total DICOM files:")
print(len(datasets))


# --------------------------------------------------
# 2. Group by SeriesInstanceUID
# --------------------------------------------------

series_dict = group_by_series(
    datasets
)


print()
print("Detected DICOM series:")
print(len(series_dict))


# --------------------------------------------------
# 3. Select CT series
# --------------------------------------------------

ct_series = None
ct_series_uid = None

for series_uid, series_datasets in series_dict.items():

    modality = getattr(
        series_datasets[0],
        "Modality",
        None
    )

    description = getattr(
        series_datasets[0],
        "SeriesDescription",
        None
    )

    print()
    print("Series UID:")
    print(series_uid)

    print("Modality:")
    print(modality)

    print("Description:")
    print(description)

    print("Number of images:")
    print(len(series_datasets))

    if modality == "CT":

        ct_series = series_datasets
        ct_series_uid = series_uid


if ct_series is None:

    raise RuntimeError(
        "No CT series found."
    )


# --------------------------------------------------
# 4. Reconstruct liver CT volume
# --------------------------------------------------

volume_hu, spacing, sorted_datasets = (
    reconstruct_ct_volume(
        ct_series
    )
)


# --------------------------------------------------
# 5. Report
# --------------------------------------------------

print()
print("LIVER PORTAL-VENOUS DICOM CT")
print("=" * 60)


print()
print("Series UID:")
print(ct_series_uid)


print()
print("Number of slices:")
print(len(sorted_datasets))


print()
print("Volume shape:")
print(volume_hu.shape)


print()
print("Voxel spacing:")
print(spacing)


print()
print("HU range:")
print(
    volume_hu.min(),
    "to",
    volume_hu.max()
)


print()
print("Mean HU:")
print(
    volume_hu.mean()
)


# --------------------------------------------------
# 6. Metadata checks
# --------------------------------------------------

first_slice = sorted_datasets[0]
last_slice = sorted_datasets[-1]


print()
print("First slice position:")
print(
    first_slice.ImagePositionPatient
)


print()
print("Last slice position:")
print(
    last_slice.ImagePositionPatient
)


print()
print("Pixel spacing:")
print(
    first_slice.PixelSpacing
)


print()
print("Slice thickness:")
print(
    getattr(
        first_slice,
        "SliceThickness",
        None
    )
)


print()
print("Rescale slope:")
print(
    getattr(
        first_slice,
        "RescaleSlope",
        None
    )
)


print()
print("Rescale intercept:")
print(
    getattr(
        first_slice,
        "RescaleIntercept",
        None
    )
)


print()
print("Series description:")
print(
    getattr(
        first_slice,
        "SeriesDescription",
        None
    )
)


# --------------------------------------------------
# 7. Assertions
# --------------------------------------------------

assert volume_hu.ndim == 3

assert volume_hu.shape[-1] == len(
    sorted_datasets
)

assert len(sorted_datasets) > 1

assert len(spacing) == 3

assert spacing[0] > 0
assert spacing[1] > 0
assert spacing[2] > 0


print()
print(
    "Liver DICOM reconstruction test passed."
)