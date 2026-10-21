import numpy as np

from src.dicom.io import (
    load_dicom_series,
    group_by_series
)

from src.dicom.series import (
    sort_dicom_slices,
    get_slice_position
)


dicom_directory = (
    "data/dicom/lidc_idri"
)


# --------------------------------------------------
# 1. Load and group DICOM datasets
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
# 3. Sort slices geometrically
# --------------------------------------------------

sorted_datasets = sort_dicom_slices(
    ct_series
)


positions = np.asarray(
    [
        get_slice_position(dataset)
        for dataset in sorted_datasets
    ],
    dtype=np.float64
)


spacing_differences = np.diff(
    positions
)


absolute_spacing = np.abs(
    spacing_differences
)


# --------------------------------------------------
# 4. Report
# --------------------------------------------------

print()
print("REAL DICOM GEOMETRY CHECK")
print("=" * 60)


print()
print("Number of slices:")
print(len(sorted_datasets))


print()
print("First position:")
print(positions[0])


print()
print("Last position:")
print(positions[-1])


print()
print("Minimum slice spacing:")
print(
    absolute_spacing.min()
)


print()
print("Maximum slice spacing:")
print(
    absolute_spacing.max()
)


print()
print("Mean slice spacing:")
print(
    absolute_spacing.mean()
)


print()
print("Median slice spacing:")
print(
    np.median(
        absolute_spacing
    )
)


print()
print("Standard deviation:")
print(
    absolute_spacing.std()
)


# --------------------------------------------------
# 5. Check for irregular spacing
# --------------------------------------------------

median_spacing = np.median(
    absolute_spacing
)

tolerance = 1e-3

irregular_intervals = np.where(
    np.abs(
        absolute_spacing
        - median_spacing
    ) > tolerance
)[0]


print()
print("Irregular intervals:")
print(
    len(irregular_intervals)
)


if len(irregular_intervals) > 0:

    print()
    print("Irregular interval indices:")
    print(irregular_intervals)


# --------------------------------------------------
# 6. Assertions
# --------------------------------------------------

assert len(sorted_datasets) == 133

assert median_spacing > 0

assert len(irregular_intervals) == 0


print()
print("Geometry check passed.")