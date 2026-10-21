from pathlib import Path

import pydicom

from src.dicom.io import (
    load_dicom_series,
    group_by_series
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

seg_directory = Path(
    "data/dicom/liver_seg"
)

dicom_root = Path(
    "data/dicom"
)


# --------------------------------------------------
# 1. Load SEG
# --------------------------------------------------

seg_dataset = None

for file_path in seg_directory.rglob("*"):

    if not file_path.is_file():
        continue

    try:
        dataset = pydicom.dcmread(
            file_path,
            stop_before_pixels=True
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
# 2. Read referenced SeriesInstanceUID
# --------------------------------------------------

referenced_series_sequence = getattr(
    seg_dataset,
    "ReferencedSeriesSequence",
    []
)


if len(referenced_series_sequence) == 0:

    raise RuntimeError(
        "SEG contains no ReferencedSeriesSequence."
    )


referenced_series_uid = str(
    referenced_series_sequence[0]
    .SeriesInstanceUID
)


print()
print("SEG REFERENCE MATCH")
print("=" * 70)

print()
print("SEG referenced SeriesInstanceUID:")
print(
    referenced_series_uid
)


# --------------------------------------------------
# 3. Scan local DICOM directories
# --------------------------------------------------

matched_series = None
matched_directory = None


for directory in dicom_root.iterdir():

    if not directory.is_dir():
        continue

    if directory == seg_directory:
        continue

    try:

        datasets = load_dicom_series(
            directory
        )

    except Exception:
        continue

    if len(datasets) == 0:
        continue

    series_dict = group_by_series(
        datasets
    )

    for series_uid, series_datasets in (
        series_dict.items()
    ):

        first_dataset = (
            series_datasets[0]
        )

        modality = getattr(
            first_dataset,
            "Modality",
            None
        )

        description = getattr(
            first_dataset,
            "SeriesDescription",
            None
        )

        print()
        print("Local directory:")
        print(directory)

        print("Series UID:")
        print(series_uid)

        print("Modality:")
        print(modality)

        print("Description:")
        print(description)

        print("Images:")
        print(
            len(series_datasets)
        )

        if (
            str(series_uid)
            == referenced_series_uid
        ):

            matched_series = (
                series_datasets
            )

            matched_directory = (
                directory
            )


# --------------------------------------------------
# 4. Result
# --------------------------------------------------

print()
print("=" * 70)


if matched_series is None:

    print(
        "Referenced CT series is NOT "
        "currently available locally."
    )

else:

    first_dataset = matched_series[0]

    print(
        "MATCH FOUND"
    )

    print()
    print("Directory:")
    print(
        matched_directory
    )

    print()
    print("Description:")
    print(
        getattr(
            first_dataset,
            "SeriesDescription",
            None
        )
    )

    print()
    print("Number of images:")
    print(
        len(matched_series)
    )