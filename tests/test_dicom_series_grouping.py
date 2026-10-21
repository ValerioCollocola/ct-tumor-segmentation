
from src.dicom.io import (
    load_dicom_series,
    group_by_series
)


dicom_directory = (
    "data/dicom/lidc_idri"
)


datasets = load_dicom_series(
    dicom_directory
)


series = group_by_series(
    datasets
)


print()
print("Total DICOM files:")
print(len(datasets))


print()
print("Detected DICOM series:")
print(len(series))


for series_uid, series_datasets in series.items():

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