from pathlib import Path
from collections import defaultdict

import pydicom
from pydicom.errors import InvalidDicomError


def load_dicom_file(file_path):
    """
    Load a single DICOM file.

    Parameters
    ----------
    file_path : str or Path
        Path to the DICOM file.

    Returns
    -------
    pydicom.dataset.FileDataset
        Loaded DICOM dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"DICOM file not found: {file_path}"
        )

    dataset = pydicom.dcmread(
        file_path
    )

    return dataset


def extract_basic_metadata(dataset):
    """
    Extract basic metadata from a DICOM dataset.
    """

    metadata = {
        "patient_id": getattr(
            dataset,
            "PatientID",
            None
        ),

        "study_instance_uid": getattr(
            dataset,
            "StudyInstanceUID",
            None
        ),

        "series_instance_uid": getattr(
            dataset,
            "SeriesInstanceUID",
            None
        ),

        "modality": getattr(
            dataset,
            "Modality",
            None
        ),

        "rows": getattr(
            dataset,
            "Rows",
            None
        ),

        "columns": getattr(
            dataset,
            "Columns",
            None
        ),

        "pixel_spacing": getattr(
            dataset,
            "PixelSpacing",
            None
        ),

        "slice_thickness": getattr(
            dataset,
            "SliceThickness",
            None
        ),

        "instance_number": getattr(
            dataset,
            "InstanceNumber",
            None
        ),

        "image_position_patient": getattr(
            dataset,
            "ImagePositionPatient",
            None
        ),

        "image_orientation_patient": getattr(
            dataset,
            "ImageOrientationPatient",
            None
        ),

        "rescale_slope": getattr(
            dataset,
            "RescaleSlope",
            None
        ),

        "rescale_intercept": getattr(
            dataset,
            "RescaleIntercept",
            None
        )
    }

    return metadata


def discover_dicom_files(directory):
    """
    Discover readable DICOM files inside a directory.

    The function does not rely on the .dcm extension.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"DICOM directory not found: {directory}"
        )

    dicom_files = []

    for file_path in directory.rglob("*"):

        if not file_path.is_file():
            continue

        try:
            dataset = pydicom.dcmread(
                file_path,
                stop_before_pixels=True
            )

            if hasattr(
                dataset,
                "SOPClassUID"
            ):
                dicom_files.append(
                    file_path
                )

        except InvalidDicomError:
            continue

        except Exception:
            continue

    return dicom_files


def load_dicom_series(directory):
    """
    Load all readable DICOM files from a directory.

    Returns
    -------
    list
        List of pydicom datasets.
    """

    dicom_files = discover_dicom_files(
        directory
    )

    datasets = []

    for file_path in dicom_files:

        dataset = pydicom.dcmread(
            file_path
        )

        datasets.append(
            dataset
        )

    return datasets


def group_by_series(datasets):
    """
    Group DICOM datasets by SeriesInstanceUID.

    Returns
    -------
    dict
        Dictionary where each key is a SeriesInstanceUID
        and each value is a list of DICOM datasets.
    """

    series = defaultdict(
        list
    )

    for dataset in datasets:

        series_uid = getattr(
            dataset,
            "SeriesInstanceUID",
            None
        )

        if series_uid is not None:

            series[
                str(series_uid)
            ].append(
                dataset
            )

    return dict(
        series
    )