import numpy as np


def get_slice_position(dataset):
    """
    Compute the spatial position of a DICOM slice
    along the slice-normal direction.

    Uses ImageOrientationPatient and
    ImagePositionPatient.
    """

    orientation = np.asarray(
        dataset.ImageOrientationPatient,
        dtype=np.float64
    )

    position = np.asarray(
        dataset.ImagePositionPatient,
        dtype=np.float64
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]

    slice_normal = np.cross(
        row_direction,
        column_direction
    )

    slice_position = np.dot(
        position,
        slice_normal
    )

    return float(slice_position)


def sort_dicom_slices(datasets):
    """
    Sort DICOM slices according to physical position.
    """

    valid_datasets = []

    for dataset in datasets:

        if not hasattr(
            dataset,
            "ImagePositionPatient"
        ):
            continue

        if not hasattr(
            dataset,
            "ImageOrientationPatient"
        ):
            continue

        valid_datasets.append(dataset)

    sorted_datasets = sorted(
        valid_datasets,
        key=get_slice_position
    )

    return sorted_datasets


def compute_slice_spacing(sorted_datasets):
    """
    Estimate spacing between consecutive DICOM slices.
    """

    if len(sorted_datasets) == 0:
        raise ValueError(
            "No DICOM slices available."
        )

    if len(sorted_datasets) == 1:

        return float(
            getattr(
                sorted_datasets[0],
                "SliceThickness",
                1.0
            )
        )

    positions = np.asarray(
        [
            get_slice_position(dataset)
            for dataset in sorted_datasets
        ],
        dtype=np.float64
    )

    differences = np.abs(
        np.diff(positions)
    )

    slice_spacing = np.median(
        differences
    )

    return float(slice_spacing)


def stack_dicom_slices(sorted_datasets):
    """
    Stack ordered DICOM slices into a 3D NumPy volume.

    Output shape:
    [rows, columns, slices]
    """

    if len(sorted_datasets) == 0:
        raise ValueError(
            "No DICOM slices available."
        )

    reference_shape = (
        sorted_datasets[0]
        .pixel_array
        .shape
    )

    slices = []

    for dataset in sorted_datasets:

        pixel_array = dataset.pixel_array

        if pixel_array.shape != reference_shape:

            raise ValueError(
                "DICOM slices have inconsistent dimensions."
            )

        slices.append(pixel_array)

    volume = np.stack(
        slices,
        axis=-1
    )

    return volume