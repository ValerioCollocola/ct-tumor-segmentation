import numpy as np

from src.dicom.series import (
    sort_dicom_slices,
    compute_slice_spacing
)

from src.dicom.intensity import (
    dicom_pixels_to_hu
)


def reconstruct_ct_volume(datasets):
    """
    Reconstruct a 3D CT volume in Hounsfield Units
    from a list of DICOM datasets belonging to the
    same CT series.

    Parameters
    ----------
    datasets : list
        List of pydicom datasets belonging to the
        same CT series.

    Returns
    -------
    volume_hu : np.ndarray
        3D CT volume in Hounsfield Units with shape:
        [rows, columns, slices]

    spacing : tuple
        Voxel spacing:
        (row_spacing, column_spacing, slice_spacing)

    sorted_datasets : list
        DICOM datasets sorted according to their
        physical spatial position.
    """

    if len(datasets) == 0:
        raise ValueError(
            "No DICOM datasets provided."
        )

    # --------------------------------------------------
    # 1. Keep only CT images
    # --------------------------------------------------

    ct_datasets = [
        dataset
        for dataset in datasets
        if getattr(
            dataset,
            "Modality",
            None
        ) == "CT"
    ]

    if len(ct_datasets) == 0:
        raise ValueError(
            "No CT DICOM datasets found."
        )


    # --------------------------------------------------
    # 2. Sort slices using DICOM geometry
    # --------------------------------------------------

    sorted_datasets = sort_dicom_slices(
        ct_datasets
    )

    if len(sorted_datasets) == 0:
        raise ValueError(
            "No CT slices with valid spatial metadata found."
        )


    # --------------------------------------------------
    # 3. Reference image dimensions
    # --------------------------------------------------

    reference_rows = int(
        sorted_datasets[0].Rows
    )

    reference_columns = int(
        sorted_datasets[0].Columns
    )


    # --------------------------------------------------
    # 4. Convert every slice to HU
    # --------------------------------------------------

    hu_slices = []

    for dataset in sorted_datasets:

        rows = int(
            dataset.Rows
        )

        columns = int(
            dataset.Columns
        )

        if (
            rows != reference_rows
            or columns != reference_columns
        ):
            raise ValueError(
                "DICOM slices have inconsistent dimensions."
            )

        hu_slice = dicom_pixels_to_hu(
            dataset
        )

        hu_slices.append(
            hu_slice
        )


    # --------------------------------------------------
    # 5. Stack slices into a 3D volume
    # --------------------------------------------------

    volume_hu = np.stack(
        hu_slices,
        axis=-1
    )


    # --------------------------------------------------
    # 6. Compute voxel spacing
    # --------------------------------------------------

    pixel_spacing = (
        sorted_datasets[0]
        .PixelSpacing
    )

    row_spacing = float(
        pixel_spacing[0]
    )

    column_spacing = float(
        pixel_spacing[1]
    )

    slice_spacing = compute_slice_spacing(
        sorted_datasets
    )

    spacing = (
        row_spacing,
        column_spacing,
        slice_spacing
    )


    return (
        volume_hu,
        spacing,
        sorted_datasets
    )