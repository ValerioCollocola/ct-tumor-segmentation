import numpy as np


def dicom_pixels_to_hu(dataset):
    """
    Convert a CT DICOM pixel array to Hounsfield Units.

    Parameters
    ----------
    dataset : pydicom.dataset.Dataset
        CT DICOM dataset containing pixel data.

    Returns
    -------
    np.ndarray
        Pixel values converted to Hounsfield Units.
    """

    pixel_array = dataset.pixel_array.astype(
        np.float32
    )

    slope = float(
        getattr(
            dataset,
            "RescaleSlope",
            1.0
        )
    )

    intercept = float(
        getattr(
            dataset,
            "RescaleIntercept",
            0.0
        )
    )

    hu_array = (
        pixel_array * slope
        + intercept
    )

    return hu_array