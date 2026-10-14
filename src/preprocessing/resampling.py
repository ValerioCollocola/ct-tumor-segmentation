import numpy as np
from scipy.ndimage import zoom


def resample_volume(
    volume,
    current_spacing,
    target_spacing,
    is_label=False
):
    current_spacing = np.asarray(
        current_spacing,
        dtype=float
    )

    target_spacing = np.asarray(
        target_spacing,
        dtype=float
    )

    zoom_factors = (
        current_spacing / target_spacing
    )

    interpolation_order = 0 if is_label else 1

    resampled_volume = zoom(
        volume,
        zoom=zoom_factors,
        order=interpolation_order
    )

    return resampled_volume, zoom_factors