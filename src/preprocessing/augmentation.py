import numpy as np


def random_flip_3d(
    image,
    label,
    axis,
):
    flipped_image = np.flip(
        image,
        axis=axis
    )

    flipped_label = np.flip(
        label,
        axis=axis
    )

    return flipped_image, flipped_label