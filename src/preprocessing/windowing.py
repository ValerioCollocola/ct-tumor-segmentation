import numpy as np


def apply_ct_window(image, center, width):
    lower = center - width / 2
    upper = center + width / 2

    windowed_image = np.clip(image, lower, upper)

    return windowed_image