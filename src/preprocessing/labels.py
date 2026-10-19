import numpy as np


def convert_to_binary_tumor_mask(
    label,
    tumor_label=2
):
    binary_label = (
        label == tumor_label
    ).astype(np.int64)

    return binary_label