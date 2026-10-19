import numpy as np


def extract_3d_patch(
    volume,
    center,
    patch_size
):
    center = np.asarray(center, dtype=int)
    patch_size = np.asarray(patch_size, dtype=int)
    volume_shape = np.asarray(volume.shape, dtype=int)

    # Controlla che la patch possa entrare nel volume
    if np.any(patch_size > volume_shape):
        raise ValueError(
            "Patch size cannot be larger than volume size."
        )

    half_size = patch_size // 2

    # Posizione iniziale ideale
    start = center - half_size

    # Impedisce di uscire dal bordo inferiore
    start = np.maximum(
        start,
        0
    )

    # Massima posizione iniziale consentita
    max_start = volume_shape - patch_size

    # Impedisce di uscire dal bordo superiore
    start = np.minimum(
        start,
        max_start
    )

    end = start + patch_size

    patch = volume[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ]

    return patch


def find_label_center(
    label,
    label_value=2
):
    coordinates = np.argwhere(
        label == label_value
    )

    if len(coordinates) == 0:
        raise ValueError(
            f"Label value {label_value} not found."
        )

    center = np.round(
        coordinates.mean(axis=0)
    ).astype(int)

    return tuple(center)

def generate_shifted_centers(
    center,
    num_patches=5,
    max_shift=(12, 12, 8),
    seed=42
):
    rng = np.random.default_rng(seed)

    center = np.asarray(
        center,
        dtype=int
    )

    max_shift = np.asarray(
        max_shift,
        dtype=int
    )

    centers = [
        tuple(center)
    ]

    for _ in range(num_patches - 1):

        shift = rng.integers(
            low=-max_shift,
            high=max_shift + 1
        )

        shifted_center = (
            center + shift
        )

        centers.append(
            tuple(shifted_center)
        )

    return centers