import numpy as np
import torch
import torch.nn.functional as F


def compute_window_starts(
    volume_size,
    patch_size,
    overlap=0.5
):
    """
    Compute sliding-window starting indices for one axis.

    Parameters
    ----------
    volume_size : int
        Size of the volume along one axis.

    patch_size : int
        Patch size along the same axis.

    overlap : float
        Fractional overlap between consecutive patches.

    Returns
    -------
    list[int]
        Starting indices.
    """

    if not 0 <= overlap < 1:
        raise ValueError(
            "overlap must be in the range [0, 1)."
        )

    if patch_size <= 0:
        raise ValueError(
            "patch_size must be positive."
        )

    if volume_size <= patch_size:
        return [0]

    stride = max(
        1,
        int(
            patch_size * (1 - overlap)
        )
    )

    starts = list(
        range(
            0,
            volume_size - patch_size + 1,
            stride
        )
    )

    final_start = (
        volume_size - patch_size
    )

    if starts[-1] != final_start:
        starts.append(
            final_start
        )

    return starts


def pad_volume_to_patch_size(
    volume,
    patch_size
):
    """
    Pad a 3D volume so every dimension is at least
    as large as the requested patch size.

    Parameters
    ----------
    volume : np.ndarray
        3D volume with shape [X, Y, Z].

    patch_size : tuple[int, int, int]
        Desired patch size.

    Returns
    -------
    padded_volume : np.ndarray
        Padded volume.

    original_shape : tuple
        Shape before padding.
    """

    if volume.ndim != 3:
        raise ValueError(
            "Expected a 3D volume."
        )

    original_shape = volume.shape

    padding = []

    for current_size, required_size in zip(
        volume.shape,
        patch_size
    ):
        missing = max(
            0,
            required_size - current_size
        )

        padding.append(
            (0, missing)
        )

    padded_volume = np.pad(
        volume,
        pad_width=padding,
        mode="constant",
        constant_values=0
    )

    return (
        padded_volume,
        original_shape
    )


def sliding_window_inference(
    model,
    volume,
    patch_size=(96, 96, 64),
    overlap=0.5,
    device="cpu"
):
    """
    Run full-volume binary segmentation using
    overlapping 3D patches.

    The model is expected to return logits with shape:

        [B, 2, X, Y, Z]

    Parameters
    ----------
    model : torch.nn.Module
        Binary segmentation model.

    volume : np.ndarray
        Preprocessed 3D input volume with shape
        [X, Y, Z].

    patch_size : tuple
        Sliding-window patch size.

    overlap : float
        Fractional overlap between patches.

    device : str
        PyTorch device.

    Returns
    -------
    tumor_probability : np.ndarray
        Full-volume tumor probability map.

    prediction_mask : np.ndarray
        Binary prediction using probability >= 0.5.
    """

    model = model.to(
        device
    )

    model.eval()

    padded_volume, original_shape = (
        pad_volume_to_patch_size(
            volume,
            patch_size
        )
    )

    x_starts = compute_window_starts(
        padded_volume.shape[0],
        patch_size[0],
        overlap
    )

    y_starts = compute_window_starts(
        padded_volume.shape[1],
        patch_size[1],
        overlap
    )

    z_starts = compute_window_starts(
        padded_volume.shape[2],
        patch_size[2],
        overlap
    )

    probability_sum = np.zeros(
        padded_volume.shape,
        dtype=np.float32
    )

    count_map = np.zeros(
        padded_volume.shape,
        dtype=np.float32
    )

    total_patches = (
        len(x_starts)
        * len(y_starts)
        * len(z_starts)
    )

    processed_patches = 0

    print()
    print("Sliding-window inference")
    print("=" * 60)

    print("Volume shape:")
    print(padded_volume.shape)

    print("Patch size:")
    print(patch_size)

    print("Overlap:")
    print(overlap)

    print("Total patches:")
    print(total_patches)

    with torch.no_grad():

        for x_start in x_starts:

            x_end = (
                x_start
                + patch_size[0]
            )

            for y_start in y_starts:

                y_end = (
                    y_start
                    + patch_size[1]
                )

                for z_start in z_starts:

                    z_end = (
                        z_start
                        + patch_size[2]
                    )

                    patch = padded_volume[
                        x_start:x_end,
                        y_start:y_end,
                        z_start:z_end
                    ]

                    patch_tensor = torch.from_numpy(
                        patch.astype(
                            np.float32
                        )
                    )

                    patch_tensor = (
                        patch_tensor
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )

                    logits = model(
                        patch_tensor
                    )

                    probabilities = F.softmax(
                        logits,
                        dim=1
                    )

                    tumor_probability = (
                        probabilities[
                            0,
                            1
                        ]
                        .cpu()
                        .numpy()
                    )

                    probability_sum[
                        x_start:x_end,
                        y_start:y_end,
                        z_start:z_end
                    ] += tumor_probability

                    count_map[
                        x_start:x_end,
                        y_start:y_end,
                        z_start:z_end
                    ] += 1.0

                    processed_patches += 1

    if np.any(
        count_map == 0
    ):
        raise RuntimeError(
            "Some voxels were not covered by any patch."
        )

    averaged_probability = (
        probability_sum
        / count_map
    )

    averaged_probability = (
        averaged_probability[
            :original_shape[0],
            :original_shape[1],
            :original_shape[2]
        ]
    )

    prediction_mask = (
        averaged_probability >= 0.5
    ).astype(
        np.uint8
    )

    print()
    print("Processed patches:")
    print(processed_patches)

    print("Tumor probability range:")
    print(
        averaged_probability.min(),
        "to",
        averaged_probability.max()
    )

    return (
        averaged_probability,
        prediction_mask
    )