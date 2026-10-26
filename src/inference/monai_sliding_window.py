import torch

from monai.inferers import sliding_window_inference


DEFAULT_ROI_SIZE = (96, 96, 64)
DEFAULT_OVERLAP = 0.5


def monai_sliding_window_logits(
    model,
    volume,
    roi_size=DEFAULT_ROI_SIZE,
    overlap=DEFAULT_OVERLAP,
    sw_batch_size=1,
    device="cpu",
):
    """
    Run MONAI sliding-window inference on a full 3D volume.

    Parameters
    ----------
    model
        PyTorch segmentation model.

    volume
        NumPy array or torch.Tensor with shape:
            (X, Y, Z)
        or:
            (1, X, Y, Z)

    roi_size
        Sliding-window patch size.

    overlap
        Fractional overlap between windows.

    sw_batch_size
        Number of windows processed in one model forward pass.

    device
        CPU or CUDA device.

    Returns
    -------
    torch.Tensor
        Full-volume logits with shape:
            (1, C, X, Y, Z)
    """

    model = model.to(device)
    model.eval()

    if not torch.is_tensor(volume):
        volume = torch.as_tensor(
            volume,
            dtype=torch.float32,
        )
    else:
        volume = volume.float()

    # --------------------------------------------------
    # Convert to MONAI expected shape:
    #
    # [X,Y,Z]       -> [1,1,X,Y,Z]
    # [1,X,Y,Z]     -> [1,1,X,Y,Z]
    # --------------------------------------------------

    if volume.ndim == 3:

        volume = (
            volume
            .unsqueeze(0)
            .unsqueeze(0)
        )

    elif volume.ndim == 4:

        volume = volume.unsqueeze(0)

    elif volume.ndim != 5:

        raise ValueError(
            "Expected volume with 3, 4 or 5 dimensions. "
            f"Received shape: {tuple(volume.shape)}"
        )

    volume = volume.to(device)

    with torch.no_grad():

        logits = sliding_window_inference(
            inputs=volume,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,

            # Equal weighting in overlapping areas.
            # Closest to our custom probability averaging.
            mode="constant",

            padding_mode="constant",
            cval=0.0,
        )

    return logits


def monai_sliding_window_probabilities(
    model,
    volume,
    roi_size=DEFAULT_ROI_SIZE,
    overlap=DEFAULT_OVERLAP,
    sw_batch_size=1,
    device="cpu",
):
    """
    Run sliding-window inference and convert logits
    to class probabilities.
    """

    logits = monai_sliding_window_logits(
        model=model,
        volume=volume,
        roi_size=roi_size,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        device=device,
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    return probabilities