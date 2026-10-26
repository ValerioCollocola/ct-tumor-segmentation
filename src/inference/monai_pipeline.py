from pathlib import Path

import numpy as np
import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
)

from src.inference.monai_sliding_window import (
    monai_sliding_window_probabilities,
)
from src.models.unet3d import UNet3D


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_MIN = -25.0
WINDOW_MAX = 125.0

TARGET_SPACING = (
    1.0,
    1.0,
    2.5,
)

PATCH_SIZE = (
    96,
    96,
    64,
)

OVERLAP = 0.5

DEFAULT_THRESHOLD = 0.5


# ============================================================
# INFERENCE PREPROCESSING
# ============================================================

def build_monai_inference_preprocessing(
    target_spacing=TARGET_SPACING,
):
    """
    Build MONAI preprocessing for inference only.

    Unlike the training preprocessing, no label is required.

    Pipeline:
        load
        channel-first
        intensity windowing / scaling
        resampling
        tensor conversion
    """

    return Compose(
        [
            LoadImaged(
                keys=["image"],
                image_only=False,
            ),

            EnsureChannelFirstd(
                keys=["image"],
            ),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=WINDOW_MIN,
                a_max=WINDOW_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            Spacingd(
                keys=["image"],
                pixdim=target_spacing,
                mode="bilinear",
                align_corners=True,
                scale_extent=True,
            ),

            EnsureTyped(
                keys=["image"],
                dtype=torch.float32,
                track_meta=True,
            ),
        ]
    )


# ============================================================
# MODEL LOADER
# ============================================================

def load_monai_binary_model(
    checkpoint_path,
    device="cpu",
):
    """
    Load the binary 3D U-Net used by the project.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    model = UNet3D(
        in_channels=1,
        num_classes=2,
        base_channels=8,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Support the checkpoint formats used during the project
    # --------------------------------------------------------

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            "Unsupported checkpoint format."
        )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # The dictionary itself may already be a state_dict.
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# PREPROCESS ONE VOLUME
# ============================================================

def preprocess_nifti_for_monai_inference(
    image_path,
    target_spacing=TARGET_SPACING,
):
    """
    Preprocess a NIfTI CT volume with MONAI.

    Returns
    -------
    image
        MONAI MetaTensor with shape:
            [1, X, Y, Z]

    metadata
        Dictionary with useful geometry information.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    transforms = (
        build_monai_inference_preprocessing(
            target_spacing=target_spacing,
        )
    )

    data = transforms(
        {
            "image": str(image_path),
        }
    )

    image = data["image"]

    metadata = {
        "shape": tuple(
            image.shape
        ),
        "spatial_shape": tuple(
            image.shape[1:]
        ),
        "dtype": str(
            image.dtype
        ),
        "min": float(
            image.min()
        ),
        "max": float(
            image.max()
        ),
    }

    # MetaTensor contains affine information.
    if hasattr(
        image,
        "affine",
    ):

        metadata["affine"] = (
            image.affine
            .detach()
            .cpu()
            .numpy()
        )

    return (
        image,
        metadata,
    )


# ============================================================
# COMPLETE INFERENCE PIPELINE
# ============================================================

def run_monai_nifti_inference(
    image_path,
    checkpoint_path,
    threshold=DEFAULT_THRESHOLD,
    target_spacing=TARGET_SPACING,
    patch_size=PATCH_SIZE,
    overlap=OVERLAP,
    sw_batch_size=1,
    device=None,
):
    """
    Full MONAI inference pipeline.

    Steps
    -----
    1. Load NIfTI
    2. Window / normalize intensities
    3. Resample to target spacing
    4. Sliding-window inference
    5. Extract tumor probability
    6. Threshold probability map

    Returns
    -------
    dict
        Contains:
            probability
            mask
            metadata
            threshold
    """

    # --------------------------------------------------------
    # Device selection
    # --------------------------------------------------------

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_monai_binary_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    image, metadata = (
        preprocess_nifti_for_monai_inference(
            image_path=image_path,
            target_spacing=target_spacing,
        )
    )

    # image:
    #
    # [1, X, Y, Z]
    #
    # Our sliding-window wrapper automatically
    # adds the batch dimension.
    # --------------------------------------------------------

    probabilities = (
        monai_sliding_window_probabilities(
            model=model,
            volume=image,
            roi_size=patch_size,
            overlap=overlap,
            sw_batch_size=sw_batch_size,
            device=device,
        )
    )

    # --------------------------------------------------------
    # Binary model:
    #
    # channel 0 = non-tumor
    # channel 1 = tumor
    # --------------------------------------------------------

    tumor_probability = (
        probabilities[
            0,
            1,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    tumor_mask = (
        tumor_probability
        >= threshold
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    metadata[
        "probability_min"
    ] = float(
        tumor_probability.min()
    )

    metadata[
        "probability_max"
    ] = float(
        tumor_probability.max()
    )

    metadata[
        "probability_mean"
    ] = float(
        tumor_probability.mean()
    )

    metadata[
        "positive_voxels"
    ] = int(
        tumor_mask.sum()
    )

    metadata[
        "device"
    ] = str(
        device
    )

    metadata[
        "patch_size"
    ] = tuple(
        patch_size
    )

    metadata[
        "overlap"
    ] = float(
        overlap
    )

    return {
        "probability": tumor_probability,
        "mask": tumor_mask,
        "metadata": metadata,
        "threshold": float(
            threshold
        ),
    }