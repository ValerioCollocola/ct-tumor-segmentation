from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.inference.monai_sliding_window import (
    monai_sliding_window_probabilities,
)
from src.inference.sliding_window import (
    sliding_window_inference as custom_sliding_window_inference,
)
from src.models.unet3d import UNet3D
from src.preprocessing.normalization import normalize_intensity
from src.preprocessing.resampling import resample_volume


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = Path(
    "data/multicase/images/liver_7.nii.gz"
)

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/"
    "best_real_binary_unet_best_tumor_dice.pt"
)

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

THRESHOLD = 0.5

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL LOADER
# ============================================================

def load_binary_model():

    print()
    print("Device:")
    print(DEVICE)

    model = UNet3D(
        in_channels=1,
        num_classes=2,
        base_channels=8,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            # The checkpoint itself may already be
            # a state_dict.
            state_dict = checkpoint

    else:

        raise TypeError(
            "Unsupported checkpoint format."
        )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def load_preprocessed_volume():

    image_nii = nib.load(
        IMAGE_PATH
    )

    image = (
        image_nii
        .get_fdata()
        .astype(np.float32)
    )

    current_spacing = np.asarray(
        image_nii.header.get_zooms()[:3],
        dtype=float,
    )

    print()
    print("Original shape:")
    print(image.shape)

    print()
    print("Original spacing:")
    print(current_spacing)

    # --------------------------------------------------------
    # Liver CT window
    # --------------------------------------------------------

    windowed = np.clip(
        image,
        -25.0,
        125.0,
    )

    # --------------------------------------------------------
    # Normalize to [0, 1]
    # --------------------------------------------------------

    normalized = normalize_intensity(
        windowed,
        -25.0,
        125.0,
    )

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    resampled, _ = resample_volume(
        normalized,
        current_spacing=current_spacing,
        target_spacing=TARGET_SPACING,
        is_label=False,
    )

    resampled = (
        resampled
        .astype(np.float32)
    )

    print()
    print("Preprocessed shape:")
    print(resampled.shape)

    print()
    print("Preprocessed range:")
    print(
        float(resampled.min()),
        float(resampled.max()),
    )

    return resampled


# ============================================================
# CUSTOM OUTPUT DIAGNOSTICS
# ============================================================

def inspect_custom_output(
    custom_output,
):

    print()
    print("=" * 70)
    print("CUSTOM RAW OUTPUT DIAGNOSTICS")
    print("=" * 70)

    print()
    print("Custom raw output type:")
    print(type(custom_output))

    if isinstance(
        custom_output,
        tuple,
    ):

        print()
        print("Tuple length:")
        print(len(custom_output))

        for index, item in enumerate(
            custom_output
        ):

            print()
            print(
                f"Item {index} type:"
            )
            print(
                type(item)
            )

            if hasattr(
                item,
                "shape",
            ):

                print(
                    f"Item {index} shape:"
                )
                print(
                    item.shape
                )

            if torch.is_tensor(
                item
            ):

                print(
                    f"Item {index} range:"
                )
                print(
                    float(item.min()),
                    float(item.max()),
                )

            elif isinstance(
                item,
                np.ndarray,
            ):

                print(
                    f"Item {index} range:"
                )
                print(
                    float(item.min()),
                    float(item.max()),
                )

    else:

        print()
        print("Raw output shape:")
        print(
            getattr(
                custom_output,
                "shape",
                None,
            )
        )

        if torch.is_tensor(
            custom_output
        ):

            print()
            print("Raw output range:")
            print(
                float(
                    custom_output.min()
                ),
                float(
                    custom_output.max()
                ),
            )

        elif isinstance(
            custom_output,
            np.ndarray,
        ):

            print()
            print("Raw output range:")
            print(
                float(
                    custom_output.min()
                ),
                float(
                    custom_output.max()
                ),
            )


# ============================================================
# PROBABILITY EXTRACTION
# ============================================================

def extract_tumor_probability(
    prediction,
):
    """
    Extract the tumor probability map from the output
    of the custom sliding-window inference.

    Supported formats:
        - tuple:
            (tumor_probability, binary_prediction, ...)
        - [X, Y, Z]
        - [1, X, Y, Z]
        - [2, X, Y, Z]
        - [1, 2, X, Y, Z]
    """

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The custom sliding-window implementation can return:
    #
    # (
    #     tumor_probability_map,
    #     binary_prediction,
    # )
    #
    # The probability map is the FIRST element.
    # --------------------------------------------------------

    if isinstance(
        prediction,
        tuple,
    ):

        if len(prediction) == 0:

            raise ValueError(
                "Custom inference returned "
                "an empty tuple."
            )

        prediction = prediction[0]

    # --------------------------------------------------------
    # Torch -> NumPy
    # --------------------------------------------------------

    if torch.is_tensor(
        prediction
    ):

        prediction = (
            prediction
            .detach()
            .cpu()
            .numpy()
        )

    prediction = np.asarray(
        prediction
    )

    # --------------------------------------------------------
    # [B, C, X, Y, Z]
    # --------------------------------------------------------

    if prediction.ndim == 5:

        if prediction.shape[0] != 1:

            raise ValueError(
                "Expected batch size 1. "
                f"Received shape: "
                f"{prediction.shape}"
            )

        prediction = prediction[0]

    # --------------------------------------------------------
    # [C, X, Y, Z]
    # --------------------------------------------------------

    if prediction.ndim == 4:

        # Two-channel logits/probabilities.
        if prediction.shape[0] == 2:

            # If values are outside [0,1],
            # assume logits and apply softmax.
            if (
                prediction.min() < 0.0
                or prediction.max() > 1.0
            ):

                tensor = torch.from_numpy(
                    prediction
                )

                prediction = (
                    torch.softmax(
                        tensor,
                        dim=0,
                    )
                    .numpy()
                )

            return (
                prediction[1]
                .astype(np.float32)
            )

        # Single-channel probability volume.
        if prediction.shape[0] == 1:

            return (
                prediction[0]
                .astype(np.float32)
            )

    # --------------------------------------------------------
    # Already [X, Y, Z]
    # --------------------------------------------------------

    if prediction.ndim == 3:

        return prediction.astype(
            np.float32
        )

    raise ValueError(
        "Unexpected prediction format. "
        f"Shape: {prediction.shape}"
    )


# ============================================================
# DICE BETWEEN TWO BINARY MASKS
# ============================================================

def dice_between_masks(
    mask_a,
    mask_b,
):

    mask_a = mask_a.astype(
        bool
    )

    mask_b = mask_b.astype(
        bool
    )

    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    denominator = (
        mask_a.sum()
        + mask_b.sum()
    )

    if denominator == 0:

        return 1.0

    return float(
        2.0
        * intersection
        / denominator
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)

    print(
        "CUSTOM VS MONAI "
        "SLIDING-WINDOW INFERENCE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    assert IMAGE_PATH.exists(), (
        f"Missing image: "
        f"{IMAGE_PATH}"
    )

    assert CHECKPOINT_PATH.exists(), (
        f"Missing checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_binary_model()

    # --------------------------------------------------------
    # Load + preprocess volume
    # --------------------------------------------------------

    volume = (
        load_preprocessed_volume()
    )

    # ========================================================
    # CUSTOM SLIDING WINDOW
    # ========================================================

    print()
    print("=" * 70)
    print("CUSTOM SLIDING WINDOW")
    print("=" * 70)

    custom_output = (
        custom_sliding_window_inference(
            model=model,
            volume=volume,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            device=DEVICE,
        )
    )

    # --------------------------------------------------------
    # Diagnostic print
    # --------------------------------------------------------

    inspect_custom_output(
        custom_output
    )

    # --------------------------------------------------------
    # Extract actual probability map
    # --------------------------------------------------------

    custom_probability = (
        extract_tumor_probability(
            custom_output
        )
    )

    print()
    print("Custom probability shape:")
    print(
        custom_probability.shape
    )

    print()
    print("Custom probability range:")
    print(
        float(
            custom_probability.min()
        ),
        float(
            custom_probability.max()
        ),
    )

    # ========================================================
    # MONAI SLIDING WINDOW
    # ========================================================

    print()
    print("=" * 70)
    print("MONAI SLIDING WINDOW")
    print("=" * 70)

    monai_output = (
        monai_sliding_window_probabilities(
            model=model,
            volume=volume,
            roi_size=PATCH_SIZE,
            overlap=OVERLAP,
            sw_batch_size=1,
            device=DEVICE,
        )
    )

    monai_probability = (
        monai_output[
            0,
            1,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    print()
    print("MONAI probability shape:")
    print(
        monai_probability.shape
    )

    print()
    print("MONAI probability range:")
    print(
        float(
            monai_probability.min()
        ),
        float(
            monai_probability.max()
        ),
    )

    # ========================================================
    # GEOMETRY CHECK
    # ========================================================

    assert (
        custom_probability.shape
        == monai_probability.shape
    )

    assert (
        custom_probability.shape
        == volume.shape
    )

    # ========================================================
    # PROBABILITY COMPARISON
    # ========================================================

    absolute_difference = np.abs(
        custom_probability
        - monai_probability
    )

    probability_mae = float(
        absolute_difference.mean()
    )

    probability_max_difference = float(
        absolute_difference.max()
    )

    probability_median_difference = float(
        np.median(
            absolute_difference
        )
    )

    print()
    print("=" * 70)
    print("PROBABILITY COMPARISON")
    print("=" * 70)

    print()
    print("Probability MAE:")
    print(
        probability_mae
    )

    print()
    print(
        "Median probability difference:"
    )
    print(
        probability_median_difference
    )

    print()
    print(
        "Maximum probability difference:"
    )
    print(
        probability_max_difference
    )

    # ========================================================
    # BINARY MASK COMPARISON
    # ========================================================

    custom_mask = (
        custom_probability
        >= THRESHOLD
    )

    monai_mask = (
        monai_probability
        >= THRESHOLD
    )

    custom_positive = int(
        custom_mask.sum()
    )

    monai_positive = int(
        monai_mask.sum()
    )

    mask_agreement = float(
        np.mean(
            custom_mask
            == monai_mask
        )
    )

    mask_dice = (
        dice_between_masks(
            custom_mask,
            monai_mask,
        )
    )

    mismatched_mask_voxels = int(
        np.sum(
            custom_mask
            != monai_mask
        )
    )

    print()
    print("=" * 70)
    print("MASK COMPARISON")
    print("=" * 70)

    print()
    print("Threshold:")
    print(
        THRESHOLD
    )

    print()
    print(
        "Custom positive voxels:"
    )
    print(
        custom_positive
    )

    print()
    print(
        "MONAI positive voxels:"
    )
    print(
        monai_positive
    )

    print()
    print(
        "Mismatched mask voxels:"
    )
    print(
        mismatched_mask_voxels
    )

    print()
    print(
        "Mask voxel agreement:"
    )
    print(
        mask_agreement
    )

    print()
    print(
        "Dice custom vs MONAI:"
    )
    print(
        mask_dice
    )

    # ========================================================
    # BASIC ASSERTIONS
    # ========================================================

    assert (
        custom_probability.min()
        >= -1e-6
    )

    assert (
        custom_probability.max()
        <= 1.0 + 1e-6
    )

    assert (
        monai_probability.min()
        >= -1e-6
    )

    assert (
        monai_probability.max()
        <= 1.0 + 1e-6
    )

    # --------------------------------------------------------
    # We do NOT yet enforce an extremely strict MAE threshold.
    #
    # First we want to observe the true numerical difference
    # between the custom and MONAI implementations.
    # --------------------------------------------------------

    assert (
        mask_agreement
        >= 0.99
    )

    print()
    print("=" * 70)

    print(
        "Custom vs MONAI "
        "sliding-window comparison completed."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()