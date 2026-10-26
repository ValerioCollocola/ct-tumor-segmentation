import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
)


WINDOW_MIN = -25.0
WINDOW_MAX = 125.0

TARGET_SPACING = (1.0, 1.0, 2.5)


def build_monai_preprocessing(
    target_spacing=TARGET_SPACING,
):
    """
    MONAI preprocessing pipeline designed to reproduce
    the existing custom preprocessing as closely as possible.

    Custom pipeline:
        1. Load CT and label
        2. CT windowing [-25, 125] HU
        3. Normalize CT to [0, 1]
        4. Resample CT with linear interpolation
        5. Resample label with nearest-neighbor interpolation

    Returns
    -------
    monai.transforms.Compose
        MONAI preprocessing pipeline.
    """

    return Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                image_only=False,
            ),

            EnsureChannelFirstd(
                keys=["image", "label"],
            ),

            # Equivalent to:
            #
            # np.clip(image, -25, 125)
            # followed by
            # (image - (-25)) / (125 - (-25))
            #
            ScaleIntensityRanged(
                keys=["image"],
                a_min=WINDOW_MIN,
                a_max=WINDOW_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            # Resample image and label together
            # while using the appropriate interpolation.
            #
            # image -> linear interpolation
            # label -> nearest neighbor
            #
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
                align_corners=True,
                scale_extent=True,
            ),

            EnsureTyped(
                keys=["image"],
                dtype=torch.float32,
                track_meta=True,
            ),

            EnsureTyped(
                keys=["label"],
                dtype=torch.int64,
                track_meta=True,
            ),
        ]
    )


def preprocess_case_monai(
    image_path,
    label_path,
    target_spacing=TARGET_SPACING,
):
    """
    Apply MONAI preprocessing to one CT/label pair.
    """

    transforms = build_monai_preprocessing(
        target_spacing=target_spacing,
    )

    data = {
        "image": str(image_path),
        "label": str(label_path),
    }

    return transforms(data)