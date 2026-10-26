from pathlib import Path

import nibabel as nib
import numpy as np

from src.preprocessing.monai_transforms import (
    preprocess_case_monai,
)
from src.preprocessing.normalization import normalize_intensity
from src.preprocessing.resampling import resample_volume


IMAGE_PATH = Path(
    "data/multicase/images/liver_0.nii.gz"
)

LABEL_PATH = Path(
    "data/multicase/labels/liver_0.nii.gz"
)

TARGET_SPACING = (1.0, 1.0, 2.5)


def main():

    print("=" * 70)
    print("MONAI PREPROCESSING COMPARISON")
    print("=" * 70)

    # --------------------------------------------------
    # CUSTOM PIPELINE
    # --------------------------------------------------

    image_nii = nib.load(IMAGE_PATH)
    label_nii = nib.load(LABEL_PATH)

    image = image_nii.get_fdata().astype(np.float32)
    label = label_nii.get_fdata().astype(np.int64)

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

    # Same liver window used in the project.
    windowed = np.clip(
        image,
        -25.0,
        125.0,
    )

    normalized = normalize_intensity(
        windowed,
        -25.0,
        125.0,
    )

    custom_image, _ = resample_volume(
        normalized,
        current_spacing=current_spacing,
        target_spacing=TARGET_SPACING,
        is_label=False,
    )

    custom_label, _ = resample_volume(
        label,
        current_spacing=current_spacing,
        target_spacing=TARGET_SPACING,
        is_label=True,
    )

    # --------------------------------------------------
    # MONAI PIPELINE
    # --------------------------------------------------

    monai_data = preprocess_case_monai(
        image_path=IMAGE_PATH,
        label_path=LABEL_PATH,
        target_spacing=TARGET_SPACING,
    )

    monai_image = (
        monai_data["image"]
        .detach()
        .cpu()
        .numpy()
    )

    monai_label = (
        monai_data["label"]
        .detach()
        .cpu()
        .numpy()
    )

    # Remove channel dimension for comparison.
    monai_image = monai_image[0]
    monai_label = monai_label[0]

    # --------------------------------------------------
    # COMPARISON
    # --------------------------------------------------

    print()
    print("-" * 70)

    print("Custom image shape:")
    print(custom_image.shape)

    print("MONAI image shape:")
    print(monai_image.shape)

    print()

    print("Custom image range:")
    print(
        float(custom_image.min()),
        float(custom_image.max()),
    )

    print("MONAI image range:")
    print(
        float(monai_image.min()),
        float(monai_image.max()),
    )

    print()

    print("Custom label values:")
    print(np.unique(custom_label))

    print("MONAI label values:")
    print(np.unique(monai_label))

    # --------------------------------------------------
    # BASIC ASSERTIONS
    # --------------------------------------------------

    assert monai_image.min() >= -1e-6
    assert monai_image.max() <= 1.0 + 1e-6

    assert set(
        np.unique(monai_label).tolist()
    ).issubset({0, 1, 2})

    assert monai_image.shape == monai_label.shape

    print()
    print("MONAI preprocessing basic checks: OK")

    # --------------------------------------------------
    # STRICTER COMPARISON
    # --------------------------------------------------

    if custom_image.shape == monai_image.shape:

        image_mae = np.mean(
            np.abs(
                custom_image.astype(np.float32)
                - monai_image.astype(np.float32)
            )
        )

        label_agreement = np.mean(
            custom_label == monai_label
        )

        print()
        print("Mean absolute image difference:")
        print(float(image_mae))

        print()

        print("Label voxel agreement:")
        print(float(label_agreement))

    else:

        print()
        print(
            "NOTE: custom and MONAI shapes differ slightly."
        )

        print(
            "This can occur because MONAI performs "
            "affine-aware physical-space resampling, "
            "while the custom implementation uses "
            "scipy.ndimage.zoom."
        )


if __name__ == "__main__":
    main()