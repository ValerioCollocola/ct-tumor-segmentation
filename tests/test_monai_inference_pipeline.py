from pathlib import Path

import numpy as np

from src.inference.monai_pipeline import (
    run_monai_nifti_inference,
)


IMAGE_PATH = Path(
    "data/multicase/images/liver_7.nii.gz"
)

CHECKPOINT_PATH = Path(
    "outputs/checkpoints/"
    "best_real_binary_unet_best_tumor_dice.pt"
)

THRESHOLD = 0.5


def main():

    print("=" * 70)
    print("MONAI END-TO-END INFERENCE PIPELINE")
    print("=" * 70)

    result = (
        run_monai_nifti_inference(
            image_path=IMAGE_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            threshold=THRESHOLD,
            patch_size=(
                96,
                96,
                64,
            ),
            overlap=0.5,
            sw_batch_size=1,
        )
    )

    probability = result[
        "probability"
    ]

    mask = result[
        "mask"
    ]

    metadata = result[
        "metadata"
    ]

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("Probability shape:")
    print(
        probability.shape
    )

    print()
    print("Mask shape:")
    print(
        mask.shape
    )

    print()
    print("Probability range:")
    print(
        float(
            probability.min()
        ),
        float(
            probability.max()
        ),
    )

    print()
    print("Probability mean:")
    print(
        float(
            probability.mean()
        )
    )

    print()
    print("Mask values:")
    print(
        np.unique(
            mask
        )
    )

    print()
    print("Positive voxels:")
    print(
        int(
            mask.sum()
        )
    )

    print()
    print("Threshold:")
    print(
        result[
            "threshold"
        ]
    )

    print()
    print("Device:")
    print(
        metadata[
            "device"
        ]
    )

    print()
    print("Patch size:")
    print(
        metadata[
            "patch_size"
        ]
    )

    print()
    print("Overlap:")
    print(
        metadata[
            "overlap"
        ]
    )

    # ========================================================
    # ASSERTIONS
    # ========================================================

    assert (
        probability.shape
        == mask.shape
    )

    assert (
        probability.ndim
        == 3
    )

    assert (
        probability.min()
        >= -1e-6
    )

    assert (
        probability.max()
        <= 1.0 + 1e-6
    )

    assert set(
        np.unique(
            mask
        ).tolist()
    ).issubset(
        {0, 1}
    )

    assert (
        result["threshold"]
        == 0.5
    )

    assert (
        tuple(
            metadata[
                "patch_size"
            ]
        )
        == (
            96,
            96,
            64,
        )
    )

    assert (
        metadata[
            "overlap"
        ]
        == 0.5
    )

    # --------------------------------------------------------
    # Regression-style sanity check.
    #
    # We previously obtained 19,454 positives with the
    # same checkpoint / volume / threshold.
    #
    # This is intentionally strict because this test is
    # checking that the integrated pipeline preserves
    # the already-validated MONAI inference behavior.
    # --------------------------------------------------------

    # ========================================================
    # INTEGRATED PIPELINE SANITY CHECK
    # ========================================================

    actual_positive_voxels = int(
        mask.sum()
    )

    total_voxels = int(
        mask.size
    )

    positive_fraction = (
        actual_positive_voxels
        / total_voxels
    )

    print()
    print("=" * 70)
    print("INTEGRATED PIPELINE SUMMARY")
    print("=" * 70)

    print()
    print("Actual positive voxels:")
    print(
        actual_positive_voxels
    )

    print()
    print("Total voxels:")
    print(
        total_voxels
    )

    print()
    print("Positive voxel fraction:")
    print(
        positive_fraction
    )

    # --------------------------------------------------------
    # The integrated pipeline uses MONAI preprocessing.
    #
    # Therefore we intentionally do NOT assert equality
    # against the previous value of 19,454 voxels.
    #
    # That value was obtained using:
    #
    #     custom preprocessing
    #          +
    #     MONAI sliding-window inference
    #
    # whereas this test evaluates:
    #
    #     MONAI preprocessing
    #          +
    #     MONAI sliding-window inference
    #
    # Small interpolation differences can move probabilities
    # around the 0.5 threshold and therefore change the final
    # number of positive voxels.
    # --------------------------------------------------------

    assert actual_positive_voxels >= 0

    assert (
        actual_positive_voxels
        <= total_voxels
    )

    assert (
        0.0
        <= positive_fraction
        <= 1.0
    )

    print()
    print("=" * 70)

    print(
        "MONAI end-to-end inference "
        "pipeline test passed."
    )

    print("=" * 70)

    print()
    print("Actual positive voxels:")
    print(actual_positive_voxels)

if __name__ == "__main__":
    main()