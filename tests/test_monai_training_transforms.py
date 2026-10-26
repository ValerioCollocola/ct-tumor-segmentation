import numpy as np

from src.preprocessing.monai_training_transforms import (
    build_monai_training_transforms,
    build_monai_validation_transforms,
)


CASE = {
    "image": "data/multicase/images/liver_0.nii.gz",
    "label": "data/multicase/labels/liver_0.nii.gz",
    "case_id": "liver_0",
}


PATCH_SIZE = (96, 96, 64)


def inspect_patch(
    sample,
    index,
):
    image = (
        sample["image"]
        .detach()
        .cpu()
        .numpy()
    )

    label = (
        sample["label"]
        .detach()
        .cpu()
        .numpy()
    )

    unique_values = np.unique(label)

    tumor_voxels = int(
        np.sum(label == 2)
    )

    liver_voxels = int(
        np.sum(label == 1)
    )

    background_voxels = int(
        np.sum(label == 0)
    )

    total_voxels = int(
        label.size
    )

    tumor_fraction = (
        tumor_voxels
        / total_voxels
    )

    print()
    print(
        f"PATCH {index}"
    )

    print("-" * 60)

    print("image shape:")
    print(image.shape)

    print()

    print("label shape:")
    print(label.shape)

    print()

    print("image range:")
    print(
        float(image.min()),
        float(image.max()),
    )

    print()

    print("label values:")
    print(unique_values)

    print()

    print("background voxels:")
    print(background_voxels)

    print()

    print("liver voxels:")
    print(liver_voxels)

    print()

    print("tumor voxels:")
    print(tumor_voxels)

    print()

    print("tumor fraction:")
    print(tumor_fraction)

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert image.shape == (
        1,
        *PATCH_SIZE,
    )

    assert label.shape == (
        1,
        *PATCH_SIZE,
    )

    assert image.min() >= -1e-6
    assert image.max() <= 1.0 + 1e-6

    assert set(
        unique_values.tolist()
    ).issubset(
        {0, 1, 2}
    )

    # We explicitly requested tumor-centered sampling.
    assert tumor_voxels > 0

    return tumor_fraction


def main():

    print("=" * 70)
    print("MONAI TRAINING TRANSFORMS TEST")
    print("=" * 70)

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    train_transforms = (
    build_monai_training_transforms()
    )

    train_samples = [
         train_transforms(CASE)
         for _ in range(5)
    ]  

    print()
    print("Number of training patches:")
    print(len(train_samples))

    assert len(train_samples) == 5

    tumor_fractions = []

    for index, sample in enumerate(
        train_samples,
        start=1,
    ):

        tumor_fraction = (
            inspect_patch(
                sample,
                index,
            )
        )

        tumor_fractions.append(
            tumor_fraction
        )

    print()
    print("=" * 70)

    print(
        "Mean tumor fraction "
        "across training patches:"
    )

    print(
        float(
            np.mean(
                tumor_fractions
            )
        )
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION TRANSFORM")
    print("=" * 70)

    val_transforms = (
        build_monai_validation_transforms()
    )

    val_sample = val_transforms(
    CASE
   )

    val_samples = [
    val_sample
 ]

    print()
    print("Number of validation patches:")
    print(len(val_samples))

    assert len(val_samples) == 1

    inspect_patch(
        val_samples[0],
        index=1,
    )

    print()
    print(
        "MONAI training / validation "
        "transforms test passed."
    )

    print()
    print("tumor center:")
    print(sample["tumor_center"])

    print()
    print("crop center:")
    print(sample["crop_center"])

    print()
    print("crop shift:")
    print(sample["crop_shift"])

if __name__ == "__main__":
    main()