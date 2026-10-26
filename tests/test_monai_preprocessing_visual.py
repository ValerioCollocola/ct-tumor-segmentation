from pathlib import Path

import matplotlib.pyplot as plt
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

OUTPUT_PATH = Path(
    "docs/images/monai_custom_preprocessing_comparison.png"
)

TARGET_SPACING = (1.0, 1.0, 2.5)


def main():

    print("=" * 70)
    print("CUSTOM VS MONAI PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------
    # Load original NIfTI
    # --------------------------------------------------

    image_nii = nib.load(IMAGE_PATH)
    label_nii = nib.load(LABEL_PATH)

    image = image_nii.get_fdata().astype(np.float32)
    label = label_nii.get_fdata().astype(np.int64)

    current_spacing = np.asarray(
        image_nii.header.get_zooms()[:3],
        dtype=float,
    )

    # --------------------------------------------------
    # CUSTOM PIPELINE
    # --------------------------------------------------

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

    custom_label = custom_label.astype(np.int64)

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
        .numpy()[0]
    )

    monai_label = (
        monai_data["label"]
        .detach()
        .cpu()
        .numpy()[0]
        .astype(np.int64)
    )

    # --------------------------------------------------
    # Quantitative comparison
    # --------------------------------------------------

    assert custom_image.shape == monai_image.shape
    assert custom_label.shape == monai_label.shape

    image_difference = np.abs(
        custom_image.astype(np.float32)
        - monai_image.astype(np.float32)
    )

    label_difference = (
        custom_label != monai_label
    )

    image_mae = float(
        image_difference.mean()
    )

    image_max_difference = float(
        image_difference.max()
    )

    mismatched_label_voxels = int(
        label_difference.sum()
    )

    total_voxels = int(
        label_difference.size
    )

    label_agreement = float(
        np.mean(
            custom_label == monai_label
        )
    )

    mismatch_percentage = (
        mismatched_label_voxels
        / total_voxels
        * 100.0
    )

    print()
    print("Image MAE:")
    print(image_mae)

    print()
    print("Maximum image difference:")
    print(image_max_difference)

    print()
    print("Label agreement:")
    print(label_agreement)

    print()
    print("Mismatched label voxels:")
    print(mismatched_label_voxels)

    print()
    print("Label mismatch percentage:")
    print(mismatch_percentage)

    # --------------------------------------------------
    # Find a useful tumor slice automatically
    # --------------------------------------------------

    tumor_per_slice = np.sum(
        custom_label == 2,
        axis=(0, 1),
    )

    slice_index = int(
        np.argmax(tumor_per_slice)
    )

    print()
    print("Visualization slice:")
    print(slice_index)

    print()
    print("Tumor voxels on slice:")
    print(
        int(
            tumor_per_slice[slice_index]
        )
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    # ---------------- Image row ----------------

    axes[0, 0].imshow(
        custom_image[:, :, slice_index].T,
        cmap="gray",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )

    axes[0, 0].set_title(
        "Custom preprocessing"
    )

    axes[0, 1].imshow(
        monai_image[:, :, slice_index].T,
        cmap="gray",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )

    axes[0, 1].set_title(
        "MONAI preprocessing"
    )

    diff_display = axes[0, 2].imshow(
        image_difference[:, :, slice_index].T,
        origin="lower",
    )

    axes[0, 2].set_title(
        "Absolute intensity difference"
    )

    fig.colorbar(
        diff_display,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.04,
    )

    # ---------------- Label row ----------------

    axes[1, 0].imshow(
        custom_label[:, :, slice_index].T,
        origin="lower",
        vmin=0,
        vmax=2,
    )

    axes[1, 0].set_title(
        "Custom label"
    )

    axes[1, 1].imshow(
        monai_label[:, :, slice_index].T,
        origin="lower",
        vmin=0,
        vmax=2,
    )

    axes[1, 1].set_title(
        "MONAI label"
    )

    axes[1, 2].imshow(
        label_difference[:, :, slice_index].T,
        origin="lower",
        vmin=0,
        vmax=1,
    )

    axes[1, 2].set_title(
        "Label disagreement"
    )

    # --------------------------------------------------
    # Cosmetic cleanup
    # --------------------------------------------------

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        (
            "Custom vs MONAI preprocessing\n"
            f"Slice {slice_index} | "
            f"Image MAE = {image_mae:.6f} | "
            f"Label agreement = "
            f"{label_agreement * 100:.4f}%"
        ),
        fontsize=14,
    )

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print("Saved figure:")
    print(OUTPUT_PATH)

    print()
    print(
        "Custom vs MONAI visual comparison passed."
    )


if __name__ == "__main__":
    main()