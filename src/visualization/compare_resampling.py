import matplotlib.pyplot as plt
import numpy as np

from src.data.io import load_nifti
from src.preprocessing.resampling import resample_volume


def compare_resampling(
    image_path,
    label_path,
    target_spacing,
    output_path
):
    image, image_affine = load_nifti(image_path)
    label, label_affine = load_nifti(label_path)

    current_spacing = np.linalg.norm(
        image_affine[:3, :3],
        axis=0
    )

    resampled_image, _ = resample_volume(
        image,
        current_spacing=current_spacing,
        target_spacing=target_spacing,
        is_label=False
    )

    resampled_label, _ = resample_volume(
        label,
        current_spacing=current_spacing,
        target_spacing=target_spacing,
        is_label=True
    )

    original_tumor_per_slice = np.sum(
        label == 2,
        axis=(0, 1)
    )

    original_slice_index = np.argmax(
        original_tumor_per_slice
    )

    resampled_tumor_per_slice = np.sum(
        resampled_label == 2,
        axis=(0, 1)
    )

    resampled_slice_index = np.argmax(
        resampled_tumor_per_slice
    )

    original_image_slice = image[:, :, original_slice_index]
    original_label_slice = label[:, :, original_slice_index]

    resampled_image_slice = resampled_image[:, :, resampled_slice_index]
    resampled_label_slice = resampled_label[:, :, resampled_slice_index]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    fig.suptitle(
        "Voxel-Spacing Resampling Comparison",
        fontsize=16
    )

    axes[0].imshow(
        original_image_slice,
        cmap="gray"
    )
    axes[0].set_title(
        f"Original CT\nSlice {original_slice_index}"
    )
    axes[0].axis("off")

    axes[1].imshow(
        original_label_slice == 2,
        cmap="Reds"
    )
    axes[1].set_title("Original Tumor Mask")
    axes[1].axis("off")

    axes[2].imshow(
        resampled_image_slice,
        cmap="gray"
    )
    axes[2].set_title(
        f"Resampled CT\nSlice {resampled_slice_index}"
    )
    axes[2].axis("off")

    axes[3].imshow(
        resampled_label_slice == 2,
        cmap="Reds"
    )
    axes[3].set_title("Resampled Tumor Mask")
    axes[3].axis("off")

    plt.tight_layout(
        rect=[0, 0, 1, 0.92]
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()