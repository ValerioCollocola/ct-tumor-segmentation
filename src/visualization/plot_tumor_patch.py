import matplotlib.pyplot as plt
import numpy as np


def plot_tumor_patch(
    image_patch,
    label_patch,
    output_path
):
    tumor_pixels_per_slice = np.sum(
        label_patch == 2,
        axis=(0, 1)
    )

    slice_index = np.argmax(
        tumor_pixels_per_slice
    )

    image_slice = image_patch[:, :, slice_index]
    label_slice = label_patch[:, :, slice_index]

    tumor_mask = label_slice == 2

    tumor_overlay = np.ma.masked_where(
        ~tumor_mask,
        tumor_mask
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    fig.suptitle(
        f"3D Tumor Patch Visualization - Slice {slice_index}",
        fontsize=16
    )

    axes[0].imshow(
        image_slice,
        cmap="gray"
    )
    axes[0].set_title("CT Patch")
    axes[0].axis("off")

    axes[1].imshow(
        tumor_mask,
        cmap="Reds"
    )
    axes[1].set_title("Tumor Mask")
    axes[1].axis("off")

    axes[2].imshow(
        image_slice,
        cmap="gray"
    )
    axes[2].imshow(
        tumor_overlay,
        cmap="Reds",
        alpha=0.5
    )
    axes[2].set_title("CT + Tumor Overlay")
    axes[2].axis("off")

    plt.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()