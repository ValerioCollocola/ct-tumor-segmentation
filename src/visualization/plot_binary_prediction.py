import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_binary_segmentation_prediction(
    image,
    label,
    outputs,
    output_path
):
    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

    prediction = (
        predictions
        .squeeze(0)
        .cpu()
        .numpy()
    )

    label = (
        label
        .squeeze(0)
        .cpu()
        .numpy()
    )

    image = (
        image
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # Trova la slice con più tumore nella ground truth
    tumor_per_slice = np.sum(
        label == 1,
        axis=(0, 1)
    )

    slice_index = np.argmax(
        tumor_per_slice
    )

    image_slice = image[
        :,
        :,
        slice_index
    ]

    ground_truth = (
        label[
            :,
            :,
            slice_index
        ]
        == 1
    )

    predicted_tumor = (
        prediction[
            :,
            :,
            slice_index
        ]
        == 1
    )

    ground_truth_overlay = np.ma.masked_where(
        ~ground_truth,
        ground_truth
    )

    prediction_overlay = np.ma.masked_where(
        ~predicted_tumor,
        predicted_tumor
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    fig.suptitle(
        f"Binary Tumor Segmentation - Slice {slice_index}",
        fontsize=16
    )

    # CT
    axes[0].imshow(
        image_slice,
        cmap="gray"
    )

    axes[0].set_title(
        "CT"
    )

    axes[0].axis(
        "off"
    )

    # Ground Truth
    axes[1].imshow(
        ground_truth,
        cmap="Reds"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].axis(
        "off"
    )

    # Prediction
    axes[2].imshow(
        predicted_tumor,
        cmap="Blues"
    )

    axes[2].set_title(
        "Prediction"
    )

    axes[2].axis(
        "off"
    )

    # Overlay
    axes[3].imshow(
        image_slice,
        cmap="gray"
    )

    axes[3].imshow(
        ground_truth_overlay,
        cmap="Reds",
        alpha=0.5
    )

    axes[3].imshow(
        prediction_overlay,
        cmap="Blues",
        alpha=0.5
    )

    axes[3].set_title(
        "GT (red) + Prediction (blue)"
    )

    axes[3].axis(
        "off"
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93
        ]
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()