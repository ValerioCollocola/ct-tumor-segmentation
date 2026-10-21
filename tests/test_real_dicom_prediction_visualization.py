import numpy as np
import matplotlib.pyplot as plt
import torch

from src.dicom.io import (
    load_dicom_series,
    group_by_series
)

from src.dicom.reconstruction import (
    reconstruct_ct_volume
)

from src.preprocessing.windowing import (
    apply_ct_window
)

from src.preprocessing.normalization import (
    normalize_intensity
)

from src.preprocessing.resampling import (
    resample_volume
)

from src.models.unet3d import (
    UNet3D
)

from src.training.checkpoint import (
    load_checkpoint
)

from src.inference.sliding_window import (
    sliding_window_inference
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

dicom_directory = (
    "data/dicom/liver_portal_venous"
)

checkpoint_path = (
    "outputs/checkpoints/"
    "best_real_binary_unet_best_tumor_dice.pt"
)

target_spacing = (
    1.0,
    1.0,
    2.5
)

patch_size = (
    96,
    96,
    64
)

overlap = 0.5

device = "cpu"


# --------------------------------------------------
# 1. Load DICOM
# --------------------------------------------------

datasets = load_dicom_series(
    dicom_directory
)

series_dict = group_by_series(
    datasets
)


ct_series = None

for _, series_datasets in series_dict.items():

    modality = getattr(
        series_datasets[0],
        "Modality",
        None
    )

    if modality == "CT":
        ct_series = series_datasets
        break


if ct_series is None:
    raise RuntimeError(
        "No CT series found."
    )


# --------------------------------------------------
# 2. Reconstruct
# --------------------------------------------------

volume_hu, original_spacing, _ = (
    reconstruct_ct_volume(
        ct_series
    )
)


# --------------------------------------------------
# 3. Preprocess
# --------------------------------------------------

window_center = 50
window_width = 150

window_min = (
    window_center
    - window_width / 2
)

window_max = (
    window_center
    + window_width / 2
)


windowed = apply_ct_window(
    volume_hu,
    center=window_center,
    width=window_width
)


normalized = normalize_intensity(
    windowed,
    min_value=window_min,
    max_value=window_max
)


resampled, _ = resample_volume(
    normalized,
    current_spacing=original_spacing,
    target_spacing=target_spacing,
    is_label=False
)


# --------------------------------------------------
# 4. Load model
# --------------------------------------------------

model = UNet3D(
    in_channels=1,
    num_classes=2,
    base_channels=8
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


model, optimizer, epoch, loss = load_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_path=checkpoint_path,
    device=device
)


# --------------------------------------------------
# 5. Full-volume inference
# --------------------------------------------------

tumor_probability, prediction_mask = (
    sliding_window_inference(
        model=model,
        volume=resampled,
        patch_size=patch_size,
        overlap=overlap,
        device=device
    )
)


# --------------------------------------------------
# 6. Find most suspicious slice
# --------------------------------------------------

slice_probability_sum = np.sum(
    tumor_probability,
    axis=(0, 1)
)


best_slice = int(
    np.argmax(
        slice_probability_sum
    )
)


print()
print("Slice with highest tumor probability:")
print(best_slice)


print()
print("Maximum probability in slice:")
print(
    tumor_probability[
        :,
        :,
        best_slice
    ].max()
)


print()
print("Predicted tumor voxels in slice:")
print(
    np.count_nonzero(
        prediction_mask[
            :,
            :,
            best_slice
        ]
    )
)


# --------------------------------------------------
# 7. Extract images
# --------------------------------------------------

ct_slice = resampled[
    :,
    :,
    best_slice
]

probability_slice = tumor_probability[
    :,
    :,
    best_slice
]

prediction_slice = prediction_mask[
    :,
    :,
    best_slice
]


# --------------------------------------------------
# 8. Plot
# --------------------------------------------------

figure, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


axes[0].imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower"
)

axes[0].set_title(
    f"CT - Slice {best_slice}"
)


axes[1].imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower"
)

probability_overlay = axes[1].imshow(
    probability_slice.T,
    origin="lower",
    alpha=0.5,
    vmin=0.0,
    vmax=1.0
)

axes[1].set_title(
    "Tumor probability"
)


axes[2].imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower"
)

axes[2].imshow(
    prediction_slice.T,
    origin="lower",
    alpha=0.5,
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "Prediction @ threshold 0.5"
)


for axis in axes:
    axis.set_axis_off()


figure.colorbar(
    probability_overlay,
    ax=axes[1],
    fraction=0.046,
    pad=0.04
)


plt.tight_layout()
plt.show()