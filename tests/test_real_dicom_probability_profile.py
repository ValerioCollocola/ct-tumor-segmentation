import numpy as np
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

from src.models.unet3d import UNet3D

from src.training.checkpoint import (
    load_checkpoint
)

from src.inference.sliding_window import (
    sliding_window_inference
)


dicom_directory = (
    "data/dicom/liver_portal_venous"
)

checkpoint_path = (
    "outputs/checkpoints/"
    "best_real_binary_unet_best_tumor_dice.pt"
)

device = "cpu"


# --------------------------------------------------
# Load DICOM
# --------------------------------------------------

datasets = load_dicom_series(
    dicom_directory
)

series_dict = group_by_series(
    datasets
)

ct_series = None

for _, series_datasets in series_dict.items():

    if getattr(
        series_datasets[0],
        "Modality",
        None
    ) == "CT":

        ct_series = series_datasets
        break


if ct_series is None:
    raise RuntimeError(
        "No CT series found."
    )


# --------------------------------------------------
# Reconstruct + preprocess
# --------------------------------------------------

volume_hu, original_spacing, _ = (
    reconstruct_ct_volume(
        ct_series
    )
)

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
    target_spacing=(1.0, 1.0, 2.5),
    is_label=False
)


# --------------------------------------------------
# Load model
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

model, optimizer, epoch, loss = (
    load_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        device=device
    )
)


# --------------------------------------------------
# Sliding-window inference
# --------------------------------------------------

tumor_probability, prediction_mask = (
    sliding_window_inference(
        model=model,
        volume=resampled,
        patch_size=(96, 96, 64),
        overlap=0.5,
        device=device
    )
)


# --------------------------------------------------
# Per-slice diagnostics
# --------------------------------------------------

slice_sum = np.sum(
    tumor_probability,
    axis=(0, 1)
)

slice_max = np.max(
    tumor_probability,
    axis=(0, 1)
)

slice_predicted_voxels = np.count_nonzero(
    prediction_mask,
    axis=(0, 1)
)


# --------------------------------------------------
# Top slices by probability sum
# --------------------------------------------------

top_indices = np.argsort(
    slice_sum
)[::-1][:10]


print()
print("TOP 10 SLICES BY TUMOR-PROBABILITY SUM")
print("=" * 70)

for rank, slice_index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank:2d}. "
        f"slice={slice_index:3d} | "
        f"sum={slice_sum[slice_index]:.4f} | "
        f"max={slice_max[slice_index]:.4f} | "
        f"pred_voxels={slice_predicted_voxels[slice_index]}"
    )


# --------------------------------------------------
# Border vs central region
# --------------------------------------------------

border_size = 10

border_indices = list(
    range(border_size)
) + list(
    range(
        tumor_probability.shape[2] - border_size,
        tumor_probability.shape[2]
    )
)

central_indices = list(
    range(
        border_size,
        tumor_probability.shape[2] - border_size
    )
)


border_mean = np.mean(
    slice_sum[
        border_indices
    ]
)

central_mean = np.mean(
    slice_sum[
        central_indices
    ]
)


print()
print("Border mean probability sum:")
print(border_mean)

print()
print("Central mean probability sum:")
print(central_mean)