from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import torch

from src.dicom.io import (
    load_dicom_series
)

from src.dicom.series import (
    sort_dicom_slices
)

from src.dicom.reconstruction import (
    reconstruct_ct_volume
)

from src.dicom.seg import (
    reconstruct_segmentation_volume
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

ct_directory = (
    "data/dicom/liver_arterial"
)

seg_directory = Path(
    "data/dicom/liver_seg"
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
# 1. Load arterial CT
# --------------------------------------------------

ct_datasets = load_dicom_series(
    ct_directory
)

ct_datasets = [
    dataset
    for dataset in ct_datasets
    if getattr(
        dataset,
        "Modality",
        None
    ) == "CT"
]

sorted_ct_datasets = sort_dicom_slices(
    ct_datasets
)


volume_hu, original_spacing, _ = (
    reconstruct_ct_volume(
        sorted_ct_datasets
    )
)


# --------------------------------------------------
# 2. Load DICOM SEG
# --------------------------------------------------

seg_dataset = None

for file_path in seg_directory.rglob("*"):

    if not file_path.is_file():
        continue

    try:
        dataset = pydicom.dcmread(
            file_path
        )
    except Exception:
        continue

    if getattr(
        dataset,
        "Modality",
        None
    ) == "SEG":
        seg_dataset = dataset
        break


if seg_dataset is None:
    raise RuntimeError(
        "No DICOM SEG object found."
    )


liver_mask, _ = (
    reconstruct_segmentation_volume(
        seg_dataset=seg_dataset,
        sorted_ct_datasets=sorted_ct_datasets,
        segment_number=1
    )
)


# --------------------------------------------------
# 3. Preprocess CT
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


resampled_ct, _ = (
    resample_volume(
        normalized,
        current_spacing=original_spacing,
        target_spacing=target_spacing,
        is_label=False
    )
)


resampled_liver_mask, _ = (
    resample_volume(
        liver_mask,
        current_spacing=original_spacing,
        target_spacing=target_spacing,
        is_label=True
    )
)


resampled_liver_mask = (
    resampled_liver_mask > 0
).astype(
    np.uint8
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


model, optimizer, epoch, loss = (
    load_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        device=device
    )
)


# --------------------------------------------------
# 5. Full-volume inference
# --------------------------------------------------

tumor_probability, raw_prediction = (
    sliding_window_inference(
        model=model,
        volume=resampled_ct,
        patch_size=patch_size,
        overlap=overlap,
        device=device
    )
)


# --------------------------------------------------
# 6. Constrain to liver
# --------------------------------------------------

constrained_probability = (
    tumor_probability
    * resampled_liver_mask
)

constrained_prediction = (
    raw_prediction
    * resampled_liver_mask
).astype(
    np.uint8
)


# --------------------------------------------------
# 7. Find slice with most constrained prediction
# --------------------------------------------------

voxels_per_slice = np.count_nonzero(
    constrained_prediction,
    axis=(0, 1)
)


best_slice = int(
    np.argmax(
        voxels_per_slice
    )
)


print()
print("Best constrained slice:")
print(best_slice)

print()
print("Constrained tumor voxels in slice:")
print(
    voxels_per_slice[
        best_slice
    ]
)

print()
print("Maximum probability in liver:")
print(
    constrained_probability[
        :,
        :,
        best_slice
    ].max()
)


# --------------------------------------------------
# 8. Extract slice
# --------------------------------------------------

ct_slice = resampled_ct[
    :,
    :,
    best_slice
]

liver_slice = resampled_liver_mask[
    :,
    :,
    best_slice
]

raw_slice = raw_prediction[
    :,
    :,
    best_slice
]

constrained_slice = constrained_prediction[
    :,
    :,
    best_slice
]

probability_slice = constrained_probability[
    :,
    :,
    best_slice
]


# --------------------------------------------------
# 9. Plot
# --------------------------------------------------

figure, axes = plt.subplots(
    1,
    4,
    figsize=(20, 5)
)


axes[0].imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower"
)

axes[0].set_title(
    f"Arterial CT - slice {best_slice}"
)


axes[1].imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower"
)

axes[1].imshow(
    liver_slice.T,
    origin="lower",
    alpha=0.30,
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "Liver ROI"
)


axes[2].imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower"
)

axes[2].imshow(
    raw_slice.T,
    origin="lower",
    alpha=0.50,
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "Raw tumor prediction"
)


axes[3].imshow(
    ct_slice.T,
    cmap="gray",
    origin="lower"
)

axes[3].imshow(
    constrained_slice.T,
    origin="lower",
    alpha=0.50,
    vmin=0,
    vmax=1
)

axes[3].set_title(
    "Liver-constrained prediction"
)


for axis in axes:
    axis.set_axis_off()


plt.tight_layout()
plt.show()