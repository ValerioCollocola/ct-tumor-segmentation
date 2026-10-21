from pathlib import Path

import numpy as np
import pydicom
import torch

from scipy.ndimage import (
    label,
    center_of_mass
)

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
# 5. Sliding-window inference
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
# 6. Apply liver constraint
# --------------------------------------------------

constrained_prediction = (
    raw_prediction
    * resampled_liver_mask
).astype(
    np.uint8
)


# --------------------------------------------------
# 7. Connected-component analysis
# --------------------------------------------------

connectivity = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)

components, num_components = label(
    constrained_prediction > 0,
    structure=connectivity
)


print()
print("CONSTRAINED COMPONENT ANALYSIS")
print("=" * 80)

print()
print("Number of connected components:")
print(num_components)


# --------------------------------------------------
# 8. Physical voxel volume
# --------------------------------------------------

voxel_volume_mm3 = (
    target_spacing[0]
    * target_spacing[1]
    * target_spacing[2]
)


print()
print("Voxel volume:")
print(
    voxel_volume_mm3,
    "mm^3"
)


# --------------------------------------------------
# 9. Component statistics
# --------------------------------------------------

component_rows = []


for component_id in range(
    1,
    num_components + 1
):

    component_mask = (
        components == component_id
    )

    voxel_count = np.count_nonzero(
        component_mask
    )

    physical_volume_mm3 = (
        voxel_count
        * voxel_volume_mm3
    )

    physical_volume_ml = (
        physical_volume_mm3
        / 1000.0
    )

    centroid = center_of_mass(
        component_mask
    )

    z_indices = np.where(
        component_mask
    )[2]

    first_slice = int(
        z_indices.min()
    )

    last_slice = int(
        z_indices.max()
    )

    maximum_probability = float(
        tumor_probability[
            component_mask
        ].max()
    )

    mean_probability = float(
        tumor_probability[
            component_mask
        ].mean()
    )


    component_rows.append(
        {
            "component_id": component_id,
            "voxel_count": voxel_count,
            "physical_volume_mm3": physical_volume_mm3,
            "physical_volume_ml": physical_volume_ml,
            "centroid": centroid,
            "first_slice": first_slice,
            "last_slice": last_slice,
            "maximum_probability": maximum_probability,
            "mean_probability": mean_probability
        }
    )


# --------------------------------------------------
# 10. Sort by component size
# --------------------------------------------------

component_rows = sorted(
    component_rows,
    key=lambda row: row[
        "voxel_count"
    ],
    reverse=True
)


# --------------------------------------------------
# 11. Print results
# --------------------------------------------------

for rank, row in enumerate(
    component_rows,
    start=1
):

    print()
    print(
        f"Component rank {rank}"
    )

    print(
        f"Component ID: "
        f"{row['component_id']}"
    )

    print(
        f"Voxel count: "
        f"{row['voxel_count']}"
    )

    print(
        f"Physical volume: "
        f"{row['physical_volume_mm3']:.2f} mm^3"
    )

    print(
        f"Physical volume: "
        f"{row['physical_volume_ml']:.4f} mL"
    )

    print(
        "Centroid:"
    )

    print(
        tuple(
            round(
                value,
                2
            )
            for value in row[
                "centroid"
            ]
        )
    )

    print(
        f"Slice range: "
        f"{row['first_slice']} "
        f"to "
        f"{row['last_slice']}"
    )

    print(
        f"Maximum probability: "
        f"{row['maximum_probability']:.4f}"
    )

    print(
        f"Mean probability: "
        f"{row['mean_probability']:.4f}"
    )


# --------------------------------------------------
# 12. Summary
# --------------------------------------------------

total_voxels = np.count_nonzero(
    constrained_prediction
)

total_volume_mm3 = (
    total_voxels
    * voxel_volume_mm3
)

total_volume_ml = (
    total_volume_mm3
    / 1000.0
)


print()
print("=" * 80)

print()
print("Total constrained positive voxels:")
print(total_voxels)

print()
print("Total constrained predicted volume:")
print(
    f"{total_volume_mm3:.2f} mm^3"
)

print()
print("Total constrained predicted volume:")
print(
    f"{total_volume_ml:.4f} mL"
)


# --------------------------------------------------
# 13. Assertions
# --------------------------------------------------

assert num_components > 0

assert total_voxels > 0

assert sum(
    row["voxel_count"]
    for row in component_rows
) == total_voxels


print()
print(
    "Constrained component analysis passed."
)