import torch

from torch.utils.data import DataLoader

from src.data.cases import discover_cases
from src.data.split import train_val_split
from src.data.dataset import TumorSegmentationDataset

from src.preprocessing.pipeline import (
    preprocess_case,
    preprocess_case_multiple_patches
)

from src.preprocessing.labels import (
    convert_to_binary_tumor_mask
)

from src.models.unet3d import UNet3D
from src.training.train_binary import (
    train_and_validate_binary
)


# --------------------------------------------------
# Riproducibilità
# --------------------------------------------------

torch.manual_seed(42)


# --------------------------------------------------
# 1. Scoperta casi
# --------------------------------------------------

cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)

case_ids = [
    case["id"]
    for case in cases
]


# --------------------------------------------------
# 2. Patient-level split
# --------------------------------------------------

train_ids, val_ids = train_val_split(
    case_ids,
    val_fraction=0.2,
    seed=42
)

train_cases = [
    case
    for case in cases
    if case["id"] in train_ids
]

val_cases = [
    case
    for case in cases
    if case["id"] in val_ids
]


# --------------------------------------------------
# 3. Training patches
# --------------------------------------------------

train_images = []
train_labels = []

for case in train_cases:

    print(
        f"Preparing training case: "
        f"{case['id']}"
    )

    result = preprocess_case_multiple_patches(
        case,
        num_patches=5,
        seed=42
    )

    for image_patch, label_patch in zip(
        result["image_patches"],
        result["label_patches"]
    ):

        binary_label = convert_to_binary_tumor_mask(
            label_patch
        )

        train_images.append(
            image_patch
        )

        train_labels.append(
            binary_label
        )


# --------------------------------------------------
# 4. Validation patches
# --------------------------------------------------

val_images = []
val_labels = []

for case in val_cases:

    print(
        f"Preparing validation case: "
        f"{case['id']}"
    )

    result = preprocess_case(
        case
    )

    binary_label = convert_to_binary_tumor_mask(
        result["label_patch"]
    )

    val_images.append(
        result["image_patch"]
    )

    val_labels.append(
        binary_label
    )


print("Binary training patches:")
print(len(train_images))

print("Binary validation patches:")
print(len(val_images))


# --------------------------------------------------
# 5. Dataset
# --------------------------------------------------

train_dataset = TumorSegmentationDataset(
    image_patches=train_images,
    label_patches=train_labels
)

val_dataset = TumorSegmentationDataset(
    image_patches=val_images,
    label_patches=val_labels
)


# --------------------------------------------------
# 6. DataLoader
# --------------------------------------------------

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)


# --------------------------------------------------
# 7. Binary 3D U-Net
# --------------------------------------------------

model = UNet3D(
    in_channels=1,
    num_classes=2,
    base_channels=8
)


# --------------------------------------------------
# 8. Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# --------------------------------------------------
# 9. Binary class weights
#
# Circa:
# non-tumor = 94.9%
# tumor     = 5.1%
# --------------------------------------------------

class_weights = torch.tensor(
    [
        0.10,
        1.90
    ],
    dtype=torch.float32
)


# --------------------------------------------------
# 10. Training
# --------------------------------------------------

history = train_and_validate_binary(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    num_epochs=10,
    device="cpu",
    checkpoint_path=(
        "outputs/checkpoints/"
        "best_real_binary_unet.pt"
    ),
    class_weights=class_weights
)


# --------------------------------------------------
# 11. Output
# --------------------------------------------------

print("Binary training history:")
print(history)


# --------------------------------------------------
# 12. Controlli
# --------------------------------------------------

assert len(train_images) == 40
assert len(val_images) == 2

assert len(
    history["train_loss"]
) == 10

assert len(
    history["val_loss"]
) == 10

assert len(
    history["train_tumor_dice"]
) == 10

assert len(
    history["val_tumor_dice"]
) == 10