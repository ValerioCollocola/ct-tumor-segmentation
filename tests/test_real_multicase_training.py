import torch

from torch.utils.data import DataLoader

from src.data.cases import discover_cases
from src.data.split import train_val_split
from src.data.dataset import TumorSegmentationDataset

from src.preprocessing.pipeline import (
    preprocess_case,
    preprocess_case_multiple_patches
)

from src.models.unet3d import UNet3D
from src.training.train import train_and_validate


# --------------------------------------------------
# 1. Scoperta dei casi reali
# --------------------------------------------------

cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)


# --------------------------------------------------
# 2. Patient-level split
# --------------------------------------------------

case_ids = [
    case["id"]
    for case in cases
]

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
# 3. Preprocessing training cases
#    5 patch tumor-aware per paziente
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

    train_images.extend(
        result["image_patches"]
    )

    train_labels.extend(
        result["label_patches"]
    )


# --------------------------------------------------
# 4. Preprocessing validation cases
#    1 patch per paziente
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

    val_images.append(
        result["image_patch"]
    )

    val_labels.append(
        result["label_patch"]
    )


# --------------------------------------------------
# 5. Informazioni dataset
# --------------------------------------------------

print("Number of training patches:")
print(len(train_images))

print("Number of validation patches:")
print(len(val_images))


# --------------------------------------------------
# 6. Dataset
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
# 7. DataLoader
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
# 8. 3D U-Net
# --------------------------------------------------

model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)


# --------------------------------------------------
# 9. Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# --------------------------------------------------
# 10. Class weights
#     derivati dalla distribuzione multipatch
# --------------------------------------------------

class_weights = torch.tensor(
    [
        0.26,  # background
        0.27,  # liver
        2.47   # tumor
    ],
    dtype=torch.float32
)


# --------------------------------------------------
# 11. Training reale multi-case
# --------------------------------------------------

history = train_and_validate(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    num_epochs=10,
    device="cpu",
    checkpoint_path=(
        "outputs/checkpoints/"
        "best_real_multicase_unet_tumor_aware.pt"
    ),
    class_weights=class_weights
)


# --------------------------------------------------
# 12. Output
# --------------------------------------------------

print("Real multi-case training history:")
print(history)


# --------------------------------------------------
# 13. Controlli
# --------------------------------------------------

assert len(train_images) == 40
assert len(train_labels) == 40

assert len(val_images) == 2
assert len(val_labels) == 2

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
    history["tumor_dice"]
) == 10