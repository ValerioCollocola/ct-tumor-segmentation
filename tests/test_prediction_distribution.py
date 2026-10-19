import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.cases import discover_cases
from src.data.split import train_val_split
from src.data.dataset import TumorSegmentationDataset

from src.preprocessing.pipeline import preprocess_case

from src.models.unet3d import UNet3D
from src.training.checkpoint import load_checkpoint


# --------------------------------------------------
# 1. Casi reali
# --------------------------------------------------

cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)

case_ids = [
    case["id"]
    for case in cases
]

train_ids, val_ids = train_val_split(
    case_ids,
    val_fraction=0.2,
    seed=42
)

val_cases = [
    case
    for case in cases
    if case["id"] in val_ids
]


# --------------------------------------------------
# 2. Preprocessing validation
# --------------------------------------------------

val_images = []
val_labels = []

for case in val_cases:

    print(
        f"Preparing validation case: "
        f"{case['id']}"
    )

    result = preprocess_case(case)

    val_images.append(
        result["image_patch"]
    )

    val_labels.append(
        result["label_patch"]
    )


# --------------------------------------------------
# 3. Dataset + DataLoader
# --------------------------------------------------

val_dataset = TumorSegmentationDataset(
    image_patches=val_images,
    label_patches=val_labels
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)


# --------------------------------------------------
# 4. Carica best model
# --------------------------------------------------

model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

model, optimizer, epoch, loss = load_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_path=(
    "outputs/checkpoints/"
    "best_real_multicase_unet_weighted.pt"
    ),
    device="cpu"
)

model.eval()


# --------------------------------------------------
# 5. Analizza le predizioni
# --------------------------------------------------

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        val_dataloader
    ):

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        print(
            f"\nValidation case {batch_index + 1}"
        )

        for class_index in range(3):

            predicted_voxels = torch.sum(
                predictions == class_index
            ).item()

            true_voxels = torch.sum(
                labels == class_index
            ).item()

            print(
                f"Class {class_index} - "
                f"Predicted: {predicted_voxels} - "
                f"Ground truth: {true_voxels}"
            )