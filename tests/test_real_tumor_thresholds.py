import torch

from torch.utils.data import DataLoader

from src.data.cases import discover_cases
from src.data.split import train_val_split
from src.data.dataset import TumorSegmentationDataset

from src.preprocessing.pipeline import preprocess_case

from src.models.unet3d import UNet3D
from src.training.checkpoint import load_checkpoint

from src.evaluation.metrics import (
    tumor_metrics_at_threshold
)


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

val_cases = [
    case
    for case in cases
    if case["id"] in val_ids
]


# --------------------------------------------------
# 3. Modello
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


# --------------------------------------------------
# 4. Carica checkpoint
# --------------------------------------------------

model, optimizer, epoch, loss = load_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_path=(
        "outputs/checkpoints/"
        "best_real_multicase_unet_tumor_aware.pt"
    ),
    device="cpu"
)

model.eval()


# --------------------------------------------------
# 5. Threshold da testare
# --------------------------------------------------

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]


# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

for case in val_cases:

    print()
    print("Validation case:")
    print(case["id"])

    result = preprocess_case(
        case
    )

    dataset = TumorSegmentationDataset(
        image_patches=[
            result["image_patch"]
        ],
        label_patches=[
            result["label_patch"]
        ]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    images, labels = next(
        iter(dataloader)
    )

    with torch.no_grad():

        outputs = model(
            images
        )

    for threshold in thresholds:

        dice, precision, recall = (
            tumor_metrics_at_threshold(
                outputs,
                labels,
                threshold=threshold,
                tumor_class=2
            )
        )

        print()
        print(
            f"Threshold: {threshold:.2f}"
        )

        print(
            f"Tumor Dice: {dice:.4f}"
        )

        print(
            f"Tumor Precision: {precision:.4f}"
        )

        print(
            f"Tumor Recall: {recall:.4f}"
        )