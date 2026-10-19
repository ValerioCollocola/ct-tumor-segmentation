import torch

from torch.utils.data import DataLoader

from src.data.cases import discover_cases
from src.data.split import train_val_split
from src.data.dataset import TumorSegmentationDataset

from src.preprocessing.pipeline import preprocess_case

from src.preprocessing.labels import (
    convert_to_binary_tumor_mask
)

from src.models.unet3d import UNet3D
from src.training.checkpoint import load_checkpoint

from src.evaluation.metrics import (
    dice_score_per_class,
    precision_recall_for_class
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
# 3. Binary U-Net
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


# --------------------------------------------------
# 4. Carica checkpoint con miglior Tumor Dice
# --------------------------------------------------

model, optimizer, epoch, loss = load_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_path=(
        "outputs/checkpoints/"
        "best_real_binary_unet_best_tumor_dice.pt"
    ),
    device="cpu"
)

model.eval()


print("Loaded checkpoint epoch:")
print(epoch)

print("Checkpoint validation loss:")
print(loss)


# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------

for case in val_cases:

    print()
    print("Validation case:")
    print(case["id"])

    result = preprocess_case(
        case
    )

    binary_label = convert_to_binary_tumor_mask(
        result["label_patch"]
    )

    dataset = TumorSegmentationDataset(
        image_patches=[
            result["image_patch"]
        ],
        label_patches=[
            binary_label
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

    # Binary:
    # class 0 = non-tumor
    # class 1 = tumor

    dice_scores = dice_score_per_class(
        outputs,
        labels,
        num_classes=2
    )

    precision, recall = precision_recall_for_class(
        outputs,
        labels,
        class_index=1
    )

    print("Tumor Dice:")
    print(dice_scores[1])

    print("Tumor Precision:")
    print(precision)

    print("Tumor Recall:")
    print(recall)