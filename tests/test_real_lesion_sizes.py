import torch

from src.data.cases import discover_cases
from src.data.split import train_val_split
from src.data.dataset import TumorSegmentationDataset

from src.preprocessing.pipeline import preprocess_case
from src.preprocessing.labels import (
    convert_to_binary_tumor_mask
)

from src.models.unet3d import UNet3D
from src.training.checkpoint import load_checkpoint

from src.evaluation.lesion_metrics import (
    lesion_sizes
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
# 3. Modello binario
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
# 4. Carica best Tumor Dice checkpoint
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


# --------------------------------------------------
# 5. Analisi dimensioni lesioni
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

    image_tensor, label_tensor = dataset[0]

    image_tensor = image_tensor.unsqueeze(0)
    label_tensor = label_tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

    predictions = torch.argmax(
        outputs,
        dim=1
    )

    prediction_numpy = (
        predictions
        .squeeze(0)
        .cpu()
        .numpy()
    )

    label_numpy = (
        label_tensor
        .squeeze(0)
        .cpu()
        .numpy()
    )

    gt_sizes = lesion_sizes(
        label_numpy
    )

    pred_sizes = lesion_sizes(
        prediction_numpy
    )


    print()
    print("Ground-truth lesion sizes:")

    for index, size in enumerate(
        gt_sizes,
        start=1
    ):

        print(
            f"GT lesion {index}: "
            f"{size} voxels"
        )


    print()
    print("Predicted lesion sizes:")

    if len(pred_sizes) == 0:

        print(
            "No predicted lesions."
        )

    else:

        for index, size in enumerate(
            pred_sizes,
            start=1
        ):

            print(
                f"Pred lesion {index}: "
                f"{size} voxels"
            )