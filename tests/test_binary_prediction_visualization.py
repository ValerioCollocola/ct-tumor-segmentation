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

from src.visualization.plot_binary_prediction import (
    plot_binary_segmentation_prediction
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
# 2. Split
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
# 4. Carica best checkpoint Tumor Dice
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
# 5. Visualizza tutti i casi validation
# --------------------------------------------------

for case in val_cases:

    print()
    print("Visualizing validation case:")
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

    image_tensor = image_tensor.unsqueeze(
        0
    )

    label_tensor = label_tensor.unsqueeze(
        0
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

    plot_binary_segmentation_prediction(
        image=image_tensor,
        label=label_tensor,
        outputs=outputs,
        output_path=(
            "outputs/"
            f"{case['id']}_binary_prediction.png"
        )
    )