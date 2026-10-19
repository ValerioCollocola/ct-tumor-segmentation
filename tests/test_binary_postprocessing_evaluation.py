import torch
import numpy as np

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
    remove_small_components,
    lesion_wise_evaluation
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
# 3. Modello
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
# 4. Checkpoint
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

    # --------------------------------------------------
    # Raw prediction
    # --------------------------------------------------

    raw_metrics = lesion_wise_evaluation(
        prediction_mask=prediction_numpy,
        ground_truth_mask=label_numpy
    )

    # --------------------------------------------------
    # Filtered prediction
    # --------------------------------------------------

    filtered_prediction = remove_small_components(
        prediction_numpy,
        min_size=20
    )

    filtered_metrics = lesion_wise_evaluation(
        prediction_mask=filtered_prediction,
        ground_truth_mask=label_numpy
    )

    # --------------------------------------------------
    # Voxel metrics post-processing
    # --------------------------------------------------

    prediction_mask = (
        filtered_prediction > 0
    )

    ground_truth_mask = (
        label_numpy > 0
    )

    true_positive = np.count_nonzero(
        prediction_mask & ground_truth_mask
    )

    false_positive = np.count_nonzero(
        prediction_mask & ~ground_truth_mask
    )

    false_negative = np.count_nonzero(
        ~prediction_mask & ground_truth_mask
    )

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (
            2 * true_positive
            / dice_denominator
        )

    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            true_positive
            / precision_denominator
        )

    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            true_positive
            / recall_denominator
        )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print()
    print("RAW lesion-wise metrics:")
    print(raw_metrics)

    print()
    print("FILTERED lesion-wise metrics:")
    print(filtered_metrics)

    print()
    print("Filtered Tumor Dice:")
    print(dice)

    print("Filtered Tumor Precision:")
    print(precision)

    print("Filtered Tumor Recall:")
    print(recall)