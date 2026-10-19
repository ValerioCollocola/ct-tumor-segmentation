import csv
from pathlib import Path

import numpy as np
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

from src.evaluation.metrics import (
    dice_score_per_class,
    precision_recall_for_class,
    iou_score_for_class
)

from src.evaluation.lesion_metrics import (
    lesion_wise_evaluation,
    remove_small_components
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
# 4. Best checkpoint per Tumor Dice
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
# 5. Evaluation case-by-case
# --------------------------------------------------

results = []

for case in val_cases:

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


    # ==================================================
    # RAW VOXEL-WISE METRICS
    # ==================================================

    dice_scores = dice_score_per_class(
        outputs,
        label_tensor,
        num_classes=2
    )

    raw_precision, raw_recall = (
        precision_recall_for_class(
            outputs,
            label_tensor,
            class_index=1
        )
    )

    raw_iou = iou_score_for_class(
        outputs,
        label_tensor,
        class_index=1
    )

    raw_dice = dice_scores[1]


    # ==================================================
    # NUMPY CONVERSION
    # ==================================================

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


    # ==================================================
    # RAW LESION-WISE METRICS
    # ==================================================

    raw_lesion_metrics = lesion_wise_evaluation(
        prediction_mask=prediction_numpy,
        ground_truth_mask=label_numpy
    )


    # ==================================================
    # POST-PROCESSING
    # ==================================================

    filtered_prediction = remove_small_components(
        prediction_numpy,
        min_size=20
    )


    # ==================================================
    # FILTERED VOXEL-WISE METRICS
    # ==================================================

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

    # Dice
    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if dice_denominator == 0:
        filtered_dice = 1.0
    else:
        filtered_dice = (
            2 * true_positive
            / dice_denominator
        )

    # IoU
    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    if iou_denominator == 0:
        filtered_iou = 1.0
    else:
        filtered_iou = (
            true_positive
            / iou_denominator
        )

    # Precision
    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:
        filtered_precision = 0.0
    else:
        filtered_precision = (
            true_positive
            / precision_denominator
        )

    # Recall
    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:
        filtered_recall = 0.0
    else:
        filtered_recall = (
            true_positive
            / recall_denominator
        )


    # ==================================================
    # FILTERED LESION-WISE METRICS
    # ==================================================

    filtered_lesion_metrics = lesion_wise_evaluation(
        prediction_mask=filtered_prediction,
        ground_truth_mask=label_numpy
    )


    # ==================================================
    # VOXEL COUNTS
    # ==================================================

    ground_truth_tumor_voxels = np.count_nonzero(
        ground_truth_mask
    )

    raw_predicted_tumor_voxels = np.count_nonzero(
        prediction_numpy
    )

    filtered_predicted_tumor_voxels = np.count_nonzero(
        filtered_prediction
    )


    # ==================================================
    # RESULT ROW
    # ==================================================

    results.append(
        {
            "case_id": case["id"],

            "raw_dice": raw_dice,
            "filtered_dice": filtered_dice,

            "raw_iou": raw_iou,
            "filtered_iou": filtered_iou,

            "raw_precision": raw_precision,
            "filtered_precision": filtered_precision,

            "raw_recall": raw_recall,
            "filtered_recall": filtered_recall,

            "gt_tumor_voxels": ground_truth_tumor_voxels,

            "raw_pred_tumor_voxels":
                raw_predicted_tumor_voxels,

            "filtered_pred_tumor_voxels":
                filtered_predicted_tumor_voxels,

            "gt_lesions":
                raw_lesion_metrics["gt_lesions"],

            "raw_pred_lesions":
                raw_lesion_metrics["pred_lesions"],

            "filtered_pred_lesions":
                filtered_lesion_metrics["pred_lesions"],

            "raw_detected_lesions":
                raw_lesion_metrics["detected_lesions"],

            "filtered_detected_lesions":
                filtered_lesion_metrics["detected_lesions"],

            "raw_missed_lesions":
                raw_lesion_metrics["missed_lesions"],

            "filtered_missed_lesions":
                filtered_lesion_metrics["missed_lesions"],

            "raw_false_positive_lesions":
                raw_lesion_metrics[
                    "false_positive_lesions"
                ],

            "filtered_false_positive_lesions":
                filtered_lesion_metrics[
                    "false_positive_lesions"
                ],

            "raw_lesion_sensitivity":
                raw_lesion_metrics[
                    "lesion_sensitivity"
                ],

            "filtered_lesion_sensitivity":
                filtered_lesion_metrics[
                    "lesion_sensitivity"
                ]
        }
    )


# --------------------------------------------------
# 6. Console summary
# --------------------------------------------------

print()
print("RAW vs FILTERED evaluation")
print("=" * 80)

for row in results:

    print()
    print(f"Case: {row['case_id']}")

    print(
        f"Dice: "
        f"{row['raw_dice']:.4f} "
        f"-> {row['filtered_dice']:.4f}"
    )

    print(
        f"IoU: "
        f"{row['raw_iou']:.4f} "
        f"-> {row['filtered_iou']:.4f}"
    )

    print(
        f"Precision: "
        f"{row['raw_precision']:.4f} "
        f"-> {row['filtered_precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{row['raw_recall']:.4f} "
        f"-> {row['filtered_recall']:.4f}"
    )

    print(
        f"Predicted lesions: "
        f"{row['raw_pred_lesions']} "
        f"-> {row['filtered_pred_lesions']}"
    )

    print(
        f"False-positive lesions: "
        f"{row['raw_false_positive_lesions']} "
        f"-> "
        f"{row['filtered_false_positive_lesions']}"
    )

    print(
        f"Lesion sensitivity: "
        f"{row['raw_lesion_sensitivity']:.4f} "
        f"-> "
        f"{row['filtered_lesion_sensitivity']:.4f}"
    )


# --------------------------------------------------
# 7. Mean metrics
# --------------------------------------------------

mean_raw_dice = sum(
    row["raw_dice"]
    for row in results
) / len(results)

mean_filtered_dice = sum(
    row["filtered_dice"]
    for row in results
) / len(results)

mean_raw_precision = sum(
    row["raw_precision"]
    for row in results
) / len(results)

mean_filtered_precision = sum(
    row["filtered_precision"]
    for row in results
) / len(results)

mean_raw_recall = sum(
    row["raw_recall"]
    for row in results
) / len(results)

mean_filtered_recall = sum(
    row["filtered_recall"]
    for row in results
) / len(results)


print()
print("=" * 80)

print("Mean RAW Dice:")
print(mean_raw_dice)

print("Mean FILTERED Dice:")
print(mean_filtered_dice)

print("Mean RAW Precision:")
print(mean_raw_precision)

print("Mean FILTERED Precision:")
print(mean_filtered_precision)

print("Mean RAW Recall:")
print(mean_raw_recall)

print("Mean FILTERED Recall:")
print(mean_filtered_recall)


# --------------------------------------------------
# 8. CSV
# --------------------------------------------------

output_path = Path(
    "outputs/evaluation/"
    "binary_validation_raw_vs_filtered.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with output_path.open(
    mode="w",
    newline="",
    encoding="utf-8"
) as csv_file:

    fieldnames = [
        "case_id",

        "raw_dice",
        "filtered_dice",

        "raw_iou",
        "filtered_iou",

        "raw_precision",
        "filtered_precision",

        "raw_recall",
        "filtered_recall",

        "gt_tumor_voxels",

        "raw_pred_tumor_voxels",
        "filtered_pred_tumor_voxels",

        "gt_lesions",

        "raw_pred_lesions",
        "filtered_pred_lesions",

        "raw_detected_lesions",
        "filtered_detected_lesions",

        "raw_missed_lesions",
        "filtered_missed_lesions",

        "raw_false_positive_lesions",
        "filtered_false_positive_lesions",

        "raw_lesion_sensitivity",
        "filtered_lesion_sensitivity"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in results:

        writer.writerow(
            row
        )


print()
print("RAW vs FILTERED CSV saved to:")
print(output_path)


# --------------------------------------------------
# 9. Controlli
# --------------------------------------------------

assert len(results) == 2

for row in results:

    assert 0 <= row["raw_dice"] <= 1
    assert 0 <= row["filtered_dice"] <= 1

    assert 0 <= row["raw_iou"] <= 1
    assert 0 <= row["filtered_iou"] <= 1

    assert 0 <= row["raw_precision"] <= 1
    assert 0 <= row["filtered_precision"] <= 1

    assert 0 <= row["raw_recall"] <= 1
    assert 0 <= row["filtered_recall"] <= 1

assert output_path.exists()