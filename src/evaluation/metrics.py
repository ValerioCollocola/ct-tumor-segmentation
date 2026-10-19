import torch
import torch.nn.functional as F


def dice_score_per_class(
    outputs,
    labels,
    num_classes=3,
    smooth=1e-6
):
    predictions = torch.argmax(
        outputs,
        dim=1
    )

    dice_scores = []

    for class_index in range(num_classes):

        prediction_mask = (
            predictions == class_index
        )

        label_mask = (
            labels == class_index
        )

        intersection = torch.sum(
            prediction_mask & label_mask
        ).float()

        denominator = (
            torch.sum(
                prediction_mask
            ).float()
            +
            torch.sum(
                label_mask
            ).float()
        )

        if denominator.item() == 0:
            dice = 1.0

        else:
            dice = (
                (
                    2 * intersection
                    + smooth
                )
                /
                (
                    denominator
                    + smooth
                )
            ).item()

        dice_scores.append(
            dice
        )

    return dice_scores


def precision_recall_for_class(
    outputs,
    labels,
    class_index=2
):
    predictions = torch.argmax(
        outputs,
        dim=1
    )

    prediction_mask = (
        predictions == class_index
    )

    label_mask = (
        labels == class_index
    )

    true_positive = torch.sum(
        prediction_mask & label_mask
    ).float()

    false_positive = torch.sum(
        prediction_mask & ~label_mask
    ).float()

    false_negative = torch.sum(
        ~prediction_mask & label_mask
    ).float()

    predicted_positive = (
        true_positive
        + false_positive
    )

    actual_positive = (
        true_positive
        + false_negative
    )

    if predicted_positive.item() == 0:
        precision = 0.0

    else:
        precision = (
            true_positive
            / predicted_positive
        ).item()

    if actual_positive.item() == 0:
        recall = 0.0

    else:
        recall = (
            true_positive
            / actual_positive
        ).item()

    return (
        precision,
        recall
    )


def tumor_probability_stats(
    outputs,
    labels,
    tumor_class=2
):
    probabilities = F.softmax(
        outputs,
        dim=1
    )

    tumor_probabilities = probabilities[
        :,
        tumor_class,
        ...
    ]

    tumor_mask = (
        labels == tumor_class
    )

    non_tumor_mask = (
        labels != tumor_class
    )

    if torch.any(tumor_mask):

        mean_probability_on_tumor = (
            tumor_probabilities[
                tumor_mask
            ].mean().item()
        )

    else:
        mean_probability_on_tumor = 0.0

    if torch.any(non_tumor_mask):

        mean_probability_on_non_tumor = (
            tumor_probabilities[
                non_tumor_mask
            ].mean().item()
        )

    else:
        mean_probability_on_non_tumor = 0.0

    max_tumor_probability = (
        tumor_probabilities.max().item()
    )

    return (
        mean_probability_on_tumor,
        mean_probability_on_non_tumor,
        max_tumor_probability
    )


def tumor_metrics_at_threshold(
    outputs,
    labels,
    threshold=0.15,
    tumor_class=2
):
    probabilities = F.softmax(
        outputs,
        dim=1
    )

    tumor_probabilities = probabilities[
        :,
        tumor_class,
        ...
    ]

    prediction_mask = (
        tumor_probabilities
        >= threshold
    )

    label_mask = (
        labels == tumor_class
    )

    true_positive = torch.sum(
        prediction_mask & label_mask
    ).float()

    false_positive = torch.sum(
        prediction_mask & ~label_mask
    ).float()

    false_negative = torch.sum(
        ~prediction_mask & label_mask
    ).float()

    predicted_positive = (
        true_positive
        + false_positive
    )

    actual_positive = (
        true_positive
        + false_negative
    )

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if dice_denominator.item() == 0:
        dice = 1.0
    else:
        dice = (
            2 * true_positive
            / dice_denominator
        ).item()

    if predicted_positive.item() == 0:
        precision = 0.0
    else:
        precision = (
            true_positive
            / predicted_positive
        ).item()

    if actual_positive.item() == 0:
        recall = 0.0
    else:
        recall = (
            true_positive
            / actual_positive
        ).item()

    return (
        dice,
        precision,
        recall
    )

def iou_score_for_class(
    outputs,
    labels,
    class_index=1
):
    predictions = torch.argmax(
        outputs,
        dim=1
    )

    prediction_mask = (
        predictions == class_index
    )

    label_mask = (
        labels == class_index
    )

    intersection = torch.sum(
        prediction_mask & label_mask
    ).float()

    union = torch.sum(
        prediction_mask | label_mask
    ).float()

    if union.item() == 0:
        iou = 1.0

    else:
        iou = (
            intersection / union
        ).item()

    return iou