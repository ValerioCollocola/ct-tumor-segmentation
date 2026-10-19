import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    outputs,
    labels,
    num_classes=3,
    smooth=1e-6
):
    probabilities = F.softmax(
        outputs,
        dim=1
    )

    labels_one_hot = F.one_hot(
        labels,
        num_classes=num_classes
    )

    labels_one_hot = labels_one_hot.permute(
        0,
        4,
        1,
        2,
        3
    ).float()

    intersection = torch.sum(
        probabilities * labels_one_hot,
        dim=(0, 2, 3, 4)
    )

    denominator = (
        torch.sum(
            probabilities,
            dim=(0, 2, 3, 4)
        )
        +
        torch.sum(
            labels_one_hot,
            dim=(0, 2, 3, 4)
        )
    )

    dice_score = (
        2 * intersection + smooth
    ) / (
        denominator + smooth
    )

    loss = 1 - dice_score.mean()

    return loss

def tumor_dice_loss(
    outputs,
    labels,
    tumor_class=2,
    smooth=1e-6
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

    tumor_ground_truth = (
        labels == tumor_class
    ).float()

    intersection = torch.sum(
        tumor_probabilities
        * tumor_ground_truth
    )

    denominator = (
        torch.sum(
            tumor_probabilities
        )
        +
        torch.sum(
            tumor_ground_truth
        )
    )

    dice_score = (
        2 * intersection + smooth
    ) / (
        denominator + smooth
    )

    loss = 1 - dice_score

    return loss

def combined_loss(
    outputs,
    labels,
    num_classes=3,
    dice_weight=1.0,
    ce_weight=1.0,
    tumor_dice_weight=1.0,
    class_weights=None
):
    if class_weights is not None:

        class_weights = class_weights.to(
            outputs.device
        )

    cross_entropy = nn.CrossEntropyLoss(
        weight=class_weights
    )

    ce_loss = cross_entropy(
        outputs,
        labels
    )

    d_loss = dice_loss(
        outputs,
        labels,
        num_classes=num_classes
    )

    t_loss = tumor_dice_loss(
        outputs,
        labels,
        tumor_class=2
    )

    total_loss = (
        ce_weight * ce_loss
        +
        dice_weight * d_loss
        +
        tumor_dice_weight * t_loss
    )

    return total_loss, ce_loss, d_loss

def binary_combined_loss(
    outputs,
    labels,
    class_weights=None,
    ce_weight=1.0,
    dice_weight=1.0,
    smooth=1e-6
):
    if class_weights is not None:
        class_weights = class_weights.to(
            outputs.device
        )

    cross_entropy = nn.CrossEntropyLoss(
        weight=class_weights
    )

    ce_loss = cross_entropy(
        outputs,
        labels
    )

    probabilities = F.softmax(
        outputs,
        dim=1
    )

    tumor_probabilities = probabilities[
        :,
        1,
        ...
    ]

    tumor_ground_truth = (
        labels == 1
    ).float()

    intersection = torch.sum(
        tumor_probabilities
        * tumor_ground_truth
    )

    denominator = (
        torch.sum(
            tumor_probabilities
        )
        +
        torch.sum(
            tumor_ground_truth
        )
    )

    dice_score = (
        2 * intersection + smooth
    ) / (
        denominator + smooth
    )

    dice_loss_value = (
        1 - dice_score
    )

    total_loss = (
        ce_weight * ce_loss
        +
        dice_weight * dice_loss_value
    )

    return (
        total_loss,
        ce_loss,
        dice_loss_value
    )