import torch

from src.evaluation.metrics import dice_score_per_class


# -----------------------------
# Output fittizio del modello
# -----------------------------

outputs = torch.randn(
    1,
    3,
    32,
    32,
    16
)


# -----------------------------
# Ground truth fittizia
# -----------------------------

labels = torch.randint(
    low=0,
    high=3,
    size=(
        1,
        32,
        32,
        16
    ),
    dtype=torch.long
)


# -----------------------------
# Dice per classe
# -----------------------------

dice_scores = dice_score_per_class(
    outputs,
    labels,
    num_classes=3
)


print("Background Dice:")
print(dice_scores[0])

print("Liver Dice:")
print(dice_scores[1])

print("Tumor Dice:")
print(dice_scores[2])


# -----------------------------
# Controlli
# -----------------------------

assert len(dice_scores) == 3

for score in dice_scores:
    assert 0 <= score <= 1