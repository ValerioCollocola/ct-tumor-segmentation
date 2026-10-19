import torch

from src.evaluation.metrics import (
    precision_recall_for_class
)


outputs = torch.randn(
    1,
    3,
    32,
    32,
    16
)

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


precision, recall = precision_recall_for_class(
    outputs,
    labels,
    class_index=2
)


print("Tumor Precision:")
print(precision)

print("Tumor Recall:")
print(recall)


assert 0 <= precision <= 1
assert 0 <= recall <= 1