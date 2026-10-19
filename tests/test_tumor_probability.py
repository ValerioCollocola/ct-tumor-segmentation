import torch

from src.evaluation.metrics import (
    tumor_probability_stats
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


tumor_prob, non_tumor_prob, max_prob = (
    tumor_probability_stats(
        outputs,
        labels,
        tumor_class=2
    )
)


print("Mean tumor probability on true tumor voxels:")
print(tumor_prob)

print("Mean tumor probability on non-tumor voxels:")
print(non_tumor_prob)

print("Maximum tumor probability:")
print(max_prob)


assert 0 <= tumor_prob <= 1
assert 0 <= non_tumor_prob <= 1
assert 0 <= max_prob <= 1