import numpy as np

from src.evaluation.lesion_metrics import (
    remove_small_components,
    lesion_sizes
)


mask = np.zeros(
    (32, 32, 16),
    dtype=np.uint8
)


# Componente grande
mask[
    5:10,
    5:10,
    4:7
] = 1


# Componente piccola
mask[
    20,
    20,
    10
] = 1


print("Original lesion sizes:")
print(
    lesion_sizes(mask)
)


filtered_mask = remove_small_components(
    mask,
    min_size=20
)


print("Filtered lesion sizes:")
print(
    lesion_sizes(
        filtered_mask
    )
)


assert lesion_sizes(
    filtered_mask
) == [75]