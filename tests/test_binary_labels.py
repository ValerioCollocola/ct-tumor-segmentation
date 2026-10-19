import numpy as np

from src.data.cases import discover_cases
from src.preprocessing.pipeline import preprocess_case
from src.preprocessing.labels import (
    convert_to_binary_tumor_mask
)


cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)

result = preprocess_case(
    cases[0]
)

label_patch = result["label_patch"]

binary_label = convert_to_binary_tumor_mask(
    label_patch
)


print("Original label values:")
print(
    np.unique(label_patch)
)

print("Binary label values:")
print(
    np.unique(binary_label)
)

print("Original tumor voxels:")
print(
    np.count_nonzero(
        label_patch == 2
    )
)

print("Binary tumor voxels:")
print(
    np.count_nonzero(
        binary_label == 1
    )
)


assert np.array_equal(
    np.unique(binary_label),
    np.array([0, 1])
)

assert np.count_nonzero(
    label_patch == 2
) == np.count_nonzero(
    binary_label == 1
)