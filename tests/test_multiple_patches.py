import numpy as np

from src.data.cases import discover_cases
from src.preprocessing.pipeline import (
    preprocess_case_multiple_patches
)


cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)

result = preprocess_case_multiple_patches(
    cases[0],
    num_patches=5
)

print("Case ID:")
print(result["id"])

print("Number of image patches:")
print(len(result["image_patches"]))

print("Number of label patches:")
print(len(result["label_patches"]))

print("Patch centers:")
print(result["centers"])

print("First image patch shape:")
print(result["image_patches"][0].shape)

print("First label patch shape:")
print(result["label_patches"][0].shape)


assert len(result["image_patches"]) == 5
assert len(result["label_patches"]) == 5

assert result["image_patches"][0].shape == (
    96,
    96,
    64
)

assert result["label_patches"][0].shape == (
    96,
    96,
    64
)

print("Tumor voxels per patch:")

for index, label_patch in enumerate(
    result["label_patches"]
):

    tumor_voxels = np.count_nonzero(
        label_patch == 2
    )

    print(
        f"Patch {index + 1}: "
        f"{tumor_voxels}"
    )

    assert tumor_voxels > 0