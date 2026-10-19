import numpy as np

from src.data.cases import discover_cases
from src.preprocessing.pipeline import preprocess_case


cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)

case = cases[0]

result = preprocess_case(
    case
)


print("Case ID:")
print(result["id"])

print("Original shape:")
print(result["original_shape"])

print("Resampled shape:")
print(result["resampled_shape"])

print("Original spacing:")
print(result["original_spacing"])

print("Target spacing:")
print(result["target_spacing"])

print("Tumor center:")
print(result["tumor_center"])

print("Image patch shape:")
print(result["image_patch"].shape)

print("Label patch shape:")
print(result["label_patch"].shape)

print("Image patch range:")
print(
    result["image_patch"].min(),
    result["image_patch"].max()
)

print("Label values:")
print(
    np.unique(
        result["label_patch"]
    )
)


assert result["image_patch"].shape == (
    96,
    96,
    64
)

assert result["label_patch"].shape == (
    96,
    96,
    64
)

assert 2 in np.unique(
    result["label_patch"]
)