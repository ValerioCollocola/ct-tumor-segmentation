import numpy as np

from src.data.cases import discover_cases
from src.data.split import train_val_split
from src.preprocessing.pipeline import preprocess_case


# --------------------------------------------------
# 1. Scoperta dei casi
# --------------------------------------------------

cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)


# --------------------------------------------------
# 2. Patient-level split
# --------------------------------------------------

case_ids = [
    case["id"]
    for case in cases
]

train_ids, val_ids = train_val_split(
    case_ids,
    val_fraction=0.2,
    seed=42
)

train_cases = [
    case
    for case in cases
    if case["id"] in train_ids
]

val_cases = [
    case
    for case in cases
    if case["id"] in val_ids
]


# --------------------------------------------------
# 3. Preprocessing training cases
# --------------------------------------------------

train_image_patches = []
train_label_patches = []

for case in train_cases:

    print(
        f"Preprocessing training case: "
        f"{case['id']}"
    )

    result = preprocess_case(
        case
    )

    train_image_patches.append(
        result["image_patch"]
    )

    train_label_patches.append(
        result["label_patch"]
    )


# --------------------------------------------------
# 4. Preprocessing validation cases
# --------------------------------------------------

val_image_patches = []
val_label_patches = []

for case in val_cases:

    print(
        f"Preprocessing validation case: "
        f"{case['id']}"
    )

    result = preprocess_case(
        case
    )

    val_image_patches.append(
        result["image_patch"]
    )

    val_label_patches.append(
        result["label_patch"]
    )


# --------------------------------------------------
# 5. Controlli
# --------------------------------------------------

print("Training patches:")
print(len(train_image_patches))

print("Validation patches:")
print(len(val_image_patches))

print("Training patch shape:")
print(train_image_patches[0].shape)

print("Validation patch shape:")
print(val_image_patches[0].shape)

print("Training label values:")
print(
    np.unique(
        train_label_patches[0]
    )
)


assert len(train_image_patches) == 8
assert len(train_label_patches) == 8

assert len(val_image_patches) == 2
assert len(val_label_patches) == 2

assert train_image_patches[0].shape == (
    96,
    96,
    64
)

assert val_image_patches[0].shape == (
    96,
    96,
    64
)