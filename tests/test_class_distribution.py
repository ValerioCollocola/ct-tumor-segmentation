import numpy as np

from src.data.cases import discover_cases
from src.data.split import train_val_split
from src.preprocessing.pipeline import preprocess_case


# --------------------------------------------------
# 1. Scoperta casi
# --------------------------------------------------

cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)

case_ids = [
    case["id"]
    for case in cases
]


# --------------------------------------------------
# 2. Patient-level split
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Conteggio voxel per classe
# --------------------------------------------------

class_counts = {
    0: 0,
    1: 0,
    2: 0
}

for case in train_cases:

    print(
        f"Analyzing training case: "
        f"{case['id']}"
    )

    result = preprocess_case(
        case
    )

    label_patch = result["label_patch"]

    for class_index in class_counts:

        class_counts[class_index] += np.count_nonzero(
            label_patch == class_index
        )


# --------------------------------------------------
# 4. Totale voxel
# --------------------------------------------------

total_voxels = sum(
    class_counts.values()
)


# --------------------------------------------------
# 5. Percentuali
# --------------------------------------------------

class_percentages = {}

for class_index, count in class_counts.items():

    percentage = (
        count / total_voxels
    ) * 100

    class_percentages[class_index] = percentage


# --------------------------------------------------
# 6. Output
# --------------------------------------------------

print()
print("Class distribution:")

print("Background voxels:")
print(class_counts[0])

print("Liver voxels:")
print(class_counts[1])

print("Tumor voxels:")
print(class_counts[2])

print()

print("Background percentage:")
print(class_percentages[0])

print("Liver percentage:")
print(class_percentages[1])

print("Tumor percentage:")
print(class_percentages[2])


# --------------------------------------------------
# 7. Controlli
# --------------------------------------------------

assert total_voxels > 0

assert (
    class_counts[0]
    + class_counts[1]
    + class_counts[2]
) == total_voxels