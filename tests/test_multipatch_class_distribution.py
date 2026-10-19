import numpy as np

from src.data.cases import discover_cases
from src.data.split import train_val_split

from src.preprocessing.pipeline import (
    preprocess_case_multiple_patches
)


# --------------------------------------------------
# 1. Casi reali
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
# 3. Conteggio classi
# --------------------------------------------------

class_counts = {
    0: 0,
    1: 0,
    2: 0
}

number_of_patches = 0


for case in train_cases:

    print(
        f"Analyzing case: {case['id']}"
    )

    result = preprocess_case_multiple_patches(
        case,
        num_patches=5,
        seed=42
    )

    for label_patch in result["label_patches"]:

        number_of_patches += 1

        for class_index in class_counts:

            class_counts[class_index] += (
                np.count_nonzero(
                    label_patch == class_index
                )
            )


# --------------------------------------------------
# 4. Totale
# --------------------------------------------------

total_voxels = sum(
    class_counts.values()
)


# --------------------------------------------------
# 5. Percentuali
# --------------------------------------------------

class_percentages = {}

for class_index, count in class_counts.items():

    class_percentages[class_index] = (
        count / total_voxels
    ) * 100


# --------------------------------------------------
# 6. Output
# --------------------------------------------------

print()
print("Number of training patches:")
print(number_of_patches)

print()

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

assert number_of_patches == 40
assert total_voxels > 0