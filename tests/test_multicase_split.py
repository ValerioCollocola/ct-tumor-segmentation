from src.data.cases import discover_cases
from src.data.split import train_val_split


# -----------------------------
# Scoperta automatica dei casi
# -----------------------------

cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)


# -----------------------------
# Estrazione dei case ID
# -----------------------------

case_ids = [
    case["id"]
    for case in cases
]


# -----------------------------
# Train / validation split
# -----------------------------

train_ids, val_ids = train_val_split(
    case_ids,
    val_fraction=0.2,
    seed=42
)


# -----------------------------
# Costruzione dei casi completi
# -----------------------------

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


# -----------------------------
# Controllo overlap
# -----------------------------

train_id_set = {
    case["id"]
    for case in train_cases
}

val_id_set = {
    case["id"]
    for case in val_cases
}

overlap = (
    train_id_set
    &
    val_id_set
)


# -----------------------------
# Output
# -----------------------------

print("Training cases:")
print(train_id_set)

print("Validation cases:")
print(val_id_set)

print("Number of training cases:")
print(len(train_cases))

print("Number of validation cases:")
print(len(val_cases))

print("Overlapping cases:")
print(overlap)


# -----------------------------
# Controlli automatici
# -----------------------------

assert len(train_cases) == 8

assert len(val_cases) == 2

assert len(overlap) == 0