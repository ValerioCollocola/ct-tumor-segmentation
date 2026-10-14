from src.data.split import train_val_split


case_ids = [
    "liver_0",
    "liver_1",
    "liver_2",
    "liver_3",
    "liver_4",
    "liver_5",
    "liver_6",
    "liver_7",
    "liver_8",
    "liver_9"
]


train_cases, val_cases = train_val_split(
    case_ids,
    val_fraction=0.2,
    seed=42
)


print("Training cases:")
print(train_cases)

print("Validation cases:")
print(val_cases)

print("Number of training cases:")
print(len(train_cases))

print("Number of validation cases:")
print(len(val_cases))

overlap = set(train_cases) & set(val_cases)

print("Overlapping cases:")
print(overlap)

assert len(overlap) == 0