import random


def train_val_split(
    case_ids,
    val_fraction=0.2,
    seed=42
):
    case_ids = list(case_ids)

    random_generator = random.Random(seed)
    random_generator.shuffle(case_ids)

    val_size = max(
        1,
        int(len(case_ids) * val_fraction)
    )

    val_cases = case_ids[:val_size]
    train_cases = case_ids[val_size:]

    return train_cases, val_cases