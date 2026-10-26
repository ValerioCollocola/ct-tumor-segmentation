import time

import numpy as np

from src.data.monai_dataset import (
    build_case_list,
    build_monai_dataset,
    build_monai_cache_dataset,
)


IMAGES_DIR = "data/multicase/images"
LABELS_DIR = "data/multicase/labels"

CASE_IDS = [
    "liver_0",
    "liver_1",
    "liver_2",
    "liver_3",
    "liver_4",
    "liver_5",
    "liver_6",
    "liver_7",
    "liver_8",
    "liver_9",
]


def inspect_sample(sample, name):

    image = (
        sample["image"]
        .detach()
        .cpu()
        .numpy()
    )

    label = (
        sample["label"]
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print(name)
    print("-" * 60)

    print("case_id:")
    print(sample["case_id"])

    print()

    print("image shape:")
    print(image.shape)

    print()

    print("label shape:")
    print(label.shape)

    print()

    print("image range:")
    print(
        float(image.min()),
        float(image.max()),
    )

    print()

    print("label values:")
    print(np.unique(label))

    assert image.shape == label.shape

    assert image.shape[0] == 1

    assert image.min() >= -1e-6
    assert image.max() <= 1.0 + 1e-6

    assert set(
        np.unique(label).tolist()
    ).issubset(
        {0, 1, 2}
    )


def main():

    print("=" * 70)
    print("MONAI DATASET / CACHE DATASET TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Build case list
    # --------------------------------------------------

    cases = build_case_list(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        case_ids=CASE_IDS,
    )

    print()
    print("Number of cases:")
    print(len(cases))

    print()
    print("First case:")
    print(cases[0])

    assert len(cases) == 10

    # --------------------------------------------------
    # Standard Dataset
    # --------------------------------------------------

    dataset = build_monai_dataset(
        cases=cases,
    )

    print()
    print("Dataset length:")
    print(len(dataset))

    start = time.perf_counter()

    sample_dataset = dataset[0]

    dataset_time = (
        time.perf_counter()
        - start
    )

    inspect_sample(
        sample_dataset,
        "STANDARD DATASET SAMPLE",
    )

    print()
    print(
        "Standard Dataset first access time:"
    )
    print(dataset_time)

    # --------------------------------------------------
    # CacheDataset
    # --------------------------------------------------

    start = time.perf_counter()

    cache_dataset = (
        build_monai_cache_dataset(
            cases=cases,
            cache_rate=1.0,
            num_workers=0,
        )
    )

    cache_build_time = (
        time.perf_counter()
        - start
    )

    print()
    print("CacheDataset length:")
    print(len(cache_dataset))

    print()
    print("Cache construction time:")
    print(cache_build_time)

    # First cached access
    start = time.perf_counter()

    sample_cache_1 = cache_dataset[0]

    cache_access_1 = (
        time.perf_counter()
        - start
    )

    inspect_sample(
        sample_cache_1,
        "CACHE DATASET SAMPLE",
    )

    # Second cached access
    start = time.perf_counter()

    sample_cache_2 = cache_dataset[0]

    cache_access_2 = (
        time.perf_counter()
        - start
    )

    print()
    print("Cache first access time:")
    print(cache_access_1)

    print()
    print("Cache second access time:")
    print(cache_access_2)

    # --------------------------------------------------
    # Equivalence check
    # --------------------------------------------------

    dataset_image = (
        sample_dataset["image"]
        .detach()
        .cpu()
        .numpy()
    )

    cache_image = (
        sample_cache_1["image"]
        .detach()
        .cpu()
        .numpy()
    )

    dataset_label = (
        sample_dataset["label"]
        .detach()
        .cpu()
        .numpy()
    )

    cache_label = (
        sample_cache_1["label"]
        .detach()
        .cpu()
        .numpy()
    )

    image_mae = float(
        np.mean(
            np.abs(
                dataset_image
                - cache_image
            )
        )
    )

    label_agreement = float(
        np.mean(
            dataset_label
            == cache_label
        )
    )

    print()
    print("Dataset vs Cache image MAE:")
    print(image_mae)

    print()
    print("Dataset vs Cache label agreement:")
    print(label_agreement)

    assert image_mae < 1e-6
    assert label_agreement == 1.0

    print()
    print(
        "MONAI Dataset / CacheDataset test passed."
    )


if __name__ == "__main__":
    main()