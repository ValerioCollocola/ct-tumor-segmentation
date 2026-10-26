from pathlib import Path

from monai.data import Dataset, CacheDataset

from src.preprocessing.monai_transforms import (
    build_monai_preprocessing,
)


def build_case_list(
    images_dir,
    labels_dir,
    case_ids,
):
    """
    Build a MONAI-compatible list of case dictionaries.

    Each case contains:
        - image path
        - label path
        - case_id
    """

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    cases = []

    for case_id in case_ids:

        image_path = (
            images_dir
            / f"{case_id}.nii.gz"
        )

        label_path = (
            labels_dir
            / f"{case_id}.nii.gz"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label: {label_path}"
            )

        cases.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "case_id": case_id,
            }
        )

    return cases


def build_monai_dataset(
    cases,
    target_spacing=(1.0, 1.0, 2.5),
):
    """
    Standard MONAI Dataset.

    Transforms are applied every time
    a sample is requested.
    """

    transforms = build_monai_preprocessing(
        target_spacing=target_spacing,
    )

    return Dataset(
        data=cases,
        transform=transforms,
    )


def build_monai_cache_dataset(
    cases,
    target_spacing=(1.0, 1.0, 2.5),
    cache_rate=1.0,
    num_workers=0,
):
    """
    MONAI CacheDataset.

    Deterministic transforms are cached in memory,
    reducing repeated preprocessing cost.
    """

    transforms = build_monai_preprocessing(
        target_spacing=target_spacing,
    )

    return CacheDataset(
        data=cases,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )