from pathlib import Path


def discover_cases(
    images_dir,
    labels_dir
):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_files = sorted(
        images_dir.glob("liver_*.nii.gz")
    )

    cases = []

    for image_path in image_files:

        case_name = image_path.name

        label_path = (
            labels_dir / case_name
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for {case_name}"
            )

        case_id = case_name.replace(
            ".nii.gz",
            ""
        )

        cases.append(
            {
                "id": case_id,
                "image": str(image_path),
                "label": str(label_path)
            }
        )

    return cases