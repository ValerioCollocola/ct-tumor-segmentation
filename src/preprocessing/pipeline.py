import numpy as np

from src.data.io import load_nifti
from src.preprocessing.windowing import apply_ct_window
from src.preprocessing.normalization import normalize_intensity
from src.preprocessing.resampling import resample_volume
from src.preprocessing.patches import (
    find_label_center,
    extract_3d_patch,
    generate_shifted_centers
)
from src.preprocessing.patches import generate_shifted_centers

def preprocess_case(
    case,
    target_spacing=(1.0, 1.0, 2.5),
    patch_size=(96, 96, 64),
    window_center=50,
    window_width=150,
    tumor_label=2
):
    # -----------------------------
    # Load CT and label
    # -----------------------------

    image, image_affine = load_nifti(
        case["image"]
    )

    label, label_affine = load_nifti(
        case["label"]
    )

    # -----------------------------
    # Windowing
    # -----------------------------

    windowed_image = apply_ct_window(
        image,
        center=window_center,
        width=window_width
    )

    lower = (
        window_center
        -
        window_width / 2
    )

    upper = (
        window_center
        +
        window_width / 2
    )

    # -----------------------------
    # Normalization
    # -----------------------------

    normalized_image = normalize_intensity(
        windowed_image,
        min_value=lower,
        max_value=upper
    )

    # -----------------------------
    # Current voxel spacing
    # -----------------------------

    current_spacing = np.linalg.norm(
        image_affine[:3, :3],
        axis=0
    )

    target_spacing = np.asarray(
        target_spacing,
        dtype=float
    )

    # -----------------------------
    # Resampling
    # -----------------------------

    resampled_image, _ = resample_volume(
        normalized_image,
        current_spacing=current_spacing,
        target_spacing=target_spacing,
        is_label=False
    )

    resampled_label, _ = resample_volume(
        label,
        current_spacing=current_spacing,
        target_spacing=target_spacing,
        is_label=True
    )

    # -----------------------------
    # Tumor center
    # -----------------------------

    tumor_center = find_label_center(
        resampled_label,
        label_value=tumor_label
    )

    # -----------------------------
    # Extract corresponding patches
    # -----------------------------

    image_patch = extract_3d_patch(
        resampled_image,
        center=tumor_center,
        patch_size=patch_size
    )

    label_patch = extract_3d_patch(
        resampled_label,
        center=tumor_center,
        patch_size=patch_size
    )

    return {
        "id": case["id"],
        "image_patch": image_patch,
        "label_patch": label_patch,
        "tumor_center": tumor_center,
        "original_shape": image.shape,
        "resampled_shape": resampled_image.shape,
        "original_spacing": current_spacing,
        "target_spacing": target_spacing
    }

def preprocess_case_multiple_patches(
    case,
    num_patches=5,
    target_spacing=(1.0, 1.0, 2.5),
    patch_size=(96, 96, 64),
    window_center=50,
    window_width=150,
    tumor_label=2,
    seed=42
):
    image, image_affine = load_nifti(
        case["image"]
    )

    label, label_affine = load_nifti(
        case["label"]
    )

    windowed_image = apply_ct_window(
        image,
        center=window_center,
        width=window_width
    )

    lower = (
        window_center
        -
        window_width / 2
    )

    upper = (
        window_center
        +
        window_width / 2
    )

    normalized_image = normalize_intensity(
        windowed_image,
        min_value=lower,
        max_value=upper
    )

    current_spacing = np.linalg.norm(
        image_affine[:3, :3],
        axis=0
    )

    target_spacing = np.asarray(
        target_spacing,
        dtype=float
    )

    resampled_image, _ = resample_volume(
        normalized_image,
        current_spacing=current_spacing,
        target_spacing=target_spacing,
        is_label=False
    )

    resampled_label, _ = resample_volume(
        label,
        current_spacing=current_spacing,
        target_spacing=target_spacing,
        is_label=True
    )

    tumor_center = find_label_center(
        resampled_label,
        label_value=tumor_label
    )

    centers = generate_shifted_centers(
        center=tumor_center,
        num_patches=num_patches,
        max_shift=(12, 12, 8),
        seed=seed
    )

    image_patches = []
    label_patches = []

    for center in centers:

        image_patch = extract_3d_patch(
            resampled_image,
            center=center,
            patch_size=patch_size
        )

        label_patch = extract_3d_patch(
            resampled_label,
            center=center,
            patch_size=patch_size
        )

        image_patches.append(
            image_patch
        )

        label_patches.append(
            label_patch
        )

    return {
        "id": case["id"],
        "image_patches": image_patches,
        "label_patches": label_patches,
        "centers": centers
    }