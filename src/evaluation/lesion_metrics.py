import numpy as np

from scipy.ndimage import label


def lesion_wise_evaluation(
    prediction_mask,
    ground_truth_mask
):
    """
    Valuta le lesioni tumorali come componenti connesse 3D.

    Parameters
    ----------
    prediction_mask : np.ndarray
        Maschera binaria predetta.
        0 = non-tumor
        1 = tumor

    ground_truth_mask : np.ndarray
        Ground-truth binaria.
        0 = non-tumor
        1 = tumor

    Returns
    -------
    dict
        Metriche lesion-wise.
    """

    prediction_mask = (
        prediction_mask > 0
    )

    ground_truth_mask = (
        ground_truth_mask > 0
    )

    # --------------------------------------------------
    # 26-connectivity in 3D
    # --------------------------------------------------

    connectivity = np.ones(
        (3, 3, 3),
        dtype=np.uint8
    )

    # --------------------------------------------------
    # Connected components
    # --------------------------------------------------

    ground_truth_components, num_gt_lesions = label(
        ground_truth_mask,
        structure=connectivity
    )

    prediction_components, num_pred_lesions = label(
        prediction_mask,
        structure=connectivity
    )

    # --------------------------------------------------
    # Ground-truth lesions detected
    # --------------------------------------------------

    detected_gt_lesions = 0

    for lesion_id in range(
        1,
        num_gt_lesions + 1
    ):

        lesion_mask = (
            ground_truth_components
            == lesion_id
        )

        overlap = np.any(
            prediction_mask
            & lesion_mask
        )

        if overlap:
            detected_gt_lesions += 1

    # --------------------------------------------------
    # Predicted lesions with no GT overlap
    # --------------------------------------------------

    false_positive_lesions = 0

    for lesion_id in range(
        1,
        num_pred_lesions + 1
    ):

        lesion_mask = (
            prediction_components
            == lesion_id
        )

        overlap = np.any(
            ground_truth_mask
            & lesion_mask
        )

        if not overlap:
            false_positive_lesions += 1

    # --------------------------------------------------
    # Missed lesions
    # --------------------------------------------------

    missed_lesions = (
        num_gt_lesions
        - detected_gt_lesions
    )

    # --------------------------------------------------
    # Lesion sensitivity
    # --------------------------------------------------

    if num_gt_lesions == 0:
        lesion_sensitivity = 0.0

    else:
        lesion_sensitivity = (
            detected_gt_lesions
            / num_gt_lesions
        )

    return {
        "gt_lesions": num_gt_lesions,
        "pred_lesions": num_pred_lesions,
        "detected_lesions": detected_gt_lesions,
        "missed_lesions": missed_lesions,
        "false_positive_lesions": false_positive_lesions,
        "lesion_sensitivity": lesion_sensitivity
    }

def lesion_sizes(
    mask
):
    mask = (
        mask > 0
    )

    connectivity = np.ones(
        (3, 3, 3),
        dtype=np.uint8
    )

    components, num_lesions = label(
        mask,
        structure=connectivity
    )

    sizes = []

    for lesion_id in range(
        1,
        num_lesions + 1
    ):

        lesion_size = np.count_nonzero(
            components == lesion_id
        )

        sizes.append(
            lesion_size
        )

    return sizes

def remove_small_components(
    mask,
    min_size=20
):
    mask = (
        mask > 0
    )

    connectivity = np.ones(
        (3, 3, 3),
        dtype=np.uint8
    )

    components, num_lesions = label(
        mask,
        structure=connectivity
    )

    filtered_mask = np.zeros_like(
        mask,
        dtype=np.uint8
    )

    for lesion_id in range(
        1,
        num_lesions + 1
    ):

        lesion_mask = (
            components == lesion_id
        )

        lesion_size = np.count_nonzero(
            lesion_mask
        )

        if lesion_size >= min_size:

            filtered_mask[
                lesion_mask
            ] = 1

    return filtered_mask