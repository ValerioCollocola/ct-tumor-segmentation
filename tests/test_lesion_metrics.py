import numpy as np

from src.evaluation.lesion_metrics import (
    lesion_wise_evaluation
)


# --------------------------------------------------
# 1. Volume sintetico
# --------------------------------------------------

shape = (
    32,
    32,
    16
)

ground_truth = np.zeros(
    shape,
    dtype=np.uint8
)

prediction = np.zeros(
    shape,
    dtype=np.uint8
)


# --------------------------------------------------
# 2. Due lesioni reali
# --------------------------------------------------

# Lesione 1
ground_truth[
    5:10,
    5:10,
    4:7
] = 1

# Lesione 2
ground_truth[
    20:25,
    20:25,
    10:13
] = 1


# --------------------------------------------------
# 3. Prediction
# --------------------------------------------------

# Rileva correttamente la prima lesione
prediction[
    6:9,
    6:9,
    4:7
] = 1

# Falso positivo separato
prediction[
    15:18,
    5:8,
    8:11
] = 1


# --------------------------------------------------
# 4. Evaluation
# --------------------------------------------------

metrics = lesion_wise_evaluation(
    prediction_mask=prediction,
    ground_truth_mask=ground_truth
)


print("Ground-truth lesions:")
print(metrics["gt_lesions"])

print("Predicted lesions:")
print(metrics["pred_lesions"])

print("Detected lesions:")
print(metrics["detected_lesions"])

print("Missed lesions:")
print(metrics["missed_lesions"])

print("False-positive lesions:")
print(metrics["false_positive_lesions"])

print("Lesion sensitivity:")
print(metrics["lesion_sensitivity"])


# --------------------------------------------------
# 5. Controlli
# --------------------------------------------------

assert metrics["gt_lesions"] == 2
assert metrics["pred_lesions"] == 2

assert metrics["detected_lesions"] == 1
assert metrics["missed_lesions"] == 1

assert metrics["false_positive_lesions"] == 1

assert metrics["lesion_sensitivity"] == 0.5