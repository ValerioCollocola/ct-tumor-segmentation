import numpy as np

from src.preprocessing.patches import extract_3d_patch


# Volume 3D fittizio
volume = np.zeros(
    (360, 360, 150)
)

patch_size = (
    96,
    96,
    64
)


# -----------------------------
# Test 1 - Centro valido
# -----------------------------

valid_center = (
    180,
    180,
    75
)

patch = extract_3d_patch(
    volume,
    center=valid_center,
    patch_size=patch_size
)

print("Test 1 - Valid center")
print("Patch shape:")
print(patch.shape)

assert patch.shape == patch_size


# -----------------------------
# Test 2 - Centro vicino al bordo
# -----------------------------

border_center = (
    20,
    20,
    20
)

border_patch = extract_3d_patch(
    volume,
    center=border_center,
    patch_size=patch_size
)

print("Test 2 - Border center")
print("Patch shape:")
print(border_patch.shape)

assert border_patch.shape == patch_size


# -----------------------------
# Test 3 - Patch troppo grande
# -----------------------------

invalid_patch_size = (
    500,
    500,
    200
)

try:
    extract_3d_patch(
        volume,
        center=valid_center,
        patch_size=invalid_patch_size
    )

    raise AssertionError(
        "Expected ValueError was not raised."
    )

except ValueError as error:
    print("Test 3 - Expected error:")
    print(error)