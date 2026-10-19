from src.preprocessing.patches import (
    generate_shifted_centers
)


tumor_center = (
    222,
    168,
    123
)

centers = generate_shifted_centers(
    center=tumor_center,
    num_patches=5,
    max_shift=(
        12,
        12,
        8
    ),
    seed=42
)

print("Generated centers:")

for center in centers:
    print(center)

print("Number of centers:")
print(len(centers))

assert len(centers) == 5
assert centers[0] == tumor_center