from src.inference.sliding_window import (
    compute_window_starts
)


volume_shape = (
    380,
    380,
    100
)

patch_size = (
    96,
    96,
    64
)

overlap = 0.5


x_starts = compute_window_starts(
    volume_shape[0],
    patch_size[0],
    overlap
)

y_starts = compute_window_starts(
    volume_shape[1],
    patch_size[1],
    overlap
)

z_starts = compute_window_starts(
    volume_shape[2],
    patch_size[2],
    overlap
)


print()
print("SLIDING WINDOW GEOMETRY")
print("=" * 60)


print()
print("X starts:")
print(x_starts)


print()
print("Y starts:")
print(y_starts)


print()
print("Z starts:")
print(z_starts)


total_patches = (
    len(x_starts)
    * len(y_starts)
    * len(z_starts)
)


print()
print("Number of X windows:")
print(len(x_starts))


print()
print("Number of Y windows:")
print(len(y_starts))


print()
print("Number of Z windows:")
print(len(z_starts))


print()
print("Total patches:")
print(total_patches)


# The last patch must reach the final voxel
assert (
    x_starts[-1]
    + patch_size[0]
    == volume_shape[0]
)

assert (
    y_starts[-1]
    + patch_size[1]
    == volume_shape[1]
)

assert (
    z_starts[-1]
    + patch_size[2]
    == volume_shape[2]
)


print()
print(
    "Sliding-window geometry test passed."
)