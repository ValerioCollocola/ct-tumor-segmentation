from src.data.io import load_nifti
from src.preprocessing.windowing import apply_ct_window

image, affine = load_nifti(
    "data/sample/image/liver_0.nii.gz"
)

windowed_image = apply_ct_window(
    image,
    center=50,
    width=150
)

print("Original range:")
print(image.min(), image.max())

print("Windowed range:")
print(windowed_image.min(), windowed_image.max())