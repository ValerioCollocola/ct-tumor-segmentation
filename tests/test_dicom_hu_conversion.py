from pydicom import examples

from src.dicom.io import (
    load_dicom_file
)

from  src.dicom.intensity import (
    dicom_pixels_to_hu
)


dicom_path = examples.get_path(
    "ct"
)

dataset = load_dicom_file(
    dicom_path
)


stored_pixels = dataset.pixel_array

hu_pixels = dicom_pixels_to_hu(
    dataset
)


print()
print("Stored pixel values")
print("=" * 50)

print("Minimum:")
print(stored_pixels.min())

print("Maximum:")
print(stored_pixels.max())


print()
print("Hounsfield Units")
print("=" * 50)

print("Minimum HU:")
print(hu_pixels.min())

print("Maximum HU:")
print(hu_pixels.max())

print("Data type:")
print(hu_pixels.dtype)


print()
print("Rescale parameters")
print("=" * 50)

print("Slope:")
print(dataset.RescaleSlope)

print("Intercept:")
print(dataset.RescaleIntercept)


assert hu_pixels.shape == stored_pixels.shape