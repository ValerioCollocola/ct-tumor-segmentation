from pydicom import examples

from src.data.dicom.io import (
    load_dicom_file,
    extract_basic_metadata
)


dicom_path = examples.get_path(
    "ct"
)

dataset = load_dicom_file(
    dicom_path
)

metadata = extract_basic_metadata(
    dataset
)


print()
print("pydicom example CT")
print("=" * 60)

print()
print("File:")
print(dicom_path)

print()
print("Metadata:")

for key, value in metadata.items():
    print(
        f"{key}: {value}"
    )


print()
print("Pixel array:")

pixel_array = dataset.pixel_array

print("Shape:")
print(pixel_array.shape)

print("Data type:")
print(pixel_array.dtype)

print("Minimum stored value:")
print(pixel_array.min())

print("Maximum stored value:")
print(pixel_array.max())