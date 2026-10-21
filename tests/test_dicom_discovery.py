from src.data.dicom.io import (
    discover_dicom_files
)


dicom_directory = (
    "data/dicom"
)


files = discover_dicom_files(
    dicom_directory
)


print()
print("DICOM files discovered:")
print(len(files))


for file_path in files[:10]:

    print(
        file_path
    )