import numpy as np

from pydicom.dataset import (
    Dataset,
    FileMetaDataset
)

from pydicom.uid import (
    ExplicitVRLittleEndian
)

from src.data.dicom.reconstruction import (
    reconstruct_ct_volume
)


def create_test_ct_slice(
    instance_number,
    z_position,
    stored_value
):
    dataset = Dataset()

    # --------------------------------------------------
    # Basic DICOM metadata
    # --------------------------------------------------

    dataset.Modality = "CT"

    dataset.InstanceNumber = (
        instance_number
    )

    dataset.Rows = 4
    dataset.Columns = 4

    dataset.PixelSpacing = [
        0.7,
        0.7
    ]

    dataset.SliceThickness = 5.0

    dataset.ImageOrientationPatient = [
        1, 0, 0,
        0, 1, 0
    ]

    dataset.ImagePositionPatient = [
        0,
        0,
        z_position
    ]


    # --------------------------------------------------
    # CT intensity metadata
    # --------------------------------------------------

    dataset.RescaleSlope = 1

    dataset.RescaleIntercept = -1024


    # --------------------------------------------------
    # Pixel metadata
    # --------------------------------------------------

    dataset.SamplesPerPixel = 1

    dataset.PhotometricInterpretation = (
        "MONOCHROME2"
    )

    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15

    dataset.PixelRepresentation = 0


    # --------------------------------------------------
    # Transfer syntax
    # --------------------------------------------------

    file_meta = FileMetaDataset()

    file_meta.TransferSyntaxUID = (
        ExplicitVRLittleEndian
    )

    dataset.file_meta = file_meta


    # --------------------------------------------------
    # Pixel data
    # --------------------------------------------------

    pixel_array = np.full(
        (4, 4),
        stored_value,
        dtype=np.uint16
    )

    dataset.PixelData = (
        pixel_array.tobytes()
    )

    return dataset


datasets = [
    create_test_ct_slice(
        instance_number=3,
        z_position=10.0,
        stored_value=1200
    ),

    create_test_ct_slice(
        instance_number=1,
        z_position=0.0,
        stored_value=1000
    ),

    create_test_ct_slice(
        instance_number=2,
        z_position=5.0,
        stored_value=1100
    )
]


volume_hu, spacing, sorted_datasets = (
    reconstruct_ct_volume(
        datasets
    )
)


print()
print("Reconstructed CT volume")
print("=" * 60)


print()
print("Volume shape:")
print(volume_hu.shape)


print()
print("Voxel spacing:")
print(spacing)


print()
print("Slice HU values:")

for slice_index in range(
    volume_hu.shape[-1]
):

    print(
        f"Slice {slice_index}:",
        volume_hu[
            0,
            0,
            slice_index
        ]
    )


print()
print("Sorted z positions:")

for dataset in sorted_datasets:

    print(
        dataset.ImagePositionPatient[2]
    )


# --------------------------------------------------
# Assertions
# --------------------------------------------------

assert volume_hu.shape == (
    4,
    4,
    3
)

assert spacing == (
    0.7,
    0.7,
    5.0
)

assert volume_hu[
    0,
    0,
    0
] == -24

assert volume_hu[
    0,
    0,
    1
] == 76

assert volume_hu[
    0,
    0,
    2
] == 176