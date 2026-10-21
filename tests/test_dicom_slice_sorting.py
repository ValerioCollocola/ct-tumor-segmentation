from pydicom.dataset import Dataset

from src.data.dicom.series import (
    get_slice_position,
    sort_dicom_slices,
    compute_slice_spacing
)


def create_test_slice(
    instance_number,
    z_position
):
    dataset = Dataset()

    dataset.InstanceNumber = instance_number

    dataset.ImageOrientationPatient = [
        1, 0, 0,
        0, 1, 0
    ]

    dataset.ImagePositionPatient = [
        0,
        0,
        z_position
    ]

    dataset.SliceThickness = 5.0

    return dataset


datasets = [
    create_test_slice(
        3,
        10.0
    ),

    create_test_slice(
        1,
        0.0
    ),

    create_test_slice(
        2,
        5.0
    )
]


print()
print("Original positions:")

for dataset in datasets:

    print(
        get_slice_position(
            dataset
        )
    )


sorted_datasets = sort_dicom_slices(
    datasets
)


print()
print("Sorted positions:")

for dataset in sorted_datasets:

    print(
        get_slice_position(
            dataset
        )
    )


slice_spacing = compute_slice_spacing(
    sorted_datasets
)


print()
print("Computed slice spacing:")
print(slice_spacing)


assert [
    get_slice_position(dataset)
    for dataset in sorted_datasets
] == [
    0.0,
    5.0,
    10.0
]

assert slice_spacing == 5.0