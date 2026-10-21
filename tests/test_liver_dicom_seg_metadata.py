from pathlib import Path

import pydicom


seg_directory = Path(
    "data/dicom/liver_seg"
)


# --------------------------------------------------
# 1. Discover DICOM SEG objects
# --------------------------------------------------

seg_datasets = []


for file_path in seg_directory.rglob("*"):

    if not file_path.is_file():
        continue

    try:

        dataset = pydicom.dcmread(
            file_path
        )

    except Exception:
        continue

    modality = getattr(
        dataset,
        "Modality",
        None
    )

    if modality == "SEG":

        seg_datasets.append(
            (
                file_path,
                dataset
            )
        )


print()
print("DICOM SEG OBJECTS FOUND")
print("=" * 70)

print()
print("Count:")
print(
    len(seg_datasets)
)


if len(seg_datasets) == 0:

    raise RuntimeError(
        "No DICOM SEG object found."
    )


# --------------------------------------------------
# 2. Inspect first SEG object
# --------------------------------------------------

seg_path, seg_dataset = (
    seg_datasets[0]
)


print()
print("SEG file:")
print(
    seg_path
)


print()
print("Basic metadata")
print("-" * 70)


print("Modality:")
print(
    getattr(
        seg_dataset,
        "Modality",
        None
    )
)


print()
print("SOP Class UID:")
print(
    getattr(
        seg_dataset,
        "SOPClassUID",
        None
    )
)


print()
print("SOP Instance UID:")
print(
    getattr(
        seg_dataset,
        "SOPInstanceUID",
        None
    )
)


print()
print("Series Instance UID:")
print(
    getattr(
        seg_dataset,
        "SeriesInstanceUID",
        None
    )
)


print()
print("Series Description:")
print(
    getattr(
        seg_dataset,
        "SeriesDescription",
        None
    )
)


print()
print("Number of frames:")
print(
    getattr(
        seg_dataset,
        "NumberOfFrames",
        None
    )
)


print()
print("Rows:")
print(
    getattr(
        seg_dataset,
        "Rows",
        None
    )
)


print()
print("Columns:")
print(
    getattr(
        seg_dataset,
        "Columns",
        None
    )
)


# --------------------------------------------------
# 3. SegmentSequence
# --------------------------------------------------

print()
print("SEGMENT SEQUENCE")
print("=" * 70)


segment_sequence = getattr(
    seg_dataset,
    "SegmentSequence",
    []
)


print()
print("Number of segments:")
print(
    len(segment_sequence)
)


for segment in segment_sequence:

    segment_number = getattr(
        segment,
        "SegmentNumber",
        None
    )

    segment_label = getattr(
        segment,
        "SegmentLabel",
        None
    )

    segment_description = getattr(
        segment,
        "SegmentDescription",
        None
    )

    algorithm_type = getattr(
        segment,
        "SegmentAlgorithmType",
        None
    )

    algorithm_name = getattr(
        segment,
        "SegmentAlgorithmName",
        None
    )

    print()
    print(
        f"Segment number: {segment_number}"
    )

    print(
        f"Segment label: {segment_label}"
    )

    print(
        f"Segment description: "
        f"{segment_description}"
    )

    print(
        f"Algorithm type: "
        f"{algorithm_type}"
    )

    print(
        f"Algorithm name: "
        f"{algorithm_name}"
    )


# --------------------------------------------------
# 4. Referenced Series
# --------------------------------------------------

print()
print("REFERENCED SERIES")
print("=" * 70)


referenced_series_sequence = getattr(
    seg_dataset,
    "ReferencedSeriesSequence",
    []
)


print()
print(
    "Number of directly referenced series:"
)
print(
    len(referenced_series_sequence)
)


for referenced_series in (
    referenced_series_sequence
):

    referenced_series_uid = getattr(
        referenced_series,
        "SeriesInstanceUID",
        None
    )

    print()
    print(
        "Referenced SeriesInstanceUID:"
    )

    print(
        referenced_series_uid
    )

    referenced_instances = getattr(
        referenced_series,
        "ReferencedInstanceSequence",
        []
    )

    print(
        "Referenced instances:"
    )

    print(
        len(referenced_instances)
    )


# --------------------------------------------------
# 5. Per-frame geometry
# --------------------------------------------------

print()
print("PER-FRAME INFORMATION")
print("=" * 70)


per_frame_sequence = getattr(
    seg_dataset,
    "PerFrameFunctionalGroupsSequence",
    []
)


print()
print("Frames with functional groups:")
print(
    len(per_frame_sequence)
)


if len(per_frame_sequence) > 0:

    first_frame = (
        per_frame_sequence[0]
    )

    segment_identification = getattr(
        first_frame,
        "SegmentIdentificationSequence",
        []
    )

    if len(
        segment_identification
    ) > 0:

        referenced_segment_number = getattr(
            segment_identification[0],
            "ReferencedSegmentNumber",
            None
        )

        print()
        print(
            "First frame referenced "
            "segment number:"
        )

        print(
            referenced_segment_number
        )


    plane_position = getattr(
        first_frame,
        "PlanePositionSequence",
        []
    )

    if len(
        plane_position
    ) > 0:

        image_position = getattr(
            plane_position[0],
            "ImagePositionPatient",
            None
        )

        print()
        print(
            "First frame "
            "ImagePositionPatient:"
        )

        print(
            image_position
        )


# --------------------------------------------------
# 6. Pixel data check
# --------------------------------------------------

print()
print("SEGMENTATION PIXEL DATA")
print("=" * 70)


segmentation_array = (
    seg_dataset.pixel_array
)


print()
print("SEG pixel array shape:")
print(
    segmentation_array.shape
)


print()
print("SEG pixel array dtype:")
print(
    segmentation_array.dtype
)


print()
print("Minimum value:")
print(
    segmentation_array.min()
)


print()
print("Maximum value:")
print(
    segmentation_array.max()
)


# --------------------------------------------------
# 7. Assertions
# --------------------------------------------------

assert getattr(
    seg_dataset,
    "Modality",
    None
) == "SEG"

assert len(
    segment_sequence
) >= 1

assert segmentation_array.ndim >= 2


print()
print(
    "DICOM SEG metadata exploration passed."
)