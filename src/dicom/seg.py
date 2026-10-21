import numpy as np


def get_frame_referenced_sop_uid(frame_group):
    """
    Extract the source CT SOPInstanceUID referenced
    by a DICOM SEG frame.

    A SEG frame usually references its source image through:

    DerivationImageSequence
        -> SourceImageSequence
            -> ReferencedSOPInstanceUID
    """

    derivation_sequence = getattr(
        frame_group,
        "DerivationImageSequence",
        []
    )

    if len(derivation_sequence) == 0:
        return None

    source_image_sequence = getattr(
        derivation_sequence[0],
        "SourceImageSequence",
        []
    )

    if len(source_image_sequence) == 0:
        return None

    referenced_uid = getattr(
        source_image_sequence[0],
        "ReferencedSOPInstanceUID",
        None
    )

    if referenced_uid is None:
        return None

    return str(
        referenced_uid
    )


def reconstruct_segmentation_volume(
    seg_dataset,
    sorted_ct_datasets,
    segment_number=1
):
    """
    Reconstruct a sparse DICOM SEG object into a
    full 3D mask aligned to the referenced CT series.

    Parameters
    ----------
    seg_dataset : pydicom.dataset.Dataset
        DICOM SEG dataset.

    sorted_ct_datasets : list
        Source CT slices sorted in physical order.

    segment_number : int
        Segment number to reconstruct.

    Returns
    -------
    mask_volume : np.ndarray
        Binary mask with shape:
        [rows, columns, slices]

    matched_frames : int
        Number of SEG frames successfully mapped
        to CT slices.
    """

    if getattr(
        seg_dataset,
        "Modality",
        None
    ) != "SEG":

        raise ValueError(
            "Dataset is not a DICOM SEG object."
        )

    if len(sorted_ct_datasets) == 0:

        raise ValueError(
            "No CT datasets provided."
        )


    # --------------------------------------------------
    # Build CT SOP UID -> slice index lookup
    # --------------------------------------------------

    sop_uid_to_index = {}

    for slice_index, ct_dataset in enumerate(
        sorted_ct_datasets
    ):

        sop_uid = str(
            ct_dataset.SOPInstanceUID
        )

        sop_uid_to_index[
            sop_uid
        ] = slice_index


    # --------------------------------------------------
    # Decode SEG frames
    # --------------------------------------------------

    seg_array = seg_dataset.pixel_array

    per_frame_sequence = getattr(
        seg_dataset,
        "PerFrameFunctionalGroupsSequence",
        []
    )

    if len(per_frame_sequence) != seg_array.shape[0]:

        raise ValueError(
            "Number of SEG frames does not match "
            "PerFrameFunctionalGroupsSequence."
        )


    # --------------------------------------------------
    # Full CT-sized mask
    # --------------------------------------------------

    rows = int(
        sorted_ct_datasets[0].Rows
    )

    columns = int(
        sorted_ct_datasets[0].Columns
    )

    number_of_slices = len(
        sorted_ct_datasets
    )

    mask_volume = np.zeros(
        (
            rows,
            columns,
            number_of_slices
        ),
        dtype=np.uint8
    )


    matched_frames = 0


    # --------------------------------------------------
    # Map SEG frames to CT slices
    # --------------------------------------------------

    for frame_index, frame_group in enumerate(
        per_frame_sequence
    ):

        segment_identification = getattr(
            frame_group,
            "SegmentIdentificationSequence",
            []
        )

        if len(segment_identification) == 0:
            continue


        referenced_segment_number = int(
            segment_identification[0]
            .ReferencedSegmentNumber
        )

        if (
            referenced_segment_number
            != segment_number
        ):
            continue


        referenced_sop_uid = (
            get_frame_referenced_sop_uid(
                frame_group
            )
        )

        if referenced_sop_uid is None:
            continue


        if referenced_sop_uid not in (
            sop_uid_to_index
        ):
            continue


        ct_slice_index = (
            sop_uid_to_index[
                referenced_sop_uid
            ]
        )


        frame_mask = (
            seg_array[
                frame_index
            ] > 0
        ).astype(
            np.uint8
        )


        mask_volume[
            :,
            :,
            ct_slice_index
        ] = np.maximum(
            mask_volume[
                :,
                :,
                ct_slice_index
            ],
            frame_mask
        )


        matched_frames += 1


    return (
        mask_volume,
        matched_frames
    )