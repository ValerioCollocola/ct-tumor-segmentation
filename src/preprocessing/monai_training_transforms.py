import numpy as np
import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Randomizable,
    RandFlipd,
    ScaleIntensityRanged,
    Spacingd,
)


WINDOW_MIN = -25.0
WINDOW_MAX = 125.0

TARGET_SPACING = (1.0, 1.0, 2.5)

PATCH_SIZE = (96, 96, 64)

MAX_SHIFT = (12, 12, 8)


class TumorCenteredCropd(
    Randomizable,
    MapTransform,
):
    """
    Tumor-centered 3D crop designed to reproduce the
    existing custom patch-sampling strategy.

    Training:
        tumor centroid + random shift

    Validation:
        deterministic tumor centroid
    """

    def __init__(
        self,
        keys,
        label_key="label",
        spatial_size=PATCH_SIZE,
        max_shift=MAX_SHIFT,
        random_shift=True,
        allow_missing_keys=False,
    ):
        MapTransform.__init__(
            self,
            keys=keys,
            allow_missing_keys=allow_missing_keys,
        )

        Randomizable.__init__(self)

        self.label_key = label_key

        self.spatial_size = np.asarray(
            spatial_size,
            dtype=int,
        )

        self.max_shift = np.asarray(
            max_shift,
            dtype=int,
        )

        self.random_shift = random_shift

    def randomize(self):

        if self.random_shift:

            self._shift = np.asarray(
                [
                    self.R.randint(
                        -int(max_s),
                        int(max_s) + 1,
                    )
                    for max_s in self.max_shift
                ],
                dtype=int,
            )

        else:

            self._shift = np.zeros(
                3,
                dtype=int,
            )

    def __call__(self, data):

        d = dict(data)

        label = d[self.label_key]

        # MONAI format:
        # [C, X, Y, Z]
        label_np = (
            label.detach()
            .cpu()
            .numpy()
        )

        tumor_coordinates = np.argwhere(
            label_np[0] == 2
        )

        if len(tumor_coordinates) == 0:
            raise ValueError(
                "No tumor voxels found in label."
            )

        tumor_center = np.round(
            tumor_coordinates.mean(axis=0)
        ).astype(int)

        self.randomize()

        center = (
            tumor_center
            + self._shift
        )

        volume_shape = np.asarray(
            label_np.shape[1:],
            dtype=int,
        )

        half_size = (
            self.spatial_size // 2
        )

        # Clamp center so the patch always stays
        # inside the volume.
        minimum_center = half_size

        maximum_center = (
            volume_shape
            - (
                self.spatial_size
                - half_size
            )
        )

        center = np.maximum(
            center,
            minimum_center,
        )

        center = np.minimum(
            center,
            maximum_center,
        )

        start = (
            center
            - half_size
        )

        end = (
            start
            + self.spatial_size
        )

        slices = (
            slice(None),
            slice(start[0], end[0]),
            slice(start[1], end[1]),
            slice(start[2], end[2]),
        )

        for key in self.keys:

            d[key] = d[key][slices]

        d["tumor_center"] = (
            tumor_center.tolist()
        )

        d["crop_center"] = (
            center.tolist()
        )

        d["crop_shift"] = (
            self._shift.tolist()
        )

        return d


def build_monai_training_transforms(
    target_spacing=TARGET_SPACING,
    patch_size=PATCH_SIZE,
):
    """
    Training pipeline:
        CT / label load
        intensity scaling
        resampling
        tumor-centroid crop + random shift
        random 3D flips
    """

    return Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                image_only=False,
            ),

            EnsureChannelFirstd(
                keys=["image", "label"],
            ),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=WINDOW_MIN,
                a_max=WINDOW_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
                align_corners=True,
                scale_extent=True,
            ),

            TumorCenteredCropd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                max_shift=MAX_SHIFT,
                random_shift=True,
            ),

            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=0,
            ),

            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=1,
            ),

            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=2,
            ),

            EnsureTyped(
                keys=["image"],
                dtype=torch.float32,
                track_meta=True,
            ),

            EnsureTyped(
                keys=["label"],
                dtype=torch.int64,
                track_meta=True,
            ),
        ]
    )


def build_monai_validation_transforms(
    target_spacing=TARGET_SPACING,
    patch_size=PATCH_SIZE,
):
    """
    Deterministic validation pipeline.

    Validation is centered exactly on the tumor centroid
    and contains no random augmentation.
    """

    return Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                image_only=False,
            ),

            EnsureChannelFirstd(
                keys=["image", "label"],
            ),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=WINDOW_MIN,
                a_max=WINDOW_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
                align_corners=True,
                scale_extent=True,
            ),

            TumorCenteredCropd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                max_shift=(0, 0, 0),
                random_shift=False,
            ),

            EnsureTyped(
                keys=["image"],
                dtype=torch.float32,
                track_meta=True,
            ),

            EnsureTyped(
                keys=["label"],
                dtype=torch.int64,
                track_meta=True,
            ),
        ]
    )