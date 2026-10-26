import numpy as np
import torch

from src.inference.monai_sliding_window import (
    monai_sliding_window_logits,
)


VOLUME_SHAPE = (
    380,
    380,
    100,
)

ROI_SIZE = (
    96,
    96,
    64,
)


class IdentitySegmentationModel(
    torch.nn.Module,
):
    """
    Minimal model used only to test sliding-window
    reconstruction.

    Output channel 0 = 0
    Output channel 1 = input image

    This allows us to verify that MONAI stitches
    the volume without changing its geometry.
    """

    def forward(
        self,
        x,
    ):

        background = torch.zeros_like(
            x
        )

        foreground = x

        return torch.cat(
            [
                background,
                foreground,
            ],
            dim=1,
        )


def main():

    print("=" * 70)
    print("MONAI SLIDING WINDOW TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Synthetic 3D volume
    # --------------------------------------------------

    rng = np.random.default_rng(
        seed=42
    )

    volume = rng.random(
        VOLUME_SHAPE,
        dtype=np.float32,
    )

    print()
    print("Input shape:")
    print(volume.shape)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = (
        IdentitySegmentationModel()
    )

    # --------------------------------------------------
    # MONAI sliding-window inference
    # --------------------------------------------------

    logits = (
        monai_sliding_window_logits(
            model=model,
            volume=volume,
            roi_size=ROI_SIZE,
            overlap=0.5,
            sw_batch_size=1,
            device="cpu",
        )
    )

    output = (
        logits
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print("Output shape:")
    print(output.shape)

    # --------------------------------------------------
    # Shape test
    # --------------------------------------------------

    expected_shape = (
        1,
        2,
        *VOLUME_SHAPE,
    )

    assert output.shape == (
        expected_shape
    )

    # --------------------------------------------------
    # Reconstruction test
    # --------------------------------------------------

    reconstructed = output[
        0,
        1,
    ]

    mae = float(
        np.mean(
            np.abs(
                reconstructed
                - volume
            )
        )
    )

    max_difference = float(
        np.max(
            np.abs(
                reconstructed
                - volume
            )
        )
    )

    print()
    print("Reconstruction MAE:")
    print(mae)

    print()

    print("Maximum difference:")
    print(max_difference)

    assert mae < 1e-6

    print()
    print(
        "MONAI sliding-window "
        "reconstruction test passed."
    )


if __name__ == "__main__":
    main()