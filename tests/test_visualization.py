from src.visualization.plot_slice import plot_slice_with_mask

plot_slice_with_mask(
    "data/sample/image/liver_0.nii.gz",
    "data/sample/label/liver_0.nii.gz",
    "outputs/liver_0_tumor_overlay.png"
)