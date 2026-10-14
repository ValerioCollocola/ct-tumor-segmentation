import numpy as np

from src.visualization.compare_resampling import compare_resampling


compare_resampling(
    image_path="data/sample/image/liver_0.nii.gz",
    label_path="data/sample/label/liver_0.nii.gz",
    target_spacing=np.array([
        1.0,
        1.0,
        2.5
    ]),
    output_path="outputs/liver_0_resampling_comparison.png"
)