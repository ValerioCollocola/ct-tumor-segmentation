from src.visualization.compare_windowing import compare_ct_windowing


compare_ct_windowing(
    "data/sample/image/liver_0.nii.gz",
    slice_index=58,
    center=50,
    width=150,
    output_path="outputs/liver_0_windowing_comparison.png"
)