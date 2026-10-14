from src.visualization.compare_preprocessing import compare_preprocessing


compare_preprocessing(
    image_path="data/sample/image/liver_0.nii.gz",
    slice_index=58,
    center=50,
    width=150,
    output_path="outputs/liver_0_preprocessing_comparison.png"
)