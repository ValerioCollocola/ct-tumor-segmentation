from src.data.io import load_nifti
from src.analysis.tumor_metrics import calculate_tumor_volume


label, affine = load_nifti(
    "data/sample/label/liver_0.nii.gz"
)

(
    tumor_voxels,
    spacing_mm,
    tumor_volume_mm3,
    tumor_volume_cm3
) = calculate_tumor_volume(
    label,
    affine
)

print("Voxel spacing (mm):", spacing_mm)
print("Voxel tumorali:", tumor_voxels)
print(
    f"Volume tumorale: "
    f"{tumor_volume_mm3:.2f} mm³"
)
print(
    f"Volume tumorale: "
    f"{tumor_volume_cm3:.2f} cm³"
)