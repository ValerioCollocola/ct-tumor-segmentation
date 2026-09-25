import numpy as np


def calculate_tumor_volume(label, affine, tumor_label=2):
    """
    Calcola il volume tumorale a partire dalla maschera NIfTI.

    Parameters
    ----------
    label : np.ndarray
        Volume 3D contenente la maschera di segmentazione.

    affine : np.ndarray
        Matrice affine NIfTI.

    tumor_label : int
        Valore che identifica il tumore nella maschera.

    Returns
    -------
    tumor_voxels : int
        Numero totale di voxel tumorali.

    spacing_mm : np.ndarray
        Dimensione dei voxel lungo i tre assi, in millimetri.

    tumor_volume_mm3 : float
        Volume tumorale in mm³.

    tumor_volume_cm3 : float
        Volume tumorale in cm³.
    """

    # Dimensione fisica di un voxel lungo X, Y e Z
    spacing_mm = np.linalg.norm(
        affine[:3, :3],
        axis=0
    )

    # Conta tutti i voxel classificati come tumore
    tumor_voxels = np.count_nonzero(
        label == tumor_label
    )

    # Volume di un singolo voxel
    voxel_volume_mm3 = np.prod(spacing_mm)

    # Volume tumorale totale
    tumor_volume_mm3 = (
        tumor_voxels * voxel_volume_mm3
    )

    # 1 cm³ = 1000 mm³
    tumor_volume_cm3 = (
        tumor_volume_mm3 / 1000
    )

    return (
        tumor_voxels,
        spacing_mm,
        tumor_volume_mm3,
        tumor_volume_cm3
    )