import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.data.io import load_nifti


def plot_slice_with_mask(image_path, label_path, output_path):

    # Carica CT e label
    image, image_affine = load_nifti(image_path)
    label, label_affine = load_nifti(label_path)

    # Calcola il numero di voxel tumorali per ogni slice
    tumor_pixels_per_slice = np.sum(label == 2, axis=(0, 1))

    # Trova la slice con il maggior numero di voxel tumorali
    max_tumor_slice_index = np.argmax(tumor_pixels_per_slice)

    print(
        f"Slice con maggiore quantità di tumore: "
        f"{max_tumor_slice_index}"
    )

    print(
        f"Voxel tumorali nella slice: "
        f"{tumor_pixels_per_slice[max_tumor_slice_index]}"
    )

    # Estrae la slice corrispondente dalla CT e dalla label
    image_slice = image[:, :, max_tumor_slice_index]
    label_slice = label[:, :, max_tumor_slice_index]

    # Crea la maschera del solo tumore
    tumor_mask = label_slice == 2

    # Nasconde tutto ciò che non appartiene al tumore
    tumor_mask_overlay = np.ma.masked_where(
        ~tumor_mask,
        tumor_mask
    )

    # Crea una figura con 3 pannelli
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # 1. CT originale
    axes[0].imshow(
        image_slice,
        cmap="gray"
    )

    axes[0].set_title(
        f"CT Slice {max_tumor_slice_index}"
    )

    axes[0].axis("off")

    # 2. Maschera tumorale
    axes[1].imshow(
        tumor_mask,
        cmap="Reds"
    )

    axes[1].set_title(
        f"Tumor Mask - Slice {max_tumor_slice_index}"
    )

    axes[1].axis("off")

    # 3. CT + overlay del tumore
    axes[2].imshow(
        image_slice,
        cmap="gray"
    )

    axes[2].imshow(
        tumor_mask_overlay,
        cmap="Reds",
        alpha=0.5
    )

    axes[2].set_title(
        f"CT + Tumor Overlay - Slice {max_tumor_slice_index}"
    )

    axes[2].axis("off")

    # Titolo generale
    fig.suptitle(
        f"Liver Tumor Visualization - "
        f"Slice {max_tumor_slice_index} - "
        f"{tumor_pixels_per_slice[max_tumor_slice_index]} tumor voxels",
        fontsize=16
    )

    plt.tight_layout()

    # Crea automaticamente la cartella di output
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salva la figura
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    # Mostra la figura
    plt.show()

    # Libera la figura dalla memoria
    plt.close()
   