import matplotlib.pyplot as plt

from src.data.io import load_nifti
from src.preprocessing.windowing import apply_ct_window
from src.preprocessing.normalization import normalize_intensity


def compare_preprocessing(
    image_path,
    slice_index,
    center,
    width,
    output_path
):
    # Carica il volume CT
    image, affine = load_nifti(image_path)

    # Applica il windowing
    windowed_image = apply_ct_window(
        image,
        center=center,
        width=width
    )

    # Calcola i limiti della finestra
    lower = center - width / 2
    upper = center + width / 2

    # Applica la normalizzazione
    normalized_image = normalize_intensity(
        windowed_image,
        min_value=lower,
        max_value=upper
    )

    # Estrae la stessa slice
    original_slice = image[:, :, slice_index]
    windowed_slice = windowed_image[:, :, slice_index]
    normalized_slice = normalized_image[:, :, slice_index]

    # Crea la figura
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    fig.suptitle(
        f"CT Preprocessing Pipeline - Slice {slice_index}",
        fontsize=16
    )

    # Originale
    axes[0].imshow(
        original_slice,
        cmap="gray"
    )
    axes[0].set_title("Original CT")
    axes[0].axis("off")

    # Windowed
    axes[1].imshow(
        windowed_slice,
        cmap="gray",
        vmin=lower,
        vmax=upper
    )
    axes[1].set_title(
        f"Windowed CT\nC:{center} W:{width}"
    )
    axes[1].axis("off")

    # Normalized
    axes[2].imshow(
        normalized_slice,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[2].set_title("Normalized CT\nRange [0, 1]")
    axes[2].axis("off")

    plt.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()