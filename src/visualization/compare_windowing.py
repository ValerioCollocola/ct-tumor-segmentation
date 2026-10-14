import matplotlib.pyplot as plt

from src.data.io import load_nifti
from src.preprocessing.windowing import apply_ct_window


def compare_ct_windowing(
    image_path,
    slice_index,
    center,
    width,
    output_path
):

    lower = center - width / 2
    upper = center + width / 2

    # Carica il volume CT
    image, affine = load_nifti(image_path)

    # Applica il windowing
    windowed_image = apply_ct_window(
        image,
        center=center,
        width=width
    )

    # Estrae la stessa slice dalle due versioni
    original_slice = image[:, :, slice_index]
    windowed_slice = windowed_image[:, :, slice_index]

    # Crea figura con due pannelli
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    # Crea il titolo della figura
    fig.suptitle(
    f"CT Intensity Windowing - Slice {slice_index}",
    fontsize=16
   )

    # CT originale
    axes[0].imshow(
        original_slice,
        cmap="gray"
    )
    axes[0].set_title(
        f"Original CT - Slice {slice_index}"
    )
    axes[0].axis("off")

    # CT windowed
    axes[1].imshow(
        windowed_slice,
        cmap="gray",
        vmin=lower,
        vmax=upper
    )
    axes[1].set_title(
        f"Windowed CT - C:{center} W:{width}"
    )
    axes[1].axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()