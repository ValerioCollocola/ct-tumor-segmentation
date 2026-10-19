import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from src.data.io import load_nifti
from src.data.dataset import TumorSegmentationDataset

from src.preprocessing.windowing import apply_ct_window
from src.preprocessing.normalization import normalize_intensity
from src.preprocessing.resampling import resample_volume
from src.preprocessing.patches import (
    find_label_center,
    extract_3d_patch
)

from src.models.unet3d import UNet3D
from src.models.losses import combined_loss

from src.visualization.plot_single_case_loss import (
    plot_single_case_loss
)


# --------------------------------------------------
# 1. Caricamento del caso reale
# --------------------------------------------------

image, image_affine = load_nifti(
    "data/sample/image/liver_0.nii.gz"
)

label, label_affine = load_nifti(
    "data/sample/label/liver_0.nii.gz"
)


# --------------------------------------------------
# 2. Windowing
# --------------------------------------------------

windowed_image = apply_ct_window(
    image,
    center=50,
    width=150
)


# --------------------------------------------------
# 3. Normalizzazione
# --------------------------------------------------

normalized_image = normalize_intensity(
    windowed_image,
    min_value=-25,
    max_value=125
)


# --------------------------------------------------
# 4. Spacing e resampling
# --------------------------------------------------

current_spacing = np.linalg.norm(
    image_affine[:3, :3],
    axis=0
)

target_spacing = np.array([
    1.0,
    1.0,
    2.5
])

resampled_image, _ = resample_volume(
    normalized_image,
    current_spacing=current_spacing,
    target_spacing=target_spacing,
    is_label=False
)

resampled_label, _ = resample_volume(
    label,
    current_spacing=current_spacing,
    target_spacing=target_spacing,
    is_label=True
)


# --------------------------------------------------
# 5. Patch tumor-centered
# --------------------------------------------------

tumor_center = find_label_center(
    resampled_label,
    label_value=2
)

patch_size = (
    96,
    96,
    64
)

image_patch = extract_3d_patch(
    resampled_image,
    center=tumor_center,
    patch_size=patch_size
)

label_patch = extract_3d_patch(
    resampled_label,
    center=tumor_center,
    patch_size=patch_size
)


# --------------------------------------------------
# 6. Dataset -> Tensor
# --------------------------------------------------

dataset = TumorSegmentationDataset(
    image_patches=[image_patch],
    label_patches=[label_patch]
)

image_tensor, label_tensor = dataset[0]

image_tensor = image_tensor.unsqueeze(0)
label_tensor = label_tensor.unsqueeze(0)


# --------------------------------------------------
# 7. Modello
# --------------------------------------------------

model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)


# --------------------------------------------------
# 8. Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# --------------------------------------------------
# 9. Training sul singolo caso
# --------------------------------------------------

num_epochs = 10

history = []

for epoch in range(num_epochs):

    model.train()

    optimizer.zero_grad()

    outputs = model(
        image_tensor
    )

    loss, ce_loss, d_loss = combined_loss(
        outputs,
        label_tensor,
        num_classes=3
    )

    loss.backward()

    optimizer.step()

    history.append(
        loss.item()
    )

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"- Combined Loss: {loss.item():.4f} "
        f"- CE: {ce_loss.item():.4f} "
        f"- Dice: {d_loss.item():.4f}"
    )


# --------------------------------------------------
# 10. Risultati
# --------------------------------------------------

print("Initial loss:")
print(history[0])

print("Final loss:")
print(history[-1])


# --------------------------------------------------
# 11. Controlli
# --------------------------------------------------

assert len(history) == num_epochs

assert torch.isfinite(
    torch.tensor(history[-1])
)

assert history[-1] < history[0]

plot_single_case_loss(
    history,
    output_path="outputs/single_case_overfit_loss.png"
)