import torch

from src.data.cases import discover_cases
from src.data.split import train_val_split
from src.data.dataset import TumorSegmentationDataset

from src.preprocessing.pipeline import preprocess_case

from src.models.unet3d import UNet3D
from src.training.checkpoint import load_checkpoint

from src.visualization.plot_prediction import (
    plot_segmentation_prediction
)


# --------------------------------------------------
# 1. Scoperta casi
# --------------------------------------------------

cases = discover_cases(
    images_dir="data/multicase/images",
    labels_dir="data/multicase/labels"
)

case_ids = [
    case["id"]
    for case in cases
]


# --------------------------------------------------
# 2. Ricrea lo stesso train/validation split
# --------------------------------------------------

train_ids, val_ids = train_val_split(
    case_ids,
    val_fraction=0.2,
    seed=42
)

val_cases = [
    case
    for case in cases
    if case["id"] in val_ids
]


print("Validation cases:")
print(val_ids)


# --------------------------------------------------
# 3. Seleziona un caso di validation
# --------------------------------------------------

case = val_cases[1]

print("Visualizing case:")
print(case["id"])


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

result = preprocess_case(
    case
)

image_patch = result["image_patch"]
label_patch = result["label_patch"]


# --------------------------------------------------
# 5. Dataset -> Tensor
# --------------------------------------------------

dataset = TumorSegmentationDataset(
    image_patches=[
        image_patch
    ],
    label_patches=[
        label_patch
    ]
)

image_tensor, label_tensor = dataset[0]

# Aggiunge batch dimension
image_tensor = image_tensor.unsqueeze(0)

label_tensor = label_tensor.unsqueeze(0)


# --------------------------------------------------
# 6. Ricrea la U-Net
# --------------------------------------------------

model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# --------------------------------------------------
# 7. Carica il best checkpoint reale
# --------------------------------------------------

model, optimizer, epoch, loss = load_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_path=(
        "outputs/checkpoints/"
        "best_real_multicase_unet.pt"
    ),
    device="cpu"
)

print("Loaded checkpoint epoch:")
print(epoch)

print("Loaded validation loss:")
print(loss)


# --------------------------------------------------
# 8. Inference
# --------------------------------------------------

model.eval()

with torch.no_grad():

    outputs = model(
        image_tensor
    )


# --------------------------------------------------
# 9. Visualizzazione
# --------------------------------------------------

plot_segmentation_prediction(
    image=image_tensor,
    label=label_tensor,
    outputs=outputs,
    output_path=(
    "outputs/"
    f"{case['id']}_prediction_comparison.png"
    )
)