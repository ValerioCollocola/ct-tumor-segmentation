import torch
from pathlib import Path

from src.models.unet3d import UNet3D
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint
)


model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

checkpoint_path = (
    "outputs/checkpoints/"
    "test_unet3d_checkpoint.pt"
)

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=1,
    loss=1.75,
    output_path=checkpoint_path
)

print("Checkpoint exists:")
print(Path(checkpoint_path).exists())


new_model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=8
)

new_optimizer = torch.optim.Adam(
    new_model.parameters(),
    lr=1e-3
)

new_model, new_optimizer, epoch, loss = load_checkpoint(
    model=new_model,
    optimizer=new_optimizer,
    checkpoint_path=checkpoint_path,
    device="cpu"
)

print("Loaded epoch:")
print(epoch)

print("Loaded loss:")
print(loss)


assert Path(checkpoint_path).exists()
assert epoch == 1
assert loss == 1.75