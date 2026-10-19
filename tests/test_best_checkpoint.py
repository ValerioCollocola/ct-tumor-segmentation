import torch

from src.models.simple_3d_cnn import Simple3DSegmentationCNN
from src.training.checkpoint import load_checkpoint


checkpoint_path = (
    "outputs/checkpoints/"
    "best_test_model.pt"
)


# -----------------------------
# Ricrea il modello
# -----------------------------

model = Simple3DSegmentationCNN(
    in_channels=1,
    num_classes=3
)


# -----------------------------
# Ricrea l'optimizer
# -----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# -----------------------------
# Carica il checkpoint
# -----------------------------

model, optimizer, epoch, loss = load_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_path=checkpoint_path,
    device="cpu"
)


# -----------------------------
# Output
# -----------------------------

print("Loaded best epoch:")
print(epoch)

print("Loaded best validation loss:")
print(loss)


# -----------------------------
# Controlli
# -----------------------------

assert epoch >= 1
assert loss > 0