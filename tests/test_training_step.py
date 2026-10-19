import torch

from src.models.simple_3d_cnn import Simple3DSegmentationCNN
from src.models.losses import combined_loss


# -----------------------------
# Batch fittizio
# -----------------------------

images = torch.randn(
    2,
    1,
    96,
    96,
    64
)

labels = torch.randint(
    low=0,
    high=3,
    size=(
        2,
        96,
        96,
        64
    ),
    dtype=torch.long
)


# -----------------------------
# Modello
# -----------------------------

model = Simple3DSegmentationCNN(
    in_channels=1,
    num_classes=3
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# -----------------------------
# Salva una copia dei pesi
# prima del training step
# -----------------------------

weights_before = {}

for name, parameter in model.named_parameters():
    weights_before[name] = parameter.detach().clone()


# -----------------------------
# Training step
# -----------------------------

optimizer.zero_grad()

outputs = model(images)

loss, ce_loss, d_loss = combined_loss(
    outputs,
    labels,
    num_classes=3
)

loss.backward()

optimizer.step()


# -----------------------------
# Verifica modifica dei pesi
# -----------------------------

weights_changed = False

for name, parameter in model.named_parameters():

    if not torch.equal(
        weights_before[name],
        parameter.detach()
    ):
        weights_changed = True
        break


# -----------------------------
# Output
# -----------------------------

print("Cross Entropy Loss:")
print(ce_loss.item())

print("Dice Loss:")
print(d_loss.item())

print("Combined Loss:")
print(loss.item())

print("Model weights changed:")
print(weights_changed)


# -----------------------------
# Controlli automatici
# -----------------------------

assert torch.isfinite(loss)

assert loss.item() > 0

assert weights_changed