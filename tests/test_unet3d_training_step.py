import torch

from src.models.unet3d import UNet3D
from src.models.losses import combined_loss


images = torch.randn(
    1,
    1,
    96,
    96,
    64
)

labels = torch.randint(
    low=0,
    high=3,
    size=(
        1,
        96,
        96,
        64
    ),
    dtype=torch.long
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

weights_before = {
    name: parameter.detach().clone()
    for name, parameter in model.named_parameters()
}

optimizer.zero_grad()

outputs = model(images)

loss, ce_loss, dice_loss = combined_loss(
    outputs,
    labels,
    num_classes=3
)

loss.backward()

optimizer.step()

weights_changed = False

for name, parameter in model.named_parameters():

    if not torch.equal(
        weights_before[name],
        parameter.detach()
    ):
        weights_changed = True
        break

print("Cross Entropy Loss:")
print(ce_loss.item())

print("Dice Loss:")
print(dice_loss.item())

print("Combined Loss:")
print(loss.item())

print("UNet weights changed:")
print(weights_changed)

assert torch.isfinite(loss)
assert loss.item() > 0
assert weights_changed