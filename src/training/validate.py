import torch

from src.models.losses import combined_loss


def validate_model(
    model,
    dataloader,
    device="cpu"
):
    model.to(device)

    model.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss, ce_loss, d_loss = combined_loss(
                outputs,
                labels,
                num_classes=3
            )

            validation_loss += loss.item()

    average_validation_loss = (
        validation_loss / len(dataloader)
    )

    return average_validation_loss