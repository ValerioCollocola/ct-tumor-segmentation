import torch

from src.models.losses import binary_combined_loss
from src.training.checkpoint import save_checkpoint
from src.evaluation.metrics import dice_score_per_class


def train_and_validate_binary(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    num_epochs=10,
    device="cpu",
    checkpoint_path="outputs/checkpoints/best_binary_unet.pt",
    class_weights=None
):
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_tumor_dice": [],
        "val_tumor_dice": []
    }

    best_val_loss = float("inf")
    best_tumor_dice = -1.0

    for epoch in range(num_epochs):

        # ==================================================
        # TRAINING
        # ==================================================

        model.train()

        train_loss = 0.0
        train_tumor_dice_total = 0.0
        training_batches = 0

        for images, labels in train_dataloader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss, ce_loss, dice_loss = binary_combined_loss(
                outputs,
                labels,
                class_weights=class_weights
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            dice_scores = dice_score_per_class(
                outputs,
                labels,
                num_classes=2
            )

            # Binary:
            # 0 = non-tumor
            # 1 = tumor
            train_tumor_dice_total += dice_scores[1]

            training_batches += 1

        average_train_loss = (
            train_loss
            / len(train_dataloader)
        )

        average_train_tumor_dice = (
            train_tumor_dice_total
            / training_batches
        )

        # ==================================================
        # VALIDATION
        # ==================================================

        model.eval()

        val_loss = 0.0
        val_tumor_dice_total = 0.0
        validation_batches = 0

        with torch.no_grad():

            for images, labels in val_dataloader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss, ce_loss, dice_loss = binary_combined_loss(
                    outputs,
                    labels,
                    class_weights=class_weights
                )

                val_loss += loss.item()

                dice_scores = dice_score_per_class(
                    outputs,
                    labels,
                    num_classes=2
                )

                val_tumor_dice_total += dice_scores[1]

                validation_batches += 1

        average_val_loss = (
            val_loss
            / len(val_dataloader)
        )

        average_val_tumor_dice = (
            val_tumor_dice_total
            / validation_batches
        )

        # ==================================================
        # HISTORY
        # ==================================================

        history["train_loss"].append(
            average_train_loss
        )

        history["val_loss"].append(
            average_val_loss
        )

        history["train_tumor_dice"].append(
            average_train_tumor_dice
        )

        history["val_tumor_dice"].append(
            average_val_tumor_dice
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"- Train Loss: {average_train_loss:.4f} "
            f"- Val Loss: {average_val_loss:.4f} "
            f"- Train Tumor Dice: "
            f"{average_train_tumor_dice:.4f} "
            f"- Val Tumor Dice: "
            f"{average_val_tumor_dice:.4f}"
        )

        # ==================================================
        # BEST VALIDATION LOSS
        # ==================================================

        if average_val_loss < best_val_loss:

            best_val_loss = average_val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=average_val_loss,
                output_path=checkpoint_path
            )

            print(
                f"Best binary model saved "
                f"at epoch {epoch + 1}"
            )

        # ==================================================
        # BEST TUMOR DICE
        # ==================================================

        if average_val_tumor_dice > best_tumor_dice:

            best_tumor_dice = (
                average_val_tumor_dice
            )

            tumor_checkpoint_path = (
                checkpoint_path.replace(
                    ".pt",
                    "_best_tumor_dice.pt"
                )
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=average_val_loss,
                output_path=tumor_checkpoint_path
            )

            print(
                f"Best binary Tumor Dice model saved "
                f"at epoch {epoch + 1} "
                f"- Tumor Dice: "
                f"{average_val_tumor_dice:.4f}"
            )

    return history