import torch

from src.models.losses import combined_loss
from src.training.checkpoint import save_checkpoint
from src.evaluation.metrics import dice_score_per_class


def train_and_validate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    num_epochs=3,
    device="cpu",
    checkpoint_path="outputs/checkpoints/best_model.pt",
    class_weights=None
):
    # Sposta il modello sul device scelto
    model.to(device)

    # --------------------------------------------------
    # Training history
    # --------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_tumor_dice": [],
        "tumor_dice": []
    }

    # Migliore validation loss osservata
    best_val_loss = float("inf")

    # Migliore Tumor Dice osservato
    best_tumor_dice = -1.0

    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------

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

            # Azzera i gradienti del batch precedente
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # --------------------------------------------------
            # Train Tumor Dice
            # --------------------------------------------------

            train_dice_scores = dice_score_per_class(
                outputs,
                labels,
                num_classes=3
            )

            train_tumor_dice_total += (
                train_dice_scores[2]
            )

            training_batches += 1

            # --------------------------------------------------
            # Loss
            # --------------------------------------------------

            loss, ce_loss, d_loss = combined_loss(
                outputs,
                labels,
                num_classes=3,
                class_weights=class_weights
            )

            # Backpropagation
            loss.backward()

            # Aggiornamento dei pesi
            optimizer.step()

            # Accumula training loss
            train_loss += loss.item()

        # --------------------------------------------------
        # Training averages
        # --------------------------------------------------

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

        tumor_dice_total = 0.0
        validation_batches = 0

        with torch.no_grad():

            for images, labels in val_dataloader:

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # --------------------------------------------------
                # Validation loss
                # --------------------------------------------------

                loss, ce_loss, d_loss = combined_loss(
                    outputs,
                    labels,
                    num_classes=3,
                    class_weights=class_weights
                )

                val_loss += loss.item()

                # --------------------------------------------------
                # Validation Tumor Dice
                # --------------------------------------------------

                dice_scores = dice_score_per_class(
                    outputs,
                    labels,
                    num_classes=3
                )

                tumor_dice_total += (
                    dice_scores[2]
                )

                validation_batches += 1

        # --------------------------------------------------
        # Validation averages
        # --------------------------------------------------

        average_val_loss = (
            val_loss
            / len(val_dataloader)
        )

        average_tumor_dice = (
            tumor_dice_total
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

        history["tumor_dice"].append(
            average_tumor_dice
        )

        # ==================================================
        # OUTPUT
        # ==================================================

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"- Train Loss: {average_train_loss:.4f} "
            f"- Val Loss: {average_val_loss:.4f} "
            f"- Train Tumor Dice: "
            f"{average_train_tumor_dice:.4f} "
            f"- Val Tumor Dice: "
            f"{average_tumor_dice:.4f}"
        )

        # ==================================================
        # BEST VALIDATION LOSS CHECKPOINT
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
                f"Best model saved "
                f"at epoch {epoch + 1}"
            )

        # ==================================================
        # BEST TUMOR DICE CHECKPOINT
        # ==================================================

        if average_tumor_dice > best_tumor_dice:

            best_tumor_dice = (
                average_tumor_dice
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
                f"Best tumor Dice model saved "
                f"at epoch {epoch + 1} "
                f"- Tumor Dice: "
                f"{average_tumor_dice:.4f}"
            )

    # --------------------------------------------------
    # Fine training
    # --------------------------------------------------

    return history