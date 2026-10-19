import matplotlib.pyplot as plt


def plot_single_case_loss(
    history,
    output_path
):
    epochs = range(
        1,
        len(history) + 1
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history,
        marker="o",
        label="Combined Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Single-Case Overfitting Test"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()