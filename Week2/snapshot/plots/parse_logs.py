import os
import re
from typing import Dict, List
import matplotlib.pyplot as plt


PLOTS_DIR = "/ghome/group07/week2/plots"
LOG_DIR = PLOTS_DIR + "/logs"
TITLES = {
    97248: "Simple (300 h.u.)",
    97263: "Simple (600 h.u.)",
    97272: "Deeper (300 h.u.)",
    97282: "Deeper (600 h.u.)",
    97287: "Pyramidal MLP",
    97291: "Bottleneck MLP",
}
FILES_TO_PARSE = [97282, 97287, 97291]

EPOCH_REGEX = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/\d+\s+-\s+"
    r"Train Loss:\s+(?P<train_loss>[\d.]+),\s+"
    r"Train Accuracy:\s+(?P<train_acc>[\d.]+),\s+"
    r"Test Loss:\s+(?P<test_loss>[\d.]+),\s+"
    r"Test Accuracy:\s+(?P<test_acc>[\d.]+)"
)


def extract_job_id(filename: str) -> int | None:
    return int(filename.split(".")[0].split("_")[-1])


def parse_log_file(filepath: str) -> List[Dict]:
    """Parse a single log file and return epoch metrics."""
    epochs = []

    with open(filepath, "r") as f:
        for line in f:
            match = EPOCH_REGEX.search(line)
            if match:
                epochs.append({
                    "epoch": int(match.group("epoch")),
                    "train_loss": float(match.group("train_loss")),
                    "train_accuracy": float(match.group("train_acc")),
                    "test_loss": float(match.group("test_loss")),
                    "test_accuracy": float(match.group("test_acc")),
                })

    return epochs


def parse_logs(log_dir: str, job_ids: List[int]) -> Dict[int, Dict]:
    """Parse logs for selected job IDs."""
    data = {}

    for filename in os.listdir(log_dir):
        job_id = extract_job_id(filename)

        if job_id is None or job_id not in job_ids:
            continue

        filepath = os.path.join(log_dir, filename)
        epochs = parse_log_file(filepath)

        data[job_id] = {
            "log_file": filename,
            "epochs": epochs,
        }

    return data


def plot_training_metrics(parsed_data, save_path, dpi=150):
    # model_ids = list(parsed_data.keys())
    model_ids = [job_id for job_id in TITLES.keys() if job_id in parsed_data]
    n_models = len(model_ids)

    fig, axes = plt.subplots(
        2, n_models,
        figsize=(5 * n_models, 8),
        sharey='row',
        sharex='col'
    )

    if n_models == 1:
        axes = axes.reshape(2, 1)

    for col, model_id in enumerate(model_ids):
        epochs_data = parsed_data[model_id]["epochs"]

        epochs = [e["epoch"] for e in epochs_data]
        train_loss = [e["train_loss"] for e in epochs_data]
        test_loss = [e["test_loss"] for e in epochs_data]
        train_acc = [e["train_accuracy"] for e in epochs_data]
        test_acc = [e["test_accuracy"] for e in epochs_data]

        ax_loss = axes[0, col]
        ax_loss.plot(epochs, train_loss, label="Train Loss")
        ax_loss.plot(epochs, test_loss, label="Test Loss")
        ax_loss.set_title(TITLES[model_id])
        ax_loss.set_xlabel("Epoch")
        if col == 0:
            ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        ax_acc = axes[1, col]
        ax_acc.plot(epochs, train_acc, label="Train Accuracy")
        ax_acc.plot(epochs, test_acc, label="Test Accuracy")
        ax_acc.set_xlabel("Epoch")
        if col == 0:
            ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    ax_loss.set_ylim(0, ax_loss.get_ylim()[1])
    ax_acc.set_ylim(0, ax_acc.get_ylim()[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parsed_data = parse_logs(LOG_DIR, FILES_TO_PARSE)
    plot_training_metrics(parsed_data, save_path=PLOTS_DIR + "/training_metrics.png")

