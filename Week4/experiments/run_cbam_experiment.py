import argparse
import subprocess
import json
import time
import sys
import os
import glob
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

CHANNEL_REDUCTIONS = [16, 8, 4]

SPATIAL_CONFIGS = [
    {"kernel": 7, "dilation": 1},
    {"kernel": 5, "dilation": 1},
    {"kernel": 3, "dilation": 1},
    {"kernel": 3, "dilation": 3},
    {"kernel": 5, "dilation": 2},
]

NUM_CBAM_BLOCKS = [1, 2, 3, 4]


def generate_experiments() -> List[Dict]:
    experiments = []
    exp_id = 1

    for r in CHANNEL_REDUCTIONS:
        for s in SPATIAL_CONFIGS:
            for n in NUM_CBAM_BLOCKS:
                name = f"cbam_r{r}_k{s['kernel']}_d{s['dilation']}_n{n}"
                experiments.append({
                    "id": exp_id,
                    "name": name,
                    "reduction": r,
                    "kernel": s["kernel"],
                    "dilation": s["dilation"],
                    "cbam_num_blocks": n,
                    "description": f"CBAM reduction={r}, kernel={s['kernel']}, dilation={s['dilation']}"
                })
                exp_id += 1

    return experiments


def parse_training_summary(summary_file: Path) -> Dict:
    results = {
        "val_accuracy": None,
        "test_accuracy": None,
        "train_accuracy_at_best_epoch": None,
        "best_epoch": None
    }

    with open(summary_file) as f:
        for line in f:
            line = line.strip()

            if "Training Accuracy at Best Epoch:" in line:
                results["train_accuracy_at_best_epoch"] = float(
                    line.split(":")[1].replace("%", "")
                ) / 100.0

            elif "Best Validation Accuracy:" in line:
                results["val_accuracy"] = float(
                    line.split(":")[1].replace("%", "")
                ) / 100.0

            elif "Best Test Accuracy:" in line:
                results["test_accuracy"] = float(
                    line.split(":")[1].replace("%", "")
                ) / 100.0

            elif line.startswith("Best Epoch:"):
                results["best_epoch"] = int(line.split(":")[1])

    return results


def run_single_experiment(exp: Dict, args, week4_dir: Path) -> Dict:
    print("\n" + "=" * 70)
    print(f"Running {exp['id']}/15: {exp['name']}")
    print(exp["description"])
    print("=" * 70)

    cmd = [
        "python", str(week4_dir / "main.py"),
        "--data_root", args.data_root,
        "--output_dir", args.output_dir,
        "--wandb_project", args.wandb_project,
        "--experiment_name", exp["name"],
        "--model_type", "cbam_optimized",
        "--cbam_reduction", str(exp["reduction"]),
        "--cbam_spatial_kernel", str(exp["kernel"]),
        "--cbam_dilation", str(exp["dilation"]),
        "--cbam_num_blocks", str(exp["cbam_num_blocks"]),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
    ]

    start = time.time()
    success, error = True, None

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        success = False
        error = str(e)

    elapsed = (time.time() - start) / 60

    exp["results"] = {
        "success": success,
        "elapsed_time_minutes": elapsed,
        "error": error,
        "val_accuracy": None,
        "test_accuracy": None,
        "train_accuracy_at_best_epoch": None,
        "best_epoch": None,
    }

    if success:
        pattern = os.path.join(
            args.output_dir, exp["name"], f"{exp['name']}_*", "training_summary.txt"
        )
        files = glob.glob(pattern)

        if files:
            summary = max(files, key=os.path.getctime)
            exp["results"].update(parse_training_summary(Path(summary)))

    return exp


def main():
    parser = argparse.ArgumentParser("CBAM Hyperparameter Search")

    parser.add_argument("--data_root", type=str,
                        default="/data/uabmcv2526/shared/dataset/2425/MIT_small_train_1")
    parser.add_argument("--output_dir", type=str,
                        default="/data/uabmcv2526/mcvstudent27/Week4/output/cbam_search")
    parser.add_argument("--wandb_project", type=str, default="C3_Week4_CBAM")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    week4_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = generate_experiments()

    for exp in experiments:
        run_single_experiment(exp, args, week4_dir)
        with open(output_dir / "progress.json", "w") as f:
            json.dump(experiments, f, indent=2)

    with open(output_dir / "cbam_results.json", "w") as f:
        json.dump(experiments, f, indent=2)

    print("\n✓ CBAM hyperparameter search complete!")


if __name__ == "__main__":
    main()
