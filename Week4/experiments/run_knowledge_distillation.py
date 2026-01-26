import argparse
from email import parser
import subprocess
import json
import time
import sys
import os
import glob
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

TEMPERATURE = [1, 2, 4, 8, 16]
ALPHA = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def generate_experiments() -> List[Dict]:
    experiments = []
    exp_id = 1

    for t in TEMPERATURE:
        for a in ALPHA:
            name = f"kn_dist_t{t}_a{a}"
            experiments.append({
                "id": exp_id,
                "name": name,
                "temperature": t,
                "alpha": a,
                "description": f"Knowledge Dist. with temperature={t}, alpha={a}"
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


def run_single_experiment(exp: Dict, args, week4_dir: Path, total: int) -> Dict:
    print("\n" + "=" * 70)
    print(f"Running {exp['id']}/{total}: {exp['name']}")
    print(exp["description"])
    print("=" * 70)

    cmd = [
        "python", str(week4_dir / "main.py"),
        # main
        "--data_root", args.data_root,
        "--output_dir", args.output_dir,
        "--wandb_project", args.wandb_project,
        "--experiment_name", exp["name"],
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        # model
        "--model_type", str(args.model_type),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--momentum", str(args.momentum),
        "--optimizer", str(args.optimizer),
        "--dropout", str(args.dropout),
        # attention
        "--cbam_reduction", str(args.cbam_reduction),
        "--cbam_spatial_kernel", str(args.cbam_spatial_kernel),
        "--cbam_dilation", str(args.cbam_dilation),
        "--cbam_num_blocks", str(args.cbam_num_blocks),
        # data augmentation
        "--use_flip",
        "--use_color",
        "--use_geometric",
        "--use_translation",
        "--aug_ratio", str(args.aug_ratio),
        # distillation
        "--use_distillation",
        "--teacher_model_type", str(args.teacher_model_type),
        "--teacher_checkpoint", str(args.teacher_checkpoint),
        "--distill_alpha", str(exp["alpha"]),
        "--distill_temperature", str(exp["temperature"]),
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
    parser = argparse.ArgumentParser("Knowledge Distillation Study")

    parser.add_argument("--data_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--model_type", type=str, default="student")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--dropout", type=float, default=0.3)
   
    parser.add_argument(
        "--cbam_reduction",
        type=int,
        default=4,
        help="Reduction ratio for channel attention in CBAMOptimizedCNN (default: 4)"
    )
    parser.add_argument(
        "--cbam_spatial_kernel",
        type=int,
        default=7,
        help="Kernel size for spatial attention in CBAMOptimizedCNN (default: 7)"
    )
    parser.add_argument(
        "--cbam_dilation",
        type=int,
        default=1,
        help="Dilation size for spatial attention in CBAMOptimizedCNN (default: 1)"
    )
    parser.add_argument(
        "--cbam_num_blocks",
        type=int,
        default=4,
        help="Number of CBAM blocks to include in CBAMOptimizedCNN (0-4, default: 4)"
    )

    # Data augmentation
    parser.add_argument("--use_flip", action="store_true")
    parser.add_argument("--use_color", action="store_true")
    parser.add_argument("--use_geometric", action="store_true")
    parser.add_argument("--use_translation", action="store_true")
    parser.add_argument("--aug_ratio", type=float, default=1.0)

    # Distillation
    parser.add_argument("--use_distillation", action="store_true")
    parser.add_argument("--teacher_model_type", type=str, default="cbam_optimized")
    parser.add_argument("--teacher_checkpoint", type=str, default="")

    args = parser.parse_args()

    week4_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = generate_experiments()

    for exp in experiments:
        run_single_experiment(exp, args, week4_dir, total=len(experiments))
        with open(output_dir / "progress.json", "w") as f:
            json.dump(experiments, f, indent=2)

    with open(output_dir / "knowl_dist_results.json", "w") as f:
        json.dump(experiments, f, indent=2)

    print("\n✓ Knowledge distillation complete!")

if __name__ == "__main__":
    main()