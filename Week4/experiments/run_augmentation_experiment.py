import argparse
from html import parser
import subprocess
import itertools
import time
import json
import os
from pathlib import Path

AUG_BLOCKS = ["flip", "color", "geometric", "translation"]
AUG_RATIOS = [0.5, 1.0, 1.5, 2.0]

def generate_experiments():
    experiments = []
    exp_id = 1

    for k in [2, 3, 4]:
        for combo in itertools.combinations(AUG_BLOCKS, k):
            for aug_ratio in AUG_RATIOS:
                name = "aug_" + "_".join(combo)
                experiments.append({
                    "id": exp_id,
                    "name": name,
                    "blocks": combo,
                    "aug_ratio": aug_ratio
                })
                exp_id += 1

    return experiments


def run_single_experiment(exp, args, week4_dir):
    print("\n" + "=" * 70)
    print(f"Running {exp['id']}/15: {exp['name']}")
    print(f"Augmentations: {exp['blocks']}")
    print("=" * 70)

    cmd = [
        "python", str(week4_dir / "main.py"),
        "--data_root", args.data_root,
        "--output_dir", args.output_dir,
        "--wandb_project", args.wandb_project,
        "--experiment_name", exp["name"],
        "--model_type", args.model_type,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--aug_ratio", str(exp["aug_ratio"]),
        "--cbam_reduction", str(args.cbam_reduction),
        "--cbam_spatial_kernel", str(args.cbam_spatial_kernel),
        "--cbam_dilation", str(args.cbam_dilation),
        "--cbam_num_blocks", str(args.cbam_num_blocks),
    ]

    # Enable selected blocks
    for block in exp["blocks"]:
        cmd.append(f"--use_{block}")

    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = (time.time() - start) / 60

    exp["elapsed_minutes"] = elapsed
    return exp


def main():
    parser = argparse.ArgumentParser("Augmentation Ablation Study")

    parser.add_argument("--data_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--wandb_project", type=str)

    parser.add_argument("--model_type", type=str, default="optimized")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
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


    args = parser.parse_args()

    week4_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = generate_experiments()

    for exp in experiments:
        run_single_experiment(exp, args, week4_dir)
        with open(output_dir / "progress.json", "w") as f:
            json.dump(experiments, f, indent=2)

    with open(output_dir / "augmentation_results.json", "w") as f:
        json.dump(experiments, f, indent=2)

    print("\n✓ Augmentation ablation complete!")


if __name__ == "__main__":
    main()
