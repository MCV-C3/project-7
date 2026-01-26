"""
Architecture Search Experiment: Depth and Width Grid Search
Tests different combinations of network depth and channel width to find optimal architecture.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Add parent directory to path to import from Week4
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configuration space for grid search
WIDTH_CONFIGS = {
    'narrow': [16, 32, 64, 128],
    'baseline': [32, 64, 128, 256],
    'wide': [48, 96, 192, 384]
}

DEPTH_CONFIGS = {
    'shallow': 3,  # 3 blocks
    'baseline': 4,  # 4 blocks  
    'deep': 5      # 5 blocks
}

FC_HIDDEN_CONFIGS = {
    'small': 256,
    'baseline': 512
}


def generate_experiments() -> List[Dict]:
    """Generate all experiment configurations for depth x width x fc_hidden grid."""
    experiments = []
    exp_id = 1
    
    for depth_name, num_blocks in DEPTH_CONFIGS.items():
        for width_name, channels_full in WIDTH_CONFIGS.items():
            for fc_name, fc_hidden in FC_HIDDEN_CONFIGS.items():
                # Adjust channels based on depth
                if num_blocks <= len(channels_full):
                    # Truncate if we need fewer blocks
                    channels = channels_full[:num_blocks]
                else:
                    # Extend by doubling the last channel value
                    channels = channels_full.copy()
                    last_channel = channels[-1]
                    while len(channels) < num_blocks:
                        channels.append(last_channel * 2)
                        last_channel = channels[-1]
                
                exp_config = {
                    'id': exp_id,
                    'name': f'{depth_name}_{width_name}_fc{fc_name}',
                    'depth_name': depth_name,
                    'width_name': width_name,
                    'fc_name': fc_name,
                    'num_blocks': num_blocks,
                    'channels': channels,
                    'channels_str': ','.join(map(str, channels)),
                    'fc_hidden': fc_hidden
                }
                experiments.append(exp_config)
                exp_id += 1
    
    return experiments


def run_single_experiment(exp: Dict, args: argparse.Namespace, week4_dir: Path) -> Dict:
    """Run a single training experiment with given configuration."""
    print(f"\n{'='*70}")
    print(f"Running Experiment {exp['id']}/18: {exp['name']}")
    print(f"  Depth: {exp['depth_name']} ({exp['num_blocks']} blocks)")
    print(f"  Width: {exp['width_name']} ({exp['channels']})")
    print(f"  FC Hidden: {exp['fc_name']} ({exp['fc_hidden']} units)")
    print(f"{'='*70}\n")
    
    # Build command - match baseline exactly except for architecture params
    cmd = [
        'python', str(week4_dir / 'main.py'),
        '--data_root', args.data_root,
        '--output_dir', args.output_dir,  # All trials save inside arch_search
        '--wandb_project', args.wandb_project,
        '--experiment_name', exp['name'],
        '--model_type', 'flexible',
        '--channels', exp['channels_str'],
        '--fc_hidden', str(exp['fc_hidden']),
        '--kernel_size', '3',  # Baseline default
        '--pooling_type', 'max',  # Baseline default
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--learning_rate', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--dropout', str(args.dropout),
        '--optimizer', args.optimizer,
        '--seed', str(args.seed),
        '--num_workers', str(args.num_workers)
    ]
    
    # Note: NO --use_scheduler (baseline doesn't use scheduler)
    
    start_time = time.time()
    
    # Run training (output goes directly to stdout/stderr)
    result = subprocess.run(cmd)
    
    elapsed_time = time.time() - start_time
    
    # Check if training succeeded
    if result.returncode != 0:
        print(f"\n⚠ Training failed with return code {result.returncode}")
        exp['results'] = {
            'val_accuracy': 0.0,
            'best_epoch': 0,
            'elapsed_time': elapsed_time,
            'output_dir': 'N/A',
            'error': f'Training failed with return code {result.returncode}'
        }
        return exp
    
    # Parse results from training summary
    output_dir = Path(args.output_dir)
    exp_dir = output_dir / exp['name']
    if exp_dir.exists():
        # Find the latest subdirectory
        subdirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
        if subdirs:
            latest_dir = subdirs[-1]
            summary_file = latest_dir / 'training_summary.txt'
            
            # Parse validation accuracy from summary
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    content = f.read()
                    val_acc = None
                    best_epoch = None
                    
                    # Extract best validation accuracy and epoch
                    for line in content.split('\n'):
                        if 'Best Validation Accuracy:' in line:
                            val_acc = float(line.split(':')[1].strip())
                        if 'Best Epoch:' in line:
                            best_epoch = int(line.split(':')[1].strip())
                    
                    if val_acc is not None and best_epoch is not None:
                        exp['results'] = {
                            'val_accuracy': val_acc,
                            'best_epoch': best_epoch,
                            'elapsed_time': elapsed_time,
                            'output_dir': str(latest_dir)
                        }
                        return exp
    
    # If parsing failed, return with null results
    exp['results'] = {
        'val_accuracy': 0.0,
        'best_epoch': 0,
        'elapsed_time': elapsed_time,
        'output_dir': 'N/A',
        'error': 'Failed to parse results'
    }
    return exp


def save_results_summary(experiments: List[Dict], output_dir: Path):
    """Save experiment results and generate top-3 summary."""
    
    # Sort by validation accuracy
    valid_exps = [e for e in experiments if 'results' in e and e['results']['val_accuracy'] > 0]
    
    if not valid_exps:
        print("\n⚠ WARNING: No valid experiments completed successfully!")
        print("Check the logs for errors.")
        # Still save the failed results
        results_file = output_dir / 'all_results.json'
        with open(results_file, 'w') as f:
            json.dump(experiments, f, indent=2)
        print(f"Failed results saved to {results_file}")
        return
    
    sorted_exps = sorted(valid_exps, key=lambda x: x['results']['val_accuracy'], reverse=True)
    
    # Save full results as JSON
    results_file = output_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    print(f"\n✓ Full results saved to {results_file}")
    
    # Generate summary text file
    summary_file = output_dir / 'arch_search_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ARCHITECTURE SEARCH: DEPTH, WIDTH & FC LAYER GRID SEARCH\n")
        f.write("="*70 + "\n\n")
        f.write("Experiment Name: arch_search\n")
        f.write("Experiment: Testing network depth, channel width, and FC layer size\n")
        f.write("Objective: Find optimal architecture for MIT Indoor Scenes (400 train images)\n")
        f.write(f"Total experiments: {len(experiments)}\n")
        f.write(f"Successful experiments: {len(valid_exps)}\n\n")
        
        f.write("="*70 + "\n")
        f.write("TOP 3 ARCHITECTURES\n")
        f.write("="*70 + "\n\n")
        
        for i, exp in enumerate(sorted_exps[:3], 1):
            f.write(f"#{i} - {exp['name']}\n")
            f.write(f"  Configuration:\n")
            f.write(f"    Depth: {exp['depth_name']} ({exp['num_blocks']} blocks)\n")
            f.write(f"    Width: {exp['width_name']} {exp['channels']}\n")
            f.write(f"    FC Hidden: {exp['fc_name']} ({exp['fc_hidden']} units)\n")
            f.write(f"  Results:\n")
            f.write(f"    Validation Accuracy: {exp['results']['val_accuracy']:.2%}\n")
            f.write(f"    Best Epoch: {exp['results']['best_epoch']}\n")
            f.write(f"    Training Time: {exp['results']['elapsed_time']/60:.1f} min\n")
            f.write(f"    Output: {exp['results']['output_dir']}\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("ALL RESULTS (sorted by validation accuracy)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'Rank':<6}{'Name':<30}{'Depth':<10}{'Width':<12}{'FC':<10}{'Val Acc':<12}{'Epoch':<8}\n")
        f.write("-"*70 + "\n")
        
        for i, exp in enumerate(sorted_exps, 1):
            f.write(f"{i:<6}{exp['name']:<30}{exp['depth_name']:<10}")
            f.write(f"{exp['width_name']:<12}{exp['fc_name']:<10}")
            f.write(f"{exp['results']['val_accuracy']:.2%}  {exp['results']['best_epoch']:<8}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*70 + "\n\n")
        
        # Analyze results
        best = sorted_exps[0]
        f.write(f"✓ Best architecture: {best['name']}\n")
        f.write(f"  - {best['num_blocks']} convolutional blocks ({best['depth_name']} depth)\n")
        f.write(f"  - {best['width_name']} width with channels {best['channels']}\n")
        f.write(f"  - {best['fc_name']} FC layer with {best['fc_hidden']} hidden units\n")
        f.write(f"  - Achieved {best['results']['val_accuracy']:.2%} validation accuracy\n\n")
        
        # Compare depths
        depth_results = {}
        for exp in sorted_exps:
            depth = exp['depth_name']
            if depth not in depth_results:
                depth_results[depth] = []
            depth_results[depth].append(exp['results']['val_accuracy'])
        
        f.write("Average accuracy by depth:\n")
        for depth in ['shallow', 'baseline', 'deep']:
            if depth in depth_results:
                avg = sum(depth_results[depth]) / len(depth_results[depth])
                f.write(f"  - {depth}: {avg:.2%} (avg of {len(depth_results[depth])} configs)\n")
        f.write("\n")
        
        # Compare widths
        width_results = {}
        for exp in sorted_exps:
            width = exp['width_name']
            if width not in width_results:
                width_results[width] = []
            width_results[width].append(exp['results']['val_accuracy'])
        
        f.write("Average accuracy by width:\n")
        for width in ['narrow', 'baseline', 'wide']:
            if width in width_results:
                avg = sum(width_results[width]) / len(width_results[width])
                f.write(f"  - {width}: {avg:.2%} (avg of {len(width_results[width])} configs)\n")
        f.write("\n")
        
        # Compare FC sizes
        fc_results = {}
        for exp in sorted_exps:
            fc = exp['fc_name']
            if fc not in fc_results:
                fc_results[fc] = []
            fc_results[fc].append(exp['results']['val_accuracy'])
        
        f.write("Average accuracy by FC layer size:\n")
        for fc in ['small', 'baseline']:
            if fc in fc_results:
                avg = sum(fc_results[fc]) / len(fc_results[fc])
                f.write(f"  - {fc}: {avg:.2%} (avg of {len(fc_results[fc])} configs)\n")
    
    print(f"✓ Summary saved to {summary_file}\n")
    
    # Print top-3 to console
    print("\n" + "="*70)
    print("TOP 3 RESULTS")
    print("="*70)
    for i, exp in enumerate(sorted_exps[:3], 1):
        print(f"#{i} {exp['name']}: {exp['results']['val_accuracy']:.2%} "
              f"({exp['depth_name']}, {exp['width_name']}, FC:{exp['fc_name']})")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Architecture Search: Depth, Width & FC Layer Grid Search')
    
    # Data parameters (match baseline exactly)
    parser.add_argument('--data_root', type=str, 
                       default='/data/uabmcv2526/shared/dataset/2425/MIT_small_train_1',
                       help='Root directory of the dataset')
    parser.add_argument('--wandb_project', type=str, default='C3_Week4',
                       help='Wandb project name')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    
    # Training hyperparameters (match baseline exactly)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--seed', type=int, default=42)
    
    # Architecture parameters NOT tested (fixed to baseline values)
    # kernel_size=3, pooling_type='max' are defaults in FlexibleCNN
    
    # Output directory
    parser.add_argument('--output_dir', type=str, 
                       default='/data/uabmcv2526/mcvstudent29/Week4/output/arch_search',
                       help='Main experiment directory where all trials will be saved')
    
    args = parser.parse_args()
    
    # Get Week4 directory (parent of experiments/)
    week4_dir = Path(__file__).parent.parent
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ARCHITECTURE SEARCH: DEPTH, WIDTH & FC LAYER GRID SEARCH")
    print("="*70)
    print(f"Experiment name: arch_search")
    print(f"Testing 18 configurations (3 depths × 3 widths × 2 FC sizes)")
    print(f"Results will be saved to: {output_dir}\n")
    
    # Generate all experiment configurations
    experiments = generate_experiments()
    
    print("Experiment configurations:")
    for exp in experiments:
        print(f"  {exp['id']}. {exp['name']}: {exp['num_blocks']} blocks, "
              f"channels {exp['channels']}, FC {exp['fc_hidden']}")
    print()
    
    # Run all experiments
    for exp in experiments:
        exp = run_single_experiment(exp, args, week4_dir)
        
        # Save intermediate results after each experiment
        with open(output_dir / 'progress.json', 'w') as f:
            json.dump(experiments, f, indent=2)
    
    # Generate final summary
    save_results_summary(experiments, output_dir)
    
    print("\n✓ Architecture search complete!")
    print(f"  Results directory: {output_dir}")
    print(f"  Summary: {output_dir / 'arch_search_summary.txt'}")


if __name__ == '__main__':
    main()
