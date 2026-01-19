"""
Architecture Search V2: Comprehensive Grid Search
Combines depth, width, and adaptive pooling search in a single experiment.

Key improvements from V1:
- Tests narrower architectures (extra_narrow)
- Only tests avg pooling (max pooling showed no benefit)
- Systematically tests adaptive pooling impact
- Fixed parsing error for train accuracy at best epoch
"""

import argparse
import subprocess
import json
import time
import sys
import os
import glob
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add parent directory to path to import from Week4
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configuration space for grid search
WIDTH_CONFIGS = {
    'extra_narrow': [8, 16, 32, 64],
    'narrow': [16, 32, 64, 128],
    'baseline': [32, 64, 128, 256]
}

DEPTH_CONFIGS = {
    'shallow': 2,   # 2 blocks
    'moderate': 3,  # 3 blocks  
    'baseline': 4   # 4 blocks (original SimpleCNN)
}

# Pooling configurations
POOLING_CONFIGS = [
    # For (5,5) and (3,3): only use fc_hidden=512, test both max and avg
    {'size': (5, 5), 'type': 'max', 'use_fc_hidden': True, 'fc_hidden': 512, 'suffix': 'pool5x5_max_fc512'},
    {'size': (5, 5), 'type': 'avg', 'use_fc_hidden': True, 'fc_hidden': 512, 'suffix': 'pool5x5_avg_fc512'},
    {'size': (3, 3), 'type': 'max', 'use_fc_hidden': True, 'fc_hidden': 512, 'suffix': 'pool3x3_max_fc512'},
    {'size': (3, 3), 'type': 'avg', 'use_fc_hidden': True, 'fc_hidden': 512, 'suffix': 'pool3x3_avg_fc512'},
    
    # For (1,1) GAP: test both fc_hidden=512 and direct classification, with both max and avg
    {'size': (1, 1), 'type': 'max', 'use_fc_hidden': True, 'fc_hidden': 512, 'suffix': 'gap_max_fc512'},
    {'size': (1, 1), 'type': 'avg', 'use_fc_hidden': True, 'fc_hidden': 512, 'suffix': 'gap_avg_fc512'},
    {'size': (1, 1), 'type': 'max', 'use_fc_hidden': False, 'fc_hidden': None, 'suffix': 'gap_max_direct'},
    {'size': (1, 1), 'type': 'avg', 'use_fc_hidden': False, 'fc_hidden': None, 'suffix': 'gap_avg_direct'},
]


def calculate_fc_params(pool_size: tuple, last_channels: int, fc_hidden: int, 
                       num_classes: int, use_fc_hidden: bool) -> int:
    """Calculate total parameters in FC layers."""
    flattened_size = last_channels * pool_size[0] * pool_size[1]
    
    if use_fc_hidden:
        fc1_params = flattened_size * fc_hidden + fc_hidden  # weights + bias
        fc2_params = fc_hidden * num_classes + num_classes
        return fc1_params + fc2_params
    else:
        return flattened_size * num_classes + num_classes


def generate_experiments() -> List[Dict]:
    """Generate all experiment configurations for depth × width × pooling grid."""
    experiments = []
    exp_id = 1
    
    for depth_name, num_blocks in DEPTH_CONFIGS.items():
        for width_name, channels_list in WIDTH_CONFIGS.items():
            for pool_config in POOLING_CONFIGS:
                # Truncate channels to match num_blocks
                channels_truncated = channels_list[:num_blocks]
                
                # Calculate FC params
                last_channel = channels_truncated[-1]
                fc_params = calculate_fc_params(
                    pool_config['size'],
                    last_channel,
                    pool_config['fc_hidden'] if pool_config['use_fc_hidden'] else 0,
                    8,  # MIT scenes classes
                    pool_config['use_fc_hidden']
                )
                
                exp = {
                    'id': exp_id,
                    'name': f"{depth_name}_{width_name}_{pool_config['suffix']}",
                    'depth_name': depth_name,
                    'num_blocks': num_blocks,
                    'width_name': width_name,
                    'channels': channels_truncated,
                    'channels_str': ','.join(map(str, channels_truncated)),
                    'pool_size': pool_config['size'],
                    'pool_type': pool_config['type'],
                    'use_fc_hidden': pool_config['use_fc_hidden'],
                    'fc_hidden': pool_config['fc_hidden'] if pool_config['use_fc_hidden'] else None,
                    'fc_params': fc_params,
                    'description': f"{num_blocks} blocks, {width_name} {channels_truncated}, "
                                 f"pool {pool_config['size']} ({pool_config['type']}), "
                                 f"{'FC512' if pool_config['use_fc_hidden'] else 'direct'}"
                }
                experiments.append(exp)
                exp_id += 1
    
    return experiments


def parse_training_summary(summary_file: Path) -> Dict:
    """Parse training_summary.txt to extract results."""
    results = {
        'val_accuracy': None,
        'test_accuracy': None,
        'train_accuracy_at_best_epoch': None,
        'best_epoch': None
    }
    
    try:
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            
        # Parse line by line
        for line in lines:
            line = line.strip()
            
            # Parse train accuracy at best epoch FIRST (format: "Training Accuracy at Best Epoch: 91.00%")
            # Must come before "Best Epoch:" check since it contains that substring
            if 'Training Accuracy at Best Epoch:' in line:
                try:
                    train_str = line.split(':')[1].strip()
                    train_str = train_str.replace('%', '')
                    results['train_accuracy_at_best_epoch'] = float(train_str) / 100.0
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Could not parse train accuracy from: {line}")
            
            # Parse best validation accuracy (format: "Best Validation Accuracy: 72.60%")
            elif 'Best Validation Accuracy:' in line:
                try:
                    # Extract percentage value
                    val_str = line.split(':')[1].strip()
                    val_str = val_str.replace('%', '')
                    results['val_accuracy'] = float(val_str) / 100.0
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Could not parse validation accuracy from: {line}")
            
            # Parse test accuracy (format: "Best Test Accuracy: 72.60%")
            elif 'Best Test Accuracy:' in line:
                try:
                    test_str = line.split(':')[1].strip()
                    test_str = test_str.replace('%', '')
                    results['test_accuracy'] = float(test_str) / 100.0
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Could not parse test accuracy from: {line}")
            
            # Parse best epoch (format: "Best Epoch: 18")
            # Must come AFTER "Training Accuracy at Best Epoch:" check
            elif 'Best Epoch:' in line:
                try:
                    epoch_str = line.split(':')[1].strip()
                    results['best_epoch'] = int(epoch_str)
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Could not parse best epoch from: {line}")
        
        # Validate we got the key metrics
        if results['val_accuracy'] is None or results['test_accuracy'] is None:
            print(f"  Warning: Missing key metrics (val={results['val_accuracy']}, test={results['test_accuracy']})")
            print(f"  This might indicate the training_summary.txt has an unexpected format")
            
    except Exception as e:
        print(f"  ERROR parsing summary file {summary_file}: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def run_single_experiment(exp: Dict, args: argparse.Namespace, week4_dir: Path) -> Dict:
    """Run a single training experiment with given configuration."""
    print(f"\n{'='*70}")
    print(f"Running Experiment {exp['id']}/72: {exp['name']}")
    print(f"  Depth: {exp['depth_name']} ({exp['num_blocks']} blocks)")
    print(f"  Width: {exp['width_name']} {exp['channels']}")
    print(f"  Pool: {exp['pool_size']} ({exp['pool_type']})")
    print(f"  FC config: {'Hidden (512)' if exp['use_fc_hidden'] else 'Direct'}")
    print(f"  FC params: {exp['fc_params']:,}")
    print(f"{'='*70}\n")
    
    # Build command
    cmd = [
        'python', str(week4_dir / 'main.py'),
        '--data_root', args.data_root,
        '--output_dir', args.output_dir,
        '--wandb_project', args.wandb_project,
        '--experiment_name', exp['name'],
        '--model_type', 'flexible',
        '--channels', exp['channels_str'],
        '--fc_hidden', str(exp['fc_hidden']) if exp['fc_hidden'] else '512',  # dummy value if not used
        '--kernel_size', '3',
        '--pooling_type', 'max',  # Conv block pooling (inside blocks)
        '--pool_output_size', f"{exp['pool_size'][0]},{exp['pool_size'][1]}",
        '--adaptive_pool_type', exp['pool_type'],  # Test both max and avg
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--learning_rate', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--dropout', str(args.dropout),
        '--optimizer', 'AdamW',
        '--seed', str(args.seed),
        '--num_workers', str(args.num_workers)
    ]
    
    # Add FC hidden flag
    if exp['use_fc_hidden']:
        cmd.append('--use_fc_hidden')
    else:
        cmd.append('--no_fc_hidden')
    
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True)
        success = True
        error_msg = None
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = str(e)
        print(f"\n⚠ ERROR: Experiment {exp['name']} failed!")
        print(f"Error: {error_msg}")
    
    elapsed_time = time.time() - start_time
    
    # Parse results from training summary
    exp['results'] = {
        'success': success,
        'elapsed_time_minutes': elapsed_time / 60,
        'val_accuracy': None,
        'test_accuracy': None,
        'train_accuracy_at_best_epoch': None,
        'best_epoch': None,
        'output_dir': None,
        'error': error_msg
    }
    
    if success:
        # Find summary file
        summary_pattern = os.path.join(args.output_dir, exp['name'], 
                                      f"{exp['name']}_*", "training_summary.txt")
        summary_files = glob.glob(summary_pattern)
        
        if summary_files:
            latest_summary = max(summary_files, key=os.path.getctime)
            print(f"  Found summary file: {latest_summary}")
            
            parsed_results = parse_training_summary(Path(latest_summary))
            exp['results'].update(parsed_results)
            exp['results']['output_dir'] = str(Path(latest_summary).parent)
            
            # Check if parsing was successful
            if (exp['results']['val_accuracy'] is not None and 
                exp['results']['test_accuracy'] is not None and
                exp['results']['train_accuracy_at_best_epoch'] is not None and
                exp['results']['best_epoch'] is not None):
                print(f"  ✓ Parsed results successfully:")
                print(f"    Val Acc: {exp['results']['val_accuracy']:.4f} | "
                      f"Test Acc: {exp['results']['test_accuracy']:.4f} | "
                      f"Train Acc @ epoch {exp['results']['best_epoch']}: {exp['results']['train_accuracy_at_best_epoch']:.4f}")
            else:
                print(f"  ⚠ Warning: Incomplete parsing from summary file!")
                print(f"    Val: {exp['results']['val_accuracy']}, Test: {exp['results']['test_accuracy']}, "
                      f"Train: {exp['results']['train_accuracy_at_best_epoch']}, Epoch: {exp['results']['best_epoch']}")
                print(f"    Check {latest_summary} for formatting issues")
        else:
            print(f"  ⚠ Warning: Could not find summary file matching pattern:")
            print(f"    {summary_pattern}")
            exp['results']['error'] = 'Summary file not found'
    
    return exp


def save_results_summary(experiments: List[Dict], output_dir: Path):
    """Save experiment results and generate summary."""
    
    # Sort by test accuracy (prefer test over val for final ranking)
    valid_exps = [e for e in experiments if e['results']['success'] and e['results']['test_accuracy'] is not None]
    
    if not valid_exps:
        print("\n⚠ WARNING: No valid experiments completed successfully!")
        print("Check the logs for errors.")
        results_file = output_dir / 'all_results.json'
        with open(results_file, 'w') as f:
            json.dump(experiments, f, indent=2)
        print(f"Failed results saved to {results_file}")
        return
    
    sorted_exps = sorted(valid_exps, key=lambda x: x['results']['test_accuracy'], reverse=True)
    
    # Save full results as JSON
    results_file = output_dir / 'arch_search_v2_results.json'
    with open(results_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    print(f"\n✓ Full results saved to {results_file}")
    
    # Generate summary text file
    summary_file = output_dir / 'arch_search_v2_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ARCHITECTURE SEARCH V2: COMPREHENSIVE GRID SEARCH\n")
        f.write("="*80 + "\n\n")
        f.write("Experiment Design:\n")
        f.write("  - Depth: 2, 3, 4 blocks (shallow/moderate/baseline)\n")
        f.write("  - Width: extra_narrow [8,16,32,64], narrow [16,32,64,128], baseline [32,64,128,256]\n")
        f.write("  - Pooling: (5,5), (3,3), (1,1) with BOTH max and avg\n")
        f.write("  - FC configs: hidden (512) for all, direct for (1,1) only\n")
        f.write(f"  - Total configurations: {len(experiments)}\n\n")
        f.write(f"Successful experiments: {len(valid_exps)}/{len(experiments)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOP 10 ARCHITECTURES (by test accuracy)\n")
        f.write("="*80 + "\n\n")
        
        for i, exp in enumerate(sorted_exps[:10], 1):
            f.write(f"#{i} - {exp['name']}\n")
            f.write(f"  Configuration:\n")
            f.write(f"    Depth: {exp['depth_name']} ({exp['num_blocks']} blocks)\n")
            f.write(f"    Width: {exp['width_name']} {exp['channels']}\n")
            f.write(f"    Pool: {exp['pool_size']} ({exp['pool_type']})\n")
            f.write(f"    FC: {'Hidden (512)' if exp['use_fc_hidden'] else 'Direct'} ({exp['fc_params']:,} params)\n")
            f.write(f"  Results:\n")
            f.write(f"    Test Accuracy: {exp['results']['test_accuracy']:.4f}\n")
            f.write(f"    Val Accuracy: {exp['results']['val_accuracy']:.4f}\n")
            f.write(f"    Train Acc @ Best: {exp['results']['train_accuracy_at_best_epoch']:.4f}\n")
            f.write(f"    Best Epoch: {exp['results']['best_epoch']}\n")
            f.write(f"    Training Time: {exp['results']['elapsed_time_minutes']:.1f} min\n")
            f.write(f"    Overfitting Gap: {exp['results']['train_accuracy_at_best_epoch'] - exp['results']['test_accuracy']:.4f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("ALL RESULTS (sorted by test accuracy)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Rank':<6}{'Name':<35}{'Test Acc':<12}{'Val Acc':<12}{'Train@Best':<12}{'Epoch':<8}\n")
        f.write("-"*80 + "\n")
        
        for i, exp in enumerate(sorted_exps, 1):
            f.write(f"{i:<6}{exp['name']:<35}")
            f.write(f"{exp['results']['test_accuracy']:.4f}    ")
            f.write(f"{exp['results']['val_accuracy']:.4f}    ")
            f.write(f"{exp['results']['train_accuracy_at_best_epoch']:.4f}    ")
            f.write(f"{exp['results']['best_epoch']:<8}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*80 + "\n\n")
        
        # Best architecture
        best = sorted_exps[0]
        f.write(f"✓ Best architecture: {best['name']}\n")
        f.write(f"  - {best['num_blocks']} blocks ({best['depth_name']})\n")
        f.write(f"  - {best['width_name']} width: {best['channels']}\n")
        f.write(f"  - Adaptive pooling: {best['pool_size']} ({best['pool_type']})\n")
        f.write(f"  - FC config: {'Hidden (512)' if best['use_fc_hidden'] else 'Direct'}\n")
        f.write(f"  - Test accuracy: {best['results']['test_accuracy']:.4f}\n")
        f.write(f"  - FC parameters: {best['fc_params']:,}\n\n")
        
        # Analyze by depth
        f.write("Average test accuracy by depth:\n")
        for depth in ['shallow', 'moderate', 'baseline']:
            depth_exps = [e for e in sorted_exps if e['depth_name'] == depth]
            if depth_exps:
                avg_acc = sum(e['results']['test_accuracy'] for e in depth_exps) / len(depth_exps)
                f.write(f"  {depth}: {avg_acc:.4f} ({len(depth_exps)} configs)\n")
        f.write("\n")
        
        # Analyze by width
        f.write("Average test accuracy by width:\n")
        for width in ['extra_narrow', 'narrow', 'baseline']:
            width_exps = [e for e in sorted_exps if e['width_name'] == width]
            if width_exps:
                avg_acc = sum(e['results']['test_accuracy'] for e in width_exps) / len(width_exps)
                f.write(f"  {width}: {avg_acc:.4f} ({len(width_exps)} configs)\n")
        f.write("\n")
        
        # Analyze by pooling size
        f.write("Average test accuracy by pooling size:\n")
        for pool_size in [(5,5), (3,3), (1,1)]:
            pool_exps = [e for e in sorted_exps if tuple(e['pool_size']) == pool_size]
            if pool_exps:
                avg_acc = sum(e['results']['test_accuracy'] for e in pool_exps) / len(pool_exps)
                f.write(f"  {pool_size}: {avg_acc:.4f} ({len(pool_exps)} configs)\n")
        f.write("\n")
        
        # Analyze by pooling type
        f.write("Average test accuracy by pooling type:\n")
        for pool_type in ['max', 'avg']:
            type_exps = [e for e in sorted_exps if e['pool_type'] == pool_type]
            if type_exps:
                avg_acc = sum(e['results']['test_accuracy'] for e in type_exps) / len(type_exps)
                f.write(f"  {pool_type}: {avg_acc:.4f} ({len(type_exps)} configs)\n")
        f.write("\n")
        
        # Analyze FC config (only for 1x1 pooling)
        f.write("Average test accuracy by FC config (for GAP 1x1 only):\n")
        gap_exps = [e for e in sorted_exps if tuple(e['pool_size']) == (1,1)]
        gap_fc_hidden = [e for e in gap_exps if e['use_fc_hidden']]
        gap_direct = [e for e in gap_exps if not e['use_fc_hidden']]
        if gap_fc_hidden:
            avg_acc = sum(e['results']['test_accuracy'] for e in gap_fc_hidden) / len(gap_fc_hidden)
            f.write(f"  With hidden (512): {avg_acc:.4f} ({len(gap_fc_hidden)} configs)\n")
        if gap_direct:
            avg_acc = sum(e['results']['test_accuracy'] for e in gap_direct) / len(gap_direct)
            f.write(f"  Direct: {avg_acc:.4f} ({len(gap_direct)} configs)\n")
        f.write("\n")
        
        # Overfitting analysis
        f.write("Top 5 configs by lowest overfitting (train - test gap):\n")
        sorted_by_overfit = sorted(sorted_exps, 
                                   key=lambda x: x['results']['train_accuracy_at_best_epoch'] - x['results']['test_accuracy'])
        for i, exp in enumerate(sorted_by_overfit[:5], 1):
            gap = exp['results']['train_accuracy_at_best_epoch'] - exp['results']['test_accuracy']
            f.write(f"  {i}. {exp['name']}: gap={gap:.4f} (test={exp['results']['test_accuracy']:.4f})\n")
    
    print(f"✓ Summary saved to {summary_file}\n")
    
    # Print top-5 to console
    print("\n" + "="*80)
    print("TOP 5 RESULTS")
    print("="*80)
    for i, exp in enumerate(sorted_exps[:5], 1):
        print(f"#{i} {exp['name']}: Test={exp['results']['test_accuracy']:.4f} "
              f"Val={exp['results']['val_accuracy']:.4f} "
              f"({exp['depth_name']}, {exp['width_name']}, {exp['pool_type']}-{exp['pool_size']})")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Architecture Search V2: Comprehensive Grid Search')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, 
                       default='/data/uabmcv2526/shared/dataset/2425/MIT_small_train_1',
                       help='Root directory of the dataset')
    parser.add_argument('--wandb_project', type=str, default='C3_Week4',
                       help='Wandb project name')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    
    # Output directory
    parser.add_argument('--output_dir', type=str, 
                       default='/data/uabmcv2526/mcvstudent29/Week4/output/arch_search_v2',
                       help='Main experiment directory where all trials will be saved')
    
    args = parser.parse_args()
    
    # Get Week4 directory
    week4_dir = Path(__file__).parent.parent
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("ARCHITECTURE SEARCH V2: COMPREHENSIVE GRID SEARCH")
    print("="*80)
    print("Testing 72 configurations:")
    print("  - 3 depths × 3 widths × 8 pooling configs (both max & avg)")
    print(f"Results will be saved to: {output_dir}\n")
    
    # Generate all experiment configurations
    experiments = generate_experiments()
    
    print("Experiment configurations:")
    for exp in experiments:
        print(f"  {exp['id']:2d}. {exp['name']}")
    print()
    
    # Run all experiments
    for exp in experiments:
        exp = run_single_experiment(exp, args, week4_dir)
        
        # Save intermediate results after each experiment
        with open(output_dir / 'progress.json', 'w') as f:
            json.dump(experiments, f, indent=2)
    
    # Generate final summary
    save_results_summary(experiments, output_dir)
    
    print("\n✓ Architecture search V2 complete!")
    print(f"  Results directory: {output_dir}")
    print(f"  Summary: {output_dir / 'arch_search_v2_summary.txt'}")


if __name__ == '__main__':
    main()
