"""
Adaptive Pooling Regularization Experiment: Spatial Pooling and FC Configuration Search

This experiment tests how different adaptive pooling strategies (after all conv blocks)
affect model regularization and generalization. By varying pooling output sizes and FC 
layer configurations, we can find the optimal balance between model capacity and 
overfitting prevention.

Hypothesis:
    Reducing the spatial dimensions before FC layers (through aggressive pooling)
    will reduce overfitting by limiting the parameter count in FC layers, which
    are the primary bottleneck where overfitting occurs on small datasets.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime

# Add parent directory to path to import from Week4
sys.path.insert(0, str(Path(__file__).parent.parent))


# Pooling strategy configurations
POOLING_CONFIGS = [
    {
        'pool_size': (5, 5),
        'pool_type': 'max',
        'use_fc_hidden': True,
        'fc_hidden': 512,
        'name': 'pool5x5_max_fc512',
        'description': '5x5 max pooling → 3,200 features → 512 → 8'
    },
    {
        'pool_size': (5, 5),
        'pool_type': 'avg',
        'use_fc_hidden': True,
        'fc_hidden': 512,
        'name': 'pool5x5_avg_fc512',
        'description': '5x5 avg pooling → 3,200 features → 512 → 8'
    },
    {
        'pool_size': (3, 3),
        'pool_type': 'max',
        'use_fc_hidden': True,
        'fc_hidden': 512,
        'name': 'pool3x3_max_fc512',
        'description': '3x3 max pooling → 1,152 features → 512 → 8'
    },
    {
        'pool_size': (3, 3),
        'pool_type': 'avg',
        'use_fc_hidden': True,
        'fc_hidden': 512,
        'name': 'pool3x3_avg_fc512',
        'description': '3x3 avg pooling → 1,152 features → 512 → 8'
    },
    {
        'pool_size': (1, 1),
        'pool_type': 'max',
        'use_fc_hidden': True,
        'fc_hidden': 512,
        'name': 'gap_max_fc512',
        'description': 'GAP (1x1) max pooling → 128 features → 512 → 8'
    },
    {
        'pool_size': (1, 1),
        'pool_type': 'avg',
        'use_fc_hidden': True,
        'fc_hidden': 512,
        'name': 'gap_avg_fc512',
        'description': 'GAP (1x1) avg pooling → 128 features → 512 → 8'
    },
    {
        'pool_size': (1, 1),
        'pool_type': 'max',
        'use_fc_hidden': False,
        'fc_hidden': 512,  # Not used but kept for consistency
        'name': 'gap_max_direct',
        'description': 'GAP (1x1) max pooling → 128 features → 8 (direct)'
    },
    {
        'pool_size': (1, 1),
        'pool_type': 'avg',
        'use_fc_hidden': False,
        'fc_hidden': 512,  # Not used but kept for consistency
        'name': 'gap_avg_direct',
        'description': 'GAP (1x1) avg pooling → 128 features → 8 (direct)'
    },
]


def calculate_fc_params(pool_size: tuple, channels: int, fc_hidden: int, 
                       num_classes: int, use_fc_hidden: bool) -> int:
    """Calculate total parameters in FC layers."""
    flattened_size = channels * pool_size[0] * pool_size[1]
    
    if use_fc_hidden:
        fc1_params = flattened_size * fc_hidden + fc_hidden  # weights + bias
        fc2_params = fc_hidden * num_classes + num_classes
        return fc1_params + fc2_params
    else:
        return flattened_size * num_classes + num_classes


def run_single_experiment(config: Dict, args: argparse.Namespace, 
                         week4_dir: Path, exp_num: int, total: int) -> Dict:
    """Run a single training experiment with given pooling configuration."""
    print(f"\n{'='*70}")
    print(f"Experiment {exp_num}/{total}: {config['name']}")
    print(f"  Description: {config['description']}")
    print(f"  Pool size: {config['pool_size']}")
    print(f"  Pool type: {config['pool_type']}")
    print(f"  FC config: {'Hidden layer (512)' if config['use_fc_hidden'] else 'Direct classification'}")
    
    # Calculate FC parameters
    fc_params = calculate_fc_params(
        config['pool_size'], 
        128,  # Last conv channel from baseline_narrow
        config['fc_hidden'],
        8,  # MIT scenes classes
        config['use_fc_hidden']
    )
    print(f"  FC parameters: {fc_params:,}")
    print(f"{'='*70}\n")
    
    # Build command
    cmd = [
        'python', str(week4_dir / 'main.py'),
        '--data_root', args.data_root,
        '--output_dir', args.output_dir,  # Parent dir, main.py will add experiment_name
        '--wandb_project', args.wandb_project,
        '--experiment_name', config['name'],
        '--model_type', 'flexible',
        '--channels', '16,32,64,128',  # baseline_narrow from arch_search
        '--fc_hidden', str(config['fc_hidden']),
        '--kernel_size', '3',
        '--pooling_type', 'max',  # Conv block pooling (inside blocks)
        '--pool_output_size', f"{config['pool_size'][0]},{config['pool_size'][1]}",
        '--adaptive_pool_type', config['pool_type'],  # Adaptive pooling type (max/avg)
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
    if config['use_fc_hidden']:
        cmd.append('--use_fc_hidden')
    else:
        cmd.append('--no_fc_hidden')
    
    start_time = time.time()
    
    try:
        # Run training (output goes directly to stdout/stderr so we can see progress)
        result = subprocess.run(cmd, check=True)
        success = True
        error_msg = None
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = str(e)
        print(f"ERROR: Experiment {config['name']} failed!")
        print(f"Error: {error_msg}")
    
    training_time = time.time() - start_time
    
    # Parse results from training summary
    results = {
        'name': config['name'],
        'pool_size': config['pool_size'],
        'pool_type': config['pool_type'],
        'use_fc_hidden': config['use_fc_hidden'],
        'fc_hidden': config['fc_hidden'] if config['use_fc_hidden'] else None,
        'fc_params': fc_params,
        'description': config['description'],
        'success': success,
        'training_time_minutes': training_time / 60,
        'val_accuracy': None,
        'test_accuracy': None,
        'train_accuracy_at_best_epoch': None,
        'best_epoch': None,
        'error': error_msg
    }
    
    if success:
        # Try to parse results from the output
        summary_pattern = os.path.join(args.output_dir, config['name'], 
                                      f"{config['name']}_*", "training_summary.txt")
        import glob
        summary_files = glob.glob(summary_pattern)
        
        if summary_files:
            print(f"  Found summary file: {summary_files[0]}")
            try:
                with open(summary_files[0], 'r') as f:
                    content = f.read()
                    # Parse validation accuracy, test accuracy, train accuracy at best epoch
                    for line in content.split('\n'):
                        if 'Best Validation Accuracy:' in line:
                            val_str = line.split(':')[1].strip().replace('%', '')
                            results['val_accuracy'] = float(val_str)
                        elif 'Best Test Accuracy:' in line:
                            test_str = line.split(':')[1].strip().replace('%', '')
                            results['test_accuracy'] = float(test_str)
                        elif 'Best Epoch:' in line:
                            results['best_epoch'] = int(line.split(':')[1].strip())
                        elif 'Training Accuracy at Best Epoch:' in line:
                            train_str = line.split(':')[1].strip().replace('%', '')
                            results['train_accuracy_at_best_epoch'] = float(train_str)
            except Exception as e:
                print(f"Warning: Could not parse results from summary file: {e}")
        else:
            print(f"  WARNING: No summary file found matching pattern: {summary_pattern}")
    
    return results


def save_results_summary(all_results: List[Dict], output_dir: Path):
    """Save comprehensive summary of all experiments."""
    
    # Save JSON
    json_path = output_dir / "adaptive_pooling_regularization_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")
    
    # Save text summary
    txt_path = output_dir / "adaptive_pooling_regularization_summary.txt"
    
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ADAPTIVE POOLING REGULARIZATION: SPATIAL POOLING & FC CONFIG SEARCH\n")
        f.write("="*70 + "\n\n")
        
        f.write("Hypothesis:\n")
        f.write("  Reducing spatial dimensions before FC layers (through aggressive pooling)\n")
        f.write("  will reduce overfitting by limiting FC parameter count, which is the\n")
        f.write("  primary bottleneck where overfitting occurs on small datasets.\n\n")
        
        f.write("Experiment Design:\n")
        f.write("  - Fixed architecture: baseline_narrow [16, 32, 64, 128]\n")
        f.write("  - Variable: Adaptive pooling output size (5x5, 3x3, 1x1)\n")
        f.write("  - Variable: Pooling type (max, avg)\n")
        f.write("  - Variable: FC configuration (512 hidden, direct)\n")
        f.write("  - Total configurations: 8\n\n")
        
        # Count successes
        successful = [r for r in all_results if r['success']]
        f.write(f"Total experiments: {len(all_results)}\n")
        f.write(f"Successful experiments: {len(successful)}\n\n")
        
        # Sort by test accuracy
        sorted_results = sorted(successful, key=lambda x: x['test_accuracy'] or 0, reverse=True)
        
        f.write("="*70 + "\n")
        f.write("TOP 3 CONFIGURATIONS\n")
        f.write("="*70 + "\n\n")
        
        for i, result in enumerate(sorted_results[:3], 1):
            f.write(f"#{i} - {result['name']}\n")
            f.write(f"  Configuration:\n")
            f.write(f"    Pool size: {result['pool_size']}\n")
            f.write(f"    Pool type: {result['pool_type']}\n")
            f.write(f"    FC config: {'Hidden (512)' if result['use_fc_hidden'] else 'Direct'}\n")
            f.write(f"  Results:\n")
            f.write(f"    Test Accuracy: {result['test_accuracy']:.2f}%\n")
            f.write(f"    Validation Accuracy: {result['val_accuracy']:.2f}%\n")
            f.write(f"    Best Epoch: {result['best_epoch']}\n")
            f.write(f"    FC Parameters: {result['fc_params']:,}\n")
            f.write(f"    Training Time: {result['training_time_minutes']:.1f} min\n\n")
        
        f.write("="*70 + "\n")
        f.write("ALL RESULTS (sorted by test accuracy)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'Rank':<6}{'Name':<25}{'Pool':<10}{'Type':<6}{'FC':<8}{'Test Acc':<10}{'FC Params':<12}{'Epoch':<8}\n")
        f.write("-"*70 + "\n")
        
        for i, result in enumerate(sorted_results, 1):
            pool_str = f"{result['pool_size'][0]}x{result['pool_size'][1]}"
            fc_str = 'FC512' if result['use_fc_hidden'] else 'Direct'
            f.write(f"{i:<6}{result['name']:<25}{pool_str:<10}{result['pool_type']:<6}{fc_str:<8}")
            f.write(f"{result['test_accuracy']:.2f}%    {result['fc_params']:>10,}  {result['best_epoch']:<8}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*70 + "\n\n")
        
        # Best configuration
        best = sorted_results[0]
        f.write(f"✓ Best configuration: {best['name']}\n")
        f.write(f"  - Pool size: {best['pool_size']}\n")
        f.write(f"  - Pool type: {best['pool_type']}\n")
        f.write(f"  - FC config: {'Hidden layer (512)' if best['use_fc_hidden'] else 'Direct classification'}\n")
        f.write(f"  - Test accuracy: {best['test_accuracy']:.2f}%\n")
        f.write(f"  - FC parameters: {best['fc_params']:,}\n\n")
        
        # Average by pool size
        f.write("Average test accuracy by pool size:\n")
        for pool_size in [(5, 5), (3, 3), (1, 1)]:
            pool_results = [r for r in sorted_results if r['pool_size'] == pool_size]
            if pool_results:
                avg_acc = sum(r['test_accuracy'] for r in pool_results) / len(pool_results)
                f.write(f"  - {pool_size[0]}x{pool_size[1]}: {avg_acc:.2f}% (avg of {len(pool_results)} configs)\n")
        
        # Average by pool type
        f.write("\nAverage test accuracy by pooling type:\n")
        for pool_type in ['max', 'avg']:
            type_results = [r for r in sorted_results if r['pool_type'] == pool_type]
            if type_results:
                avg_acc = sum(r['test_accuracy'] for r in type_results) / len(type_results)
                f.write(f"  - {pool_type}: {avg_acc:.2f}% (avg of {len(type_results)} configs)\n")
        
        # Average by FC config
        f.write("\nAverage test accuracy by FC configuration:\n")
        for use_fc in [True, False]:
            fc_results = [r for r in sorted_results if r['use_fc_hidden'] == use_fc]
            if fc_results:
                avg_acc = sum(r['test_accuracy'] for r in fc_results) / len(fc_results)
                fc_label = 'Hidden layer (512)' if use_fc else 'Direct classification'
                f.write(f"  - {fc_label}: {avg_acc:.2f}% (avg of {len(fc_results)} configs)\n")
    
    print(f"✓ Summary saved to {txt_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Pooling Regularization Search: Find optimal spatial pooling strategy"
    )
    
    # Paths
    parser.add_argument('--data_root', type=str, 
                       default='/data/uabmcv2526/shared/dataset/2425/MIT_small_train_1',
                       help='Dataset root directory')
    parser.add_argument('--output_dir', type=str,
                       default='/data/uabmcv2526/mcvstudent29/Week4/output/adaptive_pooling_regularization',
                       help='Output directory for all experiments')
    parser.add_argument('--wandb_project', type=str, default='C3_Week4',
                       help='Wandb project name')
    
    # Training hyperparameters (matching best from arch_search)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()
    
    # Setup paths
    week4_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("ADAPTIVE POOLING REGULARIZATION SEARCH")
    print("="*70)
    print(f"Testing {len(POOLING_CONFIGS)} pooling configurations")
    print(f"Output directory: {output_dir}")
    print(f"Architecture: baseline_narrow [16, 32, 64, 128]")
    print("="*70 + "\n")
    
    # Run all experiments
    all_results = []
    start_time = time.time()
    
    for i, config in enumerate(POOLING_CONFIGS, 1):
        result = run_single_experiment(config, args, week4_dir, i, len(POOLING_CONFIGS))
        all_results.append(result)
        
        # Save progress after each experiment
        progress_file = output_dir / 'progress.json'
        with open(progress_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Progress saved to {progress_file}")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {sum(1 for r in all_results if r['success'])}/{len(all_results)}")
    print("="*70 + "\n")
    
    # Save results
    save_results_summary(all_results, output_dir)
    
    print("Experiment complete! Check output directory for detailed results.\n")


if __name__ == '__main__':
    main()
