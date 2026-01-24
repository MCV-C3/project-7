"""
Hyperparameter Optimization using Optuna for Week 4 CNN from Scratch

This script optimizes the following hyperparameters:
- Batch size
- Number of epochs
- Optimizer (SGD, Adam, AdamW)
- Learning rate
- Momentum (for SGD)
- Weight decay
- Dropout

Multi-objective optimization:
1. Maximize test accuracy
2. Minimize overfitting (difference between train and test accuracy)

Note: Uses test set for evaluation (no validation set)
      Trains directly on full train set and evaluates on test set

Sampler: TPE (Tree-structured Parzen Estimator) for efficient search
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
import optuna
from optuna.trial import Trial
import wandb
from typing import Dict, Tuple, List

# Add parent directory to path to import from Week4
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_transforms
from attention_models import CBAMOptimizedCNN
from augmented_subset import AugmentedSubset


# ==================== Training Functions ====================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 0.0
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        avg_loss: Average loss over the epoch
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (if enabled)
        if max_grad_norm and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model.
    
    Args:
        model: PyTorch model
        dataloader: Evaluation data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        avg_loss: Average evaluation loss
        accuracy: Evaluation accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy



def train_and_evaluate_on_test(
    train_dataset_path: str,
    test_dataset: ImageFolder,
    batch_size: int,
    epochs: int,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    dropout: float,
    device: torch.device,
    max_grad_norm: float = 0.0
) -> Dict[str, float]:
    """
    Train on full train set with fixed data augmentation and evaluate on test set.
    
    Args:
        train_dataset_path: Path to training dataset directory
        test_dataset: Test dataset
        batch_size: Batch size for data loaders
        epochs: Number of training epochs
        optimizer_name: Name of the optimizer
        learning_rate: Learning rate
        momentum: Momentum (only for SGD)
        weight_decay: Weight decay
        dropout: Dropout probability
        device: Device to train on
        
    Returns:
        Dictionary with train and test metrics
    """
    # Create base training dataset
    base_train_dataset = ImageFolder(
        root=train_dataset_path,
        transform=build_transforms(train=True)
    )
    
    # Create augmented training dataset with fixed augmentation parameters
    aug_ratio = 1.5
    augmented_size = int(len(base_train_dataset) * aug_ratio)
    augmented_dataset = ImageFolder(
        root=train_dataset_path,
        transform=build_transforms(
            train=True,
            use_flip=True,
            use_color=True,
            use_geometric=True,
            use_translation=True,
        )
    )
    
    # Combine base and augmented datasets
    train_datasets = [
        base_train_dataset,
        AugmentedSubset(augmented_dataset, augmented_size)
    ]
    combined_train_dataset = ConcatDataset(train_datasets)
    
    # Create data loaders
    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model with CBAM attention
    model = CBAMOptimizedCNN(
        num_classes=8,
        input_channels=3,
        dropout=dropout,
        reduction=4,
        spatial_kernel=5,
        spatial_dilation=1,
        num_cbam_blocks=1
    )
    model = model.to(device)
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    # Training loop - track best test accuracy
    best_test_acc = 0.0
    final_train_acc = 0.0
    final_test_acc = 0.0
    final_test_loss = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_grad_norm
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        # Track best test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            final_train_acc = train_acc
            final_test_acc = test_acc
            final_test_loss = test_loss
    
    return {
        'train_acc': final_train_acc,
        'test_acc': final_test_acc,
        'test_loss': final_test_loss,
        'overfitting': final_train_acc - final_test_acc
    }


# ==================== Objective Function ====================

def objective(
    trial: Trial,
    train_dataset_path: str,
    test_dataset: ImageFolder,
    device: torch.device
) -> Tuple[float, float]:
    """
    Optuna objective function for multi-objective optimization.
    
    This function suggests hyperparameters, trains on train set with fixed data augmentation,
    and evaluates on test set to return two objectives.
    
    Args:
        trial: Optuna trial object
        train_dataset_path: Path to training dataset directory
        test_dataset: Test dataset
        device: Device to train on
        
    Returns:
        Tuple of (test_accuracy, overfitting) for multi-objective optimization
        - First objective: maximize test accuracy
        - Second objective: minimize overfitting
    """
    # Suggest hyperparameters (tuned for small dataset: 400 samples, 8 classes)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])  # Smaller batches for 400 samples
    epochs = trial.suggest_int('epochs', 20, 50, step=10)  # More epochs for small batches
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)  # Conservative upper bound
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)  # Stronger regularization range
    dropout = trial.suggest_float('dropout', 0.2, 0.6)  # Higher dropout for small dataset
    # Gradient clipping strength (0 disables clipping)
    max_grad_norm = trial.suggest_categorical('max_grad_norm', [0.0, 0.5, 1.0, 2.0])
    
    # Momentum only relevant for SGD
    momentum = trial.suggest_float('momentum', 0.0, 0.99) if optimizer_name == 'SGD' else 0.0
    
    print(f"\nTrial {trial.number}:")
    print(f"  batch_size={batch_size}, epochs={epochs}, optimizer={optimizer_name}")
    print(f"  learning_rate={learning_rate:.6f}, momentum={momentum:.3f}, "
          f"weight_decay={weight_decay:.6f}, dropout={dropout:.3f}, max_grad_norm={max_grad_norm}")
    print(f"  Fixed data augmentation: flip=True, color=True, geometric=True, translation=True, aug_ratio=1.5")
    
    # Train on full train set with fixed data augmentation and evaluate on test set
    print(f"\n  Training on train set with data augmentation and evaluating on test set")
    test_results = train_and_evaluate_on_test(
        train_dataset_path=train_dataset_path,
        test_dataset=test_dataset,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        dropout=dropout,
        device=device,
        max_grad_norm=max_grad_norm
    )
    
    # Multi-objective optimization:
    # Objective 1: Maximize test accuracy
    # Objective 2: Minimize overfitting (train-test gap)
    test_acc = test_results['test_acc']
    overfitting = test_results['overfitting']
    
    print(f"\nTrial {trial.number} Results:")
    print(f"  Final Train Acc: {test_results['train_acc']:.4f}")
    print(f"  Final Test Acc: {test_results['test_acc']:.4f}")
    print(f"  Final Overfitting: {overfitting:.4f}")
    
    # Return tuple: (objective1, objective2)
    # Optuna will maximize test_acc and minimize overfitting
    return test_acc, overfitting


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization for Week 4 CNN with Optuna'
    )
    
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='/data/uabmcv2526/shared/dataset/2425/MIT_small_train_1',
        help='Path to dataset root directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./hyperopt_results',
        help='Directory to save optimization results'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=30,
        help='Number of Optuna trials to run'
    )
    
    parser.add_argument(
        '--study_name',
        type=str,
        default='week4_hyperopt',
        help='Name for the Optuna study'
    )
    
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='C3_Week4_HyperOpt',
        help='Weights & Biases project name'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    test_transform = build_transforms(train=False)
    
    train_dataset_path = os.path.join(args.dataset_root, 'train')
    test_dataset_path = os.path.join(args.dataset_root, 'test')
    
    # Load test dataset for evaluation
    test_dataset = ImageFolder(root=test_dataset_path, transform=test_transform)
    
    # Check train dataset size for info
    temp_train_dataset = ImageFolder(root=train_dataset_path, transform=build_transforms(train=True))
    base_train_size = len(temp_train_dataset)
    aug_ratio = 1.5
    augmented_size = int(base_train_size * aug_ratio)
    total_train_size = base_train_size + augmented_size
    
    print(f"Base training dataset: {base_train_size} samples from {train_dataset_path}")
    print(f"Augmented samples: {augmented_size} (ratio: {aug_ratio})")
    print(f"Total training samples: {total_train_size}")
    print(f"Loaded test dataset with {len(test_dataset)} samples from {test_dataset_path}")
    
    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        name=args.study_name,
        config={
            'n_trials': args.n_trials,
            'model': 'CBAMOptimizedCNN',
            'cbam_reduction': 4,
            'cbam_spatial_kernel': 5,
            'cbam_dilation': 1,
            'cbam_num_blocks': 1,
            'use_flip': True,
            'use_color': True,
            'use_geometric': True,
            'use_translation': True,
            'aug_ratio': aug_ratio,
            'base_train_size': base_train_size,
            'total_train_size': total_train_size,
        }
    )
    
    # Create Optuna study for multi-objective optimization
    study = optuna.create_study(
        study_name=args.study_name,
        directions=['maximize', 'minimize'],  # Maximize test_acc, minimize overfitting
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"\nStarting Optuna optimization with {args.n_trials} trials...")
    print("=" * 70)
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, train_dataset_path, test_dataset, device),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    # For multi-objective optimization, show Pareto front
    print(f"\nNumber of trials: {len(study.trials)}")
    print(f"Number of Pareto optimal solutions: {len(study.best_trials)}")
    
    print("\nPareto Front (Top 5 solutions):")
    for i, trial in enumerate(study.best_trials[:5]):
        print(f"\n  Trial {trial.number}:")
        print(f"    Test Accuracy: {trial.values[0]:.4f}")
        print(f"    Overfitting: {trial.values[1]:.4f}")
        print(f"    Hyperparameters:")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")
    
    # Save results in text format
    results_path = os.path.join(args.output_dir, 'pareto_front_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Multi-Objective Optimization Results - Week 4 CNN\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Number of Pareto optimal solutions: {len(study.best_trials)}\n\n")
        f.write("Pareto Front Solutions:\n")
        f.write("=" * 70 + "\n\n")
        for i, trial in enumerate(study.best_trials):
            f.write(f"Solution {i+1} (Trial {trial.number}):\n")
            f.write(f"  Test Accuracy: {trial.values[0]:.4f}\n")
            f.write(f"  Overfitting: {trial.values[1]:.4f}\n")
            f.write(f"  Hyperparameters:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")
    
    # Save Pareto front in JSON format
    pareto_solutions = []
    for trial in study.best_trials:
        solution = {
            'trial_number': trial.number,
            'test_accuracy': float(trial.values[0]),
            'overfitting': float(trial.values[1]),
            'hyperparameters': trial.params
        }
        pareto_solutions.append(solution)
    
    json_path = os.path.join(args.output_dir, 'pareto_front.json')
    with open(json_path, 'w') as f:
        json.dump(pareto_solutions, f, indent=2)
    
    # Identify and save recommended configurations
    if len(study.best_trials) > 0:
        # Best accuracy solution
        best_acc_trial = max(study.best_trials, key=lambda t: t.values[0])
        
        # Best overfitting solution (lowest overfitting)
        best_overfit_trial = min(study.best_trials, key=lambda t: t.values[1])
        
        # Best balanced solution
        acc_values = [t.values[0] for t in study.best_trials]
        overfit_values = [t.values[1] for t in study.best_trials]
        
        acc_min, acc_max = min(acc_values), max(acc_values)
        overfit_min, overfit_max = min(overfit_values), max(overfit_values)
        
        # Normalize and find solution closest to ideal (high acc, low overfit)
        best_distance = float('inf')
        best_balanced_trial = study.best_trials[0]
        
        for trial in study.best_trials:
            norm_acc = (trial.values[0] - acc_min) / (acc_max - acc_min + 1e-8)
            norm_overfit = (trial.values[1] - overfit_min) / (overfit_max - overfit_min + 1e-8)
            # Distance from ideal point (1, 0) in normalized space
            distance = np.sqrt((1 - norm_acc)**2 + norm_overfit**2)
            if distance < best_distance:
                best_distance = distance
                best_balanced_trial = trial
        
        recommendations = {
            'best_accuracy': {
                'trial_number': best_acc_trial.number,
                'test_accuracy': float(best_acc_trial.values[0]),
                'overfitting': float(best_acc_trial.values[1]),
                'hyperparameters': best_acc_trial.params,
                'description': 'Highest test accuracy (may have more overfitting)'
            },
            'best_overfitting': {
                'trial_number': best_overfit_trial.number,
                'test_accuracy': float(best_overfit_trial.values[0]),
                'overfitting': float(best_overfit_trial.values[1]),
                'hyperparameters': best_overfit_trial.params,
                'description': 'Lowest overfitting (may have lower accuracy)'
            },
            'recommended_balanced': {
                'trial_number': best_balanced_trial.number,
                'test_accuracy': float(best_balanced_trial.values[0]),
                'overfitting': float(best_balanced_trial.values[1]),
                'hyperparameters': best_balanced_trial.params,
                'description': 'Best balance between accuracy and overfitting'
            }
        }
        
        recommendations_path = os.path.join(args.output_dir, 'recommended_configs.json')
        with open(recommendations_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  - {results_path} (text format)")
        print(f"  - {json_path} (JSON format)")
        print(f"  - {recommendations_path} (recommended configs)")
        
        print(f"\nRECOMMENDED CONFIGURATIONS:")
        print(f"\n1. Best Test Accuracy (Trial {best_acc_trial.number}):")
        print(f"   Test Acc: {best_acc_trial.values[0]:.4f}, Overfitting: {best_acc_trial.values[1]:.4f}")
        for key, value in best_acc_trial.params.items():
            print(f"   {key}: {value}")
        
        print(f"\n2. Lowest Overfitting (Trial {best_overfit_trial.number}):")
        print(f"   Test Acc: {best_overfit_trial.values[0]:.4f}, Overfitting: {best_overfit_trial.values[1]:.4f}")
        for key, value in best_overfit_trial.params.items():
            print(f"   {key}: {value}")
        
        print(f"\n3. RECOMMENDED - Best Balance (Trial {best_balanced_trial.number}):")
        print(f"   Test Acc: {best_balanced_trial.values[0]:.4f}, Overfitting: {best_balanced_trial.values[1]:.4f}")
        for key, value in best_balanced_trial.params.items():
            print(f"   {key}: {value}")
    else:
        print(f"\nResults saved to: {results_path}")
    
    # Log to wandb
    wandb.log({
        'n_trials': len(study.trials),
        'n_pareto_solutions': len(study.best_trials)
    })
    
    # Log best trials from Pareto front
    for i, trial in enumerate(study.best_trials[:5]):
        wandb.log({
            f'pareto_{i}_test_acc': trial.values[0],
            f'pareto_{i}_overfitting': trial.values[1]
        })
    
    # Create optimization visualizations
    try:
        # Pareto front visualization
        fig = optuna.visualization.plot_pareto_front(
            study,
            target_names=['Test Accuracy', 'Overfitting']
        )
        fig.write_image(os.path.join(args.output_dir, 'pareto_front.png'))
        
        # Parameter importances for each objective
        fig = optuna.visualization.plot_param_importances(
            study,
            target=lambda t: t.values[0],
            target_name='Test Accuracy'
        )
        fig.write_image(os.path.join(args.output_dir, 'param_importances_test_acc.png'))
        
        fig = optuna.visualization.plot_param_importances(
            study,
            target=lambda t: t.values[1],
            target_name='Overfitting'
        )
        fig.write_image(os.path.join(args.output_dir, 'param_importances_overfitting.png'))
        
        print(f"Visualization plots saved to: {args.output_dir}")
    except Exception as e:
        print(f"Could not save visualization plots: {e}")
    
    wandb.finish()
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
