"""
Hyperparameter Optimization using Optuna with 3-Fold Cross-Validation

This script optimizes the following hyperparameters:
- Batch size
- Number of epochs
- Optimizer (SGD, Adam, AdamW)
- Learning rate
- Momentum (for SGD)
- Weight decay

Fixed parameters (from previous experiments):
- unfreeze_blocks = 7
- use_batchnorm_blocks = True

Multi-objective optimization:
1. Maximize validation accuracy (from best epoch, not final epoch)
2. Minimize overfitting (difference between train and validation accuracy)

Sampler: TPE (Tree-structured Parzen Estimator) for efficient search
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as F
from sklearn.model_selection import KFold
import optuna
from optuna.trial import Trial
import wandb
from typing import Dict, Tuple, List

from models import WraperModel


# ==================== Training Functions ====================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
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
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
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


# ==================== Cross-Validation ====================

def cross_validate(
    trial: Trial,
    dataset: ImageFolder,
    batch_size: int,
    epochs: int,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
    n_folds: int = 3
) -> Dict[str, float]:
    """
    Perform k-fold cross-validation.
    
    Args:
        trial: Optuna trial object
        dataset: Full training dataset
        batch_size: Batch size for data loaders
        epochs: Number of training epochs
        optimizer_name: Name of the optimizer ('SGD', 'Adam', 'AdamW')
        learning_rate: Learning rate
        momentum: Momentum (only for SGD)
        weight_decay: Weight decay for regularization
        device: Device to train on
        n_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation metrics
        
    Note:
        The validation accuracy is taken from the epoch with the best validation
        accuracy, not the final epoch.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_train_accs = []
    fold_val_accs = []
    fold_val_losses = []
    fold_overfitting = []
    
    indices = list(range(len(dataset)))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n  Fold {fold + 1}/{n_folds}")
        
        # Create data subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        model = WraperModel(
            num_classes=8,
            unfreeze_blocks=7,
            dropout_blocks=0,
            dropout_value=0.5,
            use_batchnorm_blocks=True
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
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Training loop
        # We track the BEST validation accuracy across all epochs,
        # not the accuracy from the final epoch. This gives a fairer
        # comparison between hyperparameter configurations.
        best_val_acc = 0.0
        final_train_acc = 0.0
        final_val_acc = 0.0
        final_val_loss = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion, device
            )
            
            # Track best validation accuracy and corresponding train accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_train_acc = train_acc
                final_val_acc = val_acc
                final_val_loss = val_loss
            
            # Report intermediate value for pruning
            trial.report(val_acc, fold * epochs + epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Store fold results
        fold_train_accs.append(final_train_acc)
        fold_val_accs.append(final_val_acc)
        fold_val_losses.append(final_val_loss)
        fold_overfitting.append(final_train_acc - final_val_acc)
        
        print(f"  Fold {fold + 1} - Train Acc: {final_train_acc:.4f}, "
              f"Val Acc: {final_val_acc:.4f}, Overfitting: {final_train_acc - final_val_acc:.4f}")
    
    # Calculate average metrics across folds
    results = {
        'mean_train_acc': np.mean(fold_train_accs),
        'mean_val_acc': np.mean(fold_val_accs),
        'mean_val_loss': np.mean(fold_val_losses),
        'mean_overfitting': np.mean(fold_overfitting),
        'std_val_acc': np.std(fold_val_accs),
        'std_overfitting': np.std(fold_overfitting)
    }
    
    return results


# ==================== Objective Function ====================

def objective(trial: Trial, dataset: ImageFolder, device: torch.device) -> Tuple[float, float]:
    """
    Optuna objective function for multi-objective optimization.
    
    This function suggests hyperparameters, performs cross-validation,
    and returns two objectives to optimize simultaneously.
    
    Args:
        trial: Optuna trial object
        dataset: Training dataset
        device: Device to train on
        
    Returns:
        Tuple of (val_accuracy, -overfitting) for multi-objective optimization
        - First objective: maximize validation accuracy
        - Second objective: minimize overfitting (returned as negative for maximization)
    """
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 30, step=5)
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Momentum only relevant for SGD
    momentum = trial.suggest_float('momentum', 0.0, 0.99) if optimizer_name == 'SGD' else 0.0
    
    print(f"\nTrial {trial.number}:")
    print(f"  batch_size={batch_size}, epochs={epochs}, optimizer={optimizer_name}")
    print(f"  learning_rate={learning_rate:.6f}, momentum={momentum:.3f}, weight_decay={weight_decay:.6f}")
    
    # Perform cross-validation
    results = cross_validate(
        trial=trial,
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        device=device,
        n_folds=3
    )
    
    # Multi-objective optimization:
    # Objective 1: Maximize validation accuracy
    # Objective 2: Minimize overfitting (train-val gap)
    val_acc = results['mean_val_acc']
    overfitting = results['mean_overfitting']
    
    print(f"\nTrial {trial.number} Results:")
    print(f"  Mean Train Acc: {results['mean_train_acc']:.4f}")
    print(f"  Mean Val Acc: {results['mean_val_acc']:.4f} (±{results['std_val_acc']:.4f})")
    print(f"  Mean Overfitting: {results['mean_overfitting']:.4f} (±{results['std_overfitting']:.4f})")
    
    # Return tuple: (objective1, objective2)
    # Optuna will maximize both, so return negative overfitting to minimize it
    return val_acc, -overfitting


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization with Optuna and 3-fold CV'
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
        default='/data/uabmcv2526/mcvstudent29/output/hyperopt/',
        help='Directory to save optimization results'
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of Optuna trials to run'
    )
    
    parser.add_argument(
        '--study_name',
        type=str,
        default='week3_hyperopt',
        help='Name for the Optuna study'
    )
    
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='C3_Week3_HyperOpt',
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
    
    # Load dataset (only training set for cross-validation)
    transform = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(224, 224)),
    ])
    
    dataset_path = os.path.join(args.dataset_root, 'train')
    dataset = ImageFolder(root=dataset_path, transform=transform)
    print(f"Loaded dataset with {len(dataset)} samples from {dataset_path}")
    
    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        name=args.study_name,
        config={
            'n_trials': args.n_trials,
            'n_folds': 3,
            'unfreeze_blocks': 7,
            'use_batchnorm_blocks': True
        }
    )
    
    # Create Optuna study for multi-objective optimization
    # Using TPE (Tree-structured Parzen Estimator) sampler for efficient hyperparameter search
    study = optuna.create_study(
        study_name=args.study_name,
        directions=['maximize', 'maximize'],  # Maximize val_acc, maximize -overfitting (i.e., minimize overfitting)
        sampler=optuna.samplers.TPESampler(seed=42),  # TPE sampler with fixed seed
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
    )
    
    print(f"\nStarting Optuna optimization with {args.n_trials} trials...")
    print("=" * 70)
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, dataset, device),
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
        print(f"    Val Accuracy: {trial.values[0]:.4f}")
        print(f"    Overfitting: {-trial.values[1]:.4f}")  # Convert back to positive
        print(f"    Hyperparameters:")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")
    
    # Save results in text format
    results_path = os.path.join(args.output_dir, 'pareto_front_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Multi-Objective Optimization Results\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Number of Pareto optimal solutions: {len(study.best_trials)}\n\n")
        f.write("Pareto Front Solutions:\n")
        f.write("=" * 70 + "\n\n")
        for i, trial in enumerate(study.best_trials):
            f.write(f"Solution {i+1} (Trial {trial.number}):\n")
            f.write(f"  Val Accuracy: {trial.values[0]:.4f}\n")
            f.write(f"  Overfitting: {-trial.values[1]:.4f}\n")
            f.write(f"  Hyperparameters:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")
    
    # Save Pareto front in JSON format (easier to load programmatically)
    pareto_solutions = []
    for trial in study.best_trials:
        solution = {
            'trial_number': trial.number,
            'val_accuracy': float(trial.values[0]),
            'overfitting': float(-trial.values[1]),
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
        best_overfit_trial = max(study.best_trials, key=lambda t: t.values[1])
        
        # Best balanced solution (normalize and find closest to ideal point)
        acc_values = [t.values[0] for t in study.best_trials]
        overfit_values = [-t.values[1] for t in study.best_trials]  # Convert to positive
        
        acc_min, acc_max = min(acc_values), max(acc_values)
        overfit_min, overfit_max = min(overfit_values), max(overfit_values)
        
        # Normalize and find solution closest to ideal (high acc, low overfit)
        best_distance = float('inf')
        best_balanced_trial = study.best_trials[0]
        
        for trial in study.best_trials:
            norm_acc = (trial.values[0] - acc_min) / (acc_max - acc_min + 1e-8)
            norm_overfit = (-trial.values[1] - overfit_min) / (overfit_max - overfit_min + 1e-8)
            # Distance from ideal point (1, 0) in normalized space
            distance = np.sqrt((1 - norm_acc)**2 + norm_overfit**2)
            if distance < best_distance:
                best_distance = distance
                best_balanced_trial = trial
        
        recommendations = {
            'best_accuracy': {
                'trial_number': best_acc_trial.number,
                'val_accuracy': float(best_acc_trial.values[0]),
                'overfitting': float(-best_acc_trial.values[1]),
                'hyperparameters': best_acc_trial.params,
                'description': 'Highest validation accuracy (may have more overfitting)'
            },
            'best_overfitting': {
                'trial_number': best_overfit_trial.number,
                'val_accuracy': float(best_overfit_trial.values[0]),
                'overfitting': float(-best_overfit_trial.values[1]),
                'hyperparameters': best_overfit_trial.params,
                'description': 'Lowest overfitting (may have lower accuracy)'
            },
            'recommended_balanced': {
                'trial_number': best_balanced_trial.number,
                'val_accuracy': float(best_balanced_trial.values[0]),
                'overfitting': float(-best_balanced_trial.values[1]),
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
        print(f"\n1. Best Accuracy (Trial {best_acc_trial.number}):")
        print(f"   Val Acc: {best_acc_trial.values[0]:.4f}, Overfitting: {-best_acc_trial.values[1]:.4f}")
        for key, value in best_acc_trial.params.items():
            print(f"   {key}: {value}")
        
        print(f"\n2. Lowest Overfitting (Trial {best_overfit_trial.number}):")
        print(f"   Val Acc: {best_overfit_trial.values[0]:.4f}, Overfitting: {-best_overfit_trial.values[1]:.4f}")
        for key, value in best_overfit_trial.params.items():
            print(f"   {key}: {value}")
        
        print(f"\n3. RECOMMENDED - Best Balance (Trial {best_balanced_trial.number}):")
        print(f"   Val Acc: {best_balanced_trial.values[0]:.4f}, Overfitting: {-best_balanced_trial.values[1]:.4f}")
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
        wandb.log({f'pareto_{i}_val_acc': trial.values[0],
                   f'pareto_{i}_overfitting': -trial.values[1]})
    
    # Create optimization visualizations
    try:
        # Pareto front visualization
        fig = optuna.visualization.plot_pareto_front(study, target_names=['Val Accuracy', 'Neg Overfitting'])
        fig.write_image(os.path.join(args.output_dir, 'pareto_front.png'))
        
        # Parameter importances for each objective
        fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name='Val Accuracy')
        fig.write_image(os.path.join(args.output_dir, 'param_importances_val_acc.png'))
        
        fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1], target_name='Neg Overfitting')
        fig.write_image(os.path.join(args.output_dir, 'param_importances_overfitting.png'))
        
        print(f"Visualization plots saved to: {args.output_dir}")
    except Exception as e:
        print(f"Could not save visualization plots: {e}")
    
    wandb.finish()
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
