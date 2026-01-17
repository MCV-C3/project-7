import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import tqdm
import wandb
from datetime import datetime

from models import SimpleCNN, build_transforms
from helpers import plot_metrics, save_training_summary, print_model_summary, save_model_architecture, plot_confusion_matrix


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        avg_loss (float): Average training loss
        accuracy (float): Training accuracy
    """
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def test_epoch(model, dataloader, criterion, device, return_predictions=False):
    """
    Evaluate the model on the test set.
    
    Args:
        model: PyTorch model
        dataloader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        return_predictions: If True, return all predictions and labels
        
    Returns:
        avg_loss (float): Average test loss
        accuracy (float): Test accuracy
        predictions (list): All predictions (if return_predictions=True)
        labels_list (list): All true labels (if return_predictions=True)
    """
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = test_loss / total
    accuracy = correct / total
    
    if return_predictions:
        return avg_loss, accuracy, all_predictions, all_labels
    return avg_loss, accuracy


def main(args):
    """Main training function."""
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{args.experiment_name}_{timestamp}" if args.add_timestamp else args.experiment_name
    output_dir = os.path.join(args.output_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("WEEK 4 - CNN FROM SCRATCH TRAINING")
    print("=" * 70)
    print(f"Experiment: {output_dir_name}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 70 + "\n")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.experiment_name,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "dropout": args.dropout,
            "seed": args.seed,
        }
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build transforms
    train_transform = build_transforms(train=True)
    test_transform = build_transforms(train=False)
    
    # Load datasets
    print(f"Loading datasets from {args.data_root}")
    train_dataset = ImageFolder(
        root=os.path.join(args.data_root, 'train'),
        transform=train_transform
    )
    test_dataset = ImageFolder(
        root=os.path.join(args.data_root, 'test'),
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Infer input channels from first batch
    sample_batch, _ = next(iter(train_loader))
    input_channels = sample_batch.shape[1]
    print(f"\nInferred input channels: {input_channels}")
    print(f"Input image shape: {sample_batch.shape}")
    print("=" * 70 + "\n")
    
    # Create model
    num_classes = len(train_dataset.classes)
    print("Creating model...")
    model = SimpleCNN(
        num_classes=num_classes,
        input_channels=input_channels,
        dropout=args.dropout
    )
    model = model.to(device)
    print(f"✓ Model created and moved to {device}\n")
    
    # Print model summary
    print_model_summary(model)
    
    # Save architecture summary to file
    save_model_architecture(model, output_dir, input_size=(1, input_channels, 224, 224))
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    print(f"Loss function: CrossEntropyLoss")
    
    # Define optimizer
    print(f"\nSetting up optimizer: {args.optimizer}")
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Momentum: {args.momentum}" if args.optimizer == 'SGD' else "")
    
    # Learning rate scheduler (optional)
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        print(f"  Learning rate scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    
    # Training loop
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    best_val_acc = 0.0
    best_epoch = 0
    model_path = os.path.join(output_dir, "best_model.pt")
    
    print("\n" + "=" * 70)
    print(f"STARTING TRAINING - {args.epochs} EPOCHS")
    print("=" * 70)
    
    for epoch in tqdm.tqdm(range(args.epochs), desc="Training"):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        # Update scheduler
        if args.use_scheduler:
            scheduler.step(test_acc)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Print epoch results
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "best_val_acc": best_val_acc,
        })
        
        # Save best model
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"  → New best model saved! Val Acc: {best_val_acc:.4f}")
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print("=" * 70)
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(model_path))
    
    # Final evaluation with confusion matrix
    print("Computing confusion matrix on test set...")
    _, final_acc, y_pred, y_true = test_epoch(
        model, test_loader, criterion, device, return_predictions=True
    )
    print(f"Final test accuracy: {final_acc:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, train_dataset.classes, output_dir
    )
    
    # Plot metrics
    print("\nGenerating training plots...")
    plot_metrics(
        {"loss": train_losses, "accuracy": train_accuracies},
        {"loss": test_losses, "accuracy": test_accuracies},
        "loss",
        output_dir=output_dir
    )
    plot_metrics(
        {"loss": train_losses, "accuracy": train_accuracies},
        {"loss": test_losses, "accuracy": test_accuracies},
        "accuracy",
        output_dir=output_dir
    )
    
    # Save training summary
    save_training_summary(
        output_dir=output_dir,
        config=vars(args),
        best_val_acc=best_val_acc,
        best_epoch=best_epoch
    )
    
    # Close wandb
    wandb.finish()
    
    print("\n" + "=" * 70)
    print("ALL OUTPUTS SAVED")
    print("=" * 70)
    print(f"Directory: {output_dir}")
    print(f"  - best_model.pt")
    print(f"  - architecture_summary.txt")
    print(f"  - confusion_matrix.png")
    print(f"  - per_class_accuracy.txt")
    print(f"  - loss.png")
    print(f"  - accuracy.png")
    print(f"  - training_summary.txt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN from scratch for image classification")
    
    # Data parameters
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/uabmcv2526/shared/dataset/2425/MIT_small_train_1",
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save outputs"
    )
    
    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["SGD", "Adam", "AdamW"],
        default="AdamW",
        help="Optimizer to use"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability"
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        help="Use learning rate scheduler"
    )
    
    # Experiment parameters
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="baseline_cnn",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--add_timestamp",
        action="store_true",
        help="Add timestamp to experiment name to avoid overwriting previous runs"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="C3_Week4",
        help="Wandb project name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers"
    )
    
    args = parser.parse_args()
    main(args)
