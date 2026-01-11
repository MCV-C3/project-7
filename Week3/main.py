from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel, WraperModel, build_transforms
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm

from torchvision.models.mnasnet import _InvertedResidual
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop
from helpers import detect_mnasnet_type, add_batchnorm_after_block
import argparse

import wandb
import os

# Train function
def train(model, dataloader, criterion, optimizer, device, reg_type: str = "none", reg_lambda: float = 0.0, reg_l1_ratio: float = 0.5):
    """Train for one epoch and optionally include L1/L2/Elastic Net regularization in the loss.

    Args:
        model: torch.nn.Module
        dataloader: DataLoader
        criterion: loss function
        optimizer: optimizer
        device: torch.device
        reg_type: 'none' | 'l1' | 'l2' | 'elastic'
        reg_lambda: regularization strength (global multiplier)
        reg_l1_ratio: mixing ratio for Elastic Net (alpha), 0 <= alpha <= 1

    Returns:
        avg_loss, accuracy
    """
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Regularization handling
        # - L1 is added explicitly to the loss when requested
        # - L2 is applied via optimizer weight_decay (using AdamW) when requested
        if reg_lambda > 0.0:
            if reg_type == "l1":
                l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                loss = loss + reg_lambda * l1_norm
            elif reg_type == "l2":
                # L2 handled by optimizer weight_decay; do not add explicit L2 term to the loss
                pass
            elif reg_type == "elastic":
                # Elastic Net: L1 portion added to loss, L2 portion handled by optimizer via weight_decay
                # L1 coefficient should be alpha * lambda
                l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                loss = loss + reg_lambda * (reg_l1_ratio * l1_norm)

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


def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

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

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str, OUTPUT_DIR: str = "./"):
    """
    Plots and saves metrics for training and testing.

    Args:
        train_metrics (Dict): Dictionary containing training metrics.
        test_metrics (Dict): Dictionary containing testing metrics.
        metric_name (str): The name of the metric to plot (e.g., "loss", "accuracy").

    Saves:
        - loss.png for loss plots
        - metrics.png for other metrics plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f'Train {metric_name.capitalize()}')
    plt.plot(test_metrics[metric_name], label=f'Test {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot with the appropriate name
    filename = "loss.png" if metric_name.lower() == "loss" else "metrics.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Plot saved as {filename}")

    plt.close()  # Close the figure to free memory

# Data augmentation example
def get_data_transforms():
    """
    Returns a Compose object with data augmentation transformations.
    """
    return Compose([
        RandomResizedCrop(size=224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def plot_computational_graph(model: torch.nn.Module, input_size: tuple, filename: str = "computational_graph"):
    """
    Generates and saves a plot of the computational graph of the model.

    Args:
        model (torch.nn.Module): The PyTorch model to visualize.
        input_size (tuple): The size of the dummy input tensor (e.g., (batch_size, input_dim)).
        filename (str): Name of the file to save the graph image.
    """
    model.eval()  # Set the model to evaluation mode
    
    # Generate a dummy input based on the specified input size
    dummy_input = torch.randn(*input_size)

    # Create a graph from the model
    graph = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True).render(filename, format="png")

    print(f"Computational graph saved as {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--disable_residual",
        type=int,
        default=None,
        help="Index of _InvertedResidual block to disable (None = baseline)"
    )

    parser.add_argument(
        "--unfreeze_blocks",
        type=int,
        default=0,
        help="Number of backbone blocks to unfreeze from the end"
    )

    parser.add_argument(
        "--dropout_blocks",
        type=int,
        default=0,
        help="Number of backbone blocks to add dropout from the beginning"
    )

    parser.add_argument(
        "--dropout_value",
        type=float,
        default=0.5,
        help="Dropout value to use in dropout layers"
    )

    parser.add_argument(
        "--reg_type",
        type=str,
        choices=["none", "l1", "l2", "elastic"],
        default="none",
        help="Regularizer type: none, l1, l2, elastic"
    )

    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.0,
        help="Regularization strength (L1 and/or L2)"
    )

    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.5,
        help="L1 ratio for Elastic Net (alpha in [0,1]). Only used when --reg_type elastic"
    )

    parser.add_argument(
        "--aug_type",
        type=str,
        choices=["none", "flip", "color", "geometric", "translation"],
        default="none",
        help="Data augmentation type: none, flip, color, geometric, translation"
    )

    parser.add_argument(
        "--aug_ratio",
        type=float,
        default=0.0,
        help="Ratio of training data to augment (0.0 to 1.0)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["SGD", "Adam", "AdamW"],
        default="AdamW",
        help="Optimizer to use"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer (L2 regularization)"
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer (only used if optimizer=SGD)"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom experiment name (overrides auto-generated name)"
    )

    args = parser.parse_args()

    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = (
            f"progressive_dropout_{args.dropout_blocks}_blocks_{args.dropout_value}_value"
            f"_reg_{args.reg_type}_{args.reg_lambda}_l1ratio_{args.l1_ratio}"
            f"_aug_{args.aug_type}_{args.aug_ratio}"
        )

    DATASET_ROOT = '/data/uabmcv2526/shared/dataset/2425/MIT_small_train_1'
    OUTPUT_PATH = '/data/uabmcv2526/mcvstudent29/output/best_config_training/'
    OUTPUT_PATH = os.path.join(
        OUTPUT_PATH,
        exp_name
    )
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Use command-line arguments for hyperparameters
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    CURRENT_MODE = 'baseline'  # Options: 'baseline', 'multilayer', 'dropout', 'batchnorm', 'finetune', 'finetune_progressive'
    UNFREEZE_EPOCH = 5
    # REMOVED_RESIDUALS = ["brown"]  # Options: list of {"green", "yellow", "brown"}

    wandb.init(
        project="C3_Week3_regularizers",
        name=exp_name,
        config={
            "disabled_residual": args.disable_residual,
            "mode": CURRENT_MODE,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "reg_type": args.reg_type,
            "reg_lambda": args.reg_lambda,
            "l1_ratio": args.l1_ratio,
            "aug_type": args.aug_type,
            "aug_ratio": args.aug_ratio,
        }
    )


    torch.manual_seed(42)

    # Test transformation (always no augmentation)
    test_transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(224, 224)),
    ])
    
    # Training transformation (with or without augmentation)
    if args.aug_type == "none" or args.aug_ratio == 0.0:
        train_transformation = test_transformation
    else:
        print(f"DEBUG: Building transforms with aug_type={args.aug_type}, aug_ratio={args.aug_ratio}")
        train_transformation = build_transforms(
            use_flip=(args.aug_type == "flip"),
            use_color=(args.aug_type == "color"),
            use_geometric=(args.aug_type == "geometric"),
            use_translation=(args.aug_type == "translation"),
            aug_ratio=args.aug_ratio
        )
        print(f"DEBUG: Transforms built: {train_transformation}")
    
    data_train = ImageFolder(root=os.path.join(DATASET_ROOT, 'train'), transform=train_transformation)
    data_test = ImageFolder(root=os.path.join(DATASET_ROOT, 'test'), transform=test_transformation)

    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=1, pin_memory=True, shuffle=False, num_workers=8)

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = WraperModel(
        num_classes=8,
        unfreeze_blocks=args.unfreeze_blocks,
        dropout_blocks=args.dropout_blocks,
        dropout_value=args.dropout_value,
        use_batchnorm_blocks=False
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = LEARNING_RATE * 0.1 if args.unfreeze_blocks != 0 else LEARNING_RATE

    # Map regularizer settings to optimizer weight_decay for L2 behavior:
    # - l2: set weight_decay = reg_lambda
    # - elastic: set weight_decay = (1 - l1_ratio) * reg_lambda (L2 portion handled by optimizer)
    # - none / l1: weight_decay = 0.0 (no L2 applied via optimizer)
    if args.reg_type == "l2":
        weight_decay = args.reg_lambda
    elif args.reg_type == "elastic":
        weight_decay = (1.0 - args.l1_ratio) * args.reg_lambda
    else:
        weight_decay = 0.0

    # Create optimizer based on user selection
    # Note: For regularization experiments, weight_decay is computed from reg_lambda
    # For hyperparameter optimization experiments, use args.weight_decay directly
    final_weight_decay = weight_decay if weight_decay > 0 else args.weight_decay
    
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=args.momentum,
            weight_decay=final_weight_decay
        )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=final_weight_decay
        )
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=final_weight_decay
        )
    
    num_epochs = EPOCHS


    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    # Track best validation accuracy and save best model
    best_val_acc = 0.0
    best_epoch = 0
    model_path = os.path.join(OUTPUT_PATH, "best_model.pt")

    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE MODEL"):
        # # Collect all inverted residual blocks in a fixed order
        # inverted_residuals = [
        #     m for m in model.modules() if isinstance(m, _InvertedResidual)
        # ]

        # # Disable exactly one residual if requested
        # if args.disable_residual is not None:
        #     assert 0 <= args.disable_residual < len(inverted_residuals), \
        #         "Invalid residual index"

        #     inverted_residuals[args.disable_residual].apply_residual = False
        #     print(f"Disabled residual block #{args.disable_residual}")


        # ADVO INI AÑADIR WARM UP
        # -------- WARM-UP -> FINE-TUNING --------
        # if epoch == UNFREEZE_EPOCH:
        #     print(" Starting fine-tuning with BatchNorm")

        #     # Descongelar bloques progresivamente
        #     backbone_blocks = list(model.backbone.layers)
        #     for block in backbone_blocks[-args.unfreeze_blocks:]:
        #         for p in block.parameters():
        #             p.requires_grad = True

        #     # Activar BatchNorm en los bloques descongelados
        #     if args.unfreeze_blocks > 0:
        #         start_idx = len(backbone_blocks) - args.unfreeze_blocks
        #         for i in range(start_idx, len(backbone_blocks)):
        #             backbone_blocks[i] = add_batchnorm_after_block(
        #                 backbone_blocks[i]
        #             )
        #         model.backbone.layers = nn.Sequential(*backbone_blocks)

        #     # Reducir learning rate
        #     optimizer = optim.AdamW(
        #         filter(lambda p: p.requires_grad, model.parameters()),
        #         lr=LEARNING_RATE * 0.1,
        #         weight_decay=weight_decay
        #     )
        # ADVO FIN AÑADIR WARM UP

        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, args.reg_type, args.reg_lambda, args.l1_ratio)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save best model based on validation accuracy
        if test_accuracy > best_val_acc:
            best_val_acc = test_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"  → New best model saved! Val Acc: {best_val_acc:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_val_acc": best_val_acc,
        })
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss", OUTPUT_DIR=OUTPUT_PATH)
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy", OUTPUT_DIR=OUTPUT_PATH)
