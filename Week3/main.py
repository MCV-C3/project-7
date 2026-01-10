from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel, WraperModel
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm

from torchvision.models.mnasnet import _InvertedResidual
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop
from helpers import detect_mnasnet_type
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

    args = parser.parse_args()

    exp_name = (
        f"progressive_dropout_{args.dropout_blocks}_blocks_{args.dropout_value}_value"
        f"_reg_{args.reg_type}_{args.reg_lambda}_l1ratio_{args.l1_ratio}"
    )

    DATASET_ROOT = '/data/uabmcv2526/shared/dataset/2425/MIT_small_train_1'
    OUTPUT_PATH = '/data/uabmcv2526/mcvstudent29/output/regularizers'
    OUTPUT_PATH = os.path.join(
        OUTPUT_PATH,
        exp_name
    )
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    EPOCHS = 20
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
            "reg_type": args.reg_type,
            "reg_lambda": args.reg_lambda,
            "l1_ratio": args.l1_ratio,
        }
    )


    torch.manual_seed(42)

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])
    
    data_train = ImageFolder(root=os.path.join(DATASET_ROOT, 'train'), transform=transformation)
    data_test = ImageFolder(root=os.path.join(DATASET_ROOT, 'test'), transform=transformation)

    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=1, pin_memory=True, shuffle=False, num_workers=8)

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = WraperModel(
        num_classes=8,
        unfreeze_blocks=args.unfreeze_blocks,
        dropout_blocks=args.dropout_blocks,
        dropout_value=args.dropout_value,
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

    # Use AdamW to apply weight decay in a decoupled manner (recommended for Adam-family optimizers)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_epochs = EPOCHS


    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

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


        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, args.reg_type, args.reg_lambda, args.l1_ratio)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        })
        
    model_path = os.path.join(OUTPUT_PATH, "saved_model.pt")
    torch.save(model.state_dict(), model_path)

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss", OUTPUT_DIR=OUTPUT_PATH)
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy", OUTPUT_DIR=OUTPUT_PATH)
