import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict
import torch
from torchview import draw_graph


def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str, output_dir: str = "./"):
    """
    Plots and saves metrics for training and testing.

    Args:
        train_metrics (Dict): Dictionary containing training metrics.
        test_metrics (Dict): Dictionary containing testing metrics.
        metric_name (str): The name of the metric to plot (e.g., "loss", "accuracy").
        output_dir (str): Directory to save the plot.

    Saves:
        - loss.png for loss plots
        - accuracy.png for accuracy plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f'Train {metric_name.capitalize()}', marker='o')
    plt.plot(test_metrics[metric_name], label=f'Test {metric_name.capitalize()}', marker='s')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name.capitalize(), fontsize=12)
    plt.title(f'{metric_name.capitalize()} Over Epochs', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot with the appropriate name
    filename = f"{metric_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved as {filepath}")

    plt.close()  # Close the figure to free memory


def save_training_summary(output_dir: str, config: Dict, best_val_acc: float, best_epoch: int, train_acc_at_best: float = None):
    """
    Save a summary of the training run to a text file.
    
    Args:
        output_dir (str): Directory to save the summary
        config (Dict): Configuration dictionary
        best_val_acc (float): Best validation accuracy achieved
        best_epoch (int): Epoch at which best validation accuracy was achieved
        train_acc_at_best (float): Training accuracy at best epoch
    """
    summary_path = os.path.join(output_dir, "training_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write("-" * 50 + "\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Best Validation Accuracy: {best_val_acc*100:.2f}%\n")
        f.write(f"  Best Test Accuracy: {best_val_acc*100:.2f}%\n")  # Same as validation for now
        f.write(f"  Best Epoch: {best_epoch}\n")
        f.write(f"  Training Accuracy at Best Epoch: {train_acc_at_best*100:.2f}%\n")
        f.write("=" * 50 + "\n")
    
    print(f"Training summary saved to {summary_path}")


def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Size of input tensor (batch_size, channels, height, width)
    """
    import torch
    
    print("=" * 70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 70)
    print(model)
    print("=" * 70)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            dummy_input = torch.randn(*input_size)
            output = model(dummy_input.to(next(model.parameters()).device))
            print(f"\nInput Shape: {dummy_input.shape}")
            print(f"Output Shape: {output.shape}")
    except Exception as e:
        print(f"\nCould not perform test forward pass: {e}")
    
    print("=" * 70)


def save_model_architecture(model, output_dir: str, input_size=(1, 3, 224, 224)):
    """
    Save model architecture details to a text file.
    
    Args:
        model: PyTorch model
        output_dir: Directory to save the architecture summary
        input_size: Size of input tensor
    """
    arch_path = os.path.join(output_dir, "architecture_summary.txt")
    
    with open(arch_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL ARCHITECTURE SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        # Model structure
        f.write(str(model) + "\n\n")
        f.write("=" * 70 + "\n")
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        f.write("\nParameter Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n")
        
        # Layer-wise parameter count
        f.write("\n" + "=" * 70 + "\n")
        f.write("Layer-wise Parameters:\n")
        f.write("-" * 70 + "\n")
        for name, param in model.named_parameters():
            f.write(f"{name:50s} {param.numel():>12,} params\n")
        
        # Input/Output shapes
        f.write("\n" + "=" * 70 + "\n")
        try:
            with torch.no_grad():
                dummy_input = torch.randn(*input_size)
                output = model(dummy_input.to(next(model.parameters()).device))
                f.write(f"Input Shape: {tuple(dummy_input.shape)}\n")
                f.write(f"Output Shape: {tuple(output.shape)}\n")
        except Exception as e:
            f.write(f"Could not perform test forward pass: {e}\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"Architecture summary saved to {arch_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: str):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    
    plt.close()
    
    # Also save per-class accuracy to text file
    acc_path = os.path.join(output_dir, "per_class_accuracy.txt")
    with open(acc_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("PER-CLASS ACCURACY\n")
        f.write("=" * 50 + "\n\n")
        
        for i, class_name in enumerate(class_names):
            class_acc = cm_normalized[i, i]
            f.write(f"{class_name:30s}: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})\n")
        
        overall_acc = np.trace(cm) / np.sum(cm)
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n")
        f.write("=" * 50 + "\n")
    
    print(f"Per-class accuracy saved to {acc_path}")


def save_architecture_diagram(model, output_dir: str, input_size=(1, 3, 224, 224)):
    """
    Generate and save a visual diagram of the model architecture.
    
    Args:
        model: PyTorch model
        output_dir: Directory to save the diagram
        input_size: Size of input tensor
    """
    try:
        # Generate architecture graph
        model_graph = draw_graph(
            model, 
            input_size=input_size,
            expand_nested=True,
            graph_dir='TB',  # Top to bottom
            hide_inner_tensors=True,
            hide_module_functions=True,
            roll=True
        )
        
        # Save as PNG
        arch_diagram_path = os.path.join(output_dir, "architecture_diagram")
        model_graph.visual_graph.render(
            filename=arch_diagram_path,
            format='png',
            cleanup=True  # Remove the intermediate dot file
        )
        
        print(f"Architecture diagram saved to {arch_diagram_path}.png")
        
    except Exception as e:
        print(f"Could not generate architecture diagram: {e}")
        print("You may need to install: pip install torchview graphviz")
