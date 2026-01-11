from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel, DeeperModel, PyramidalMLP, BottleneckMLP, PatchMLP, PatchMLP2
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import os
import sys


# Train function
def train(model, dataloader, criterion, optimizer, device):
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

def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str):
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
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.close()  # Close the figure to free memory



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

def plot_image(i, predictions_array, true_labels, images, class_names):
    predictions_array = predictions_array[i]
    true_label = true_labels[i]
    img = images[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # CHW → HWC
    plt.imshow(np.transpose(img, (1, 2, 0)))

    predicted_label = np.argmax(predictions_array)

    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(
        f"Pred: {class_names[predicted_label]}\nGT: {class_names[true_label]}",
        color=color
    )


def plot_value_array(i, predictions_array, true_labels, class_names):
    predictions_array = predictions_array[i]
    true_label = true_labels[i]

    plt.grid(False)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks([])

    thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#888888")
    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def tablaAciertos(predictions, labels, images, class_names, save_path):
    numrows = 5
    numcols = 3

    plt.figure(figsize=(2*2*numcols, 2*numrows))

    for i in range(15):
        plt.subplot(numrows, 2*numcols, 2*i+1)
        plot_image(i, predictions, labels, images, class_names)

        plt.subplot(numrows, 2*numcols, 2*i+2)
        plot_value_array(i, predictions, labels, class_names)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    torch.manual_seed(42)

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224), antialias=True),
                                ])
    TRAIN_PATH = "/export/home/group07/mcv/datasets/C3/2526/places_reduced/train"
    TEST_PATH = "/export/home/group07/mcv/datasets/C3/2526/places_reduced/val"
    RUNS_PATH = sys.argv[1]
    # RUNS_PATH = "/export/home/group07/week2/runs/"
    # runs = [int(d) for d in os.listdir(RUNS_PATH) if os.path.isdir(d)]
    # current_run = max(runs) + 1 if runs else 0
    # RUNS_PATH += f"{current_run}/"
    os.makedirs(RUNS_PATH, exist_ok=True)

    data_train = ImageFolder(TRAIN_PATH, transform=transformation)
    data_test = ImageFolder(TEST_PATH, transform=transformation)
    class_names = data_train.classes

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)

    C, H, W = np.array(data_train[0][0]).shape


    model = BottleneckMLP(input_d=C*H*W, hidden_dims=(4096, 2048, 1024, 2048, 4096), output_d=11)
    plot_computational_graph(model, input_size=(1, C*H*W))  # Batch size of 1, input_dim=10
    
    # PATCH_SIZE = 25
    # HIDDEN_DIM = 1028
    
    # model = PatchMLP2(
    #     input_c=C, 
    #     input_h=H, 
    #     input_w=W, 
    #     patch_size=PATCH_SIZE, 
    #     hidden_d=HIDDEN_DIM, 
    #     output_d=11
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    best_val_accuracy = 0.0
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # save the best model
        best_val_acc = test_accuracy
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": test_loss,
                "val_accuracy": test_accuracy,
            },
            RUNS_PATH+"best.pt",
        )

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss")
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy")

    # save the last model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": test_loss,
            "val_accuracy": test_accuracy,
        },
        RUNS_PATH+"last.pt",
    )


    # =========================================================
    # TABLE OF PREDICTIONS (Tabla de aciertos)
    # =========================================================
    model.eval()

    all_predictions = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images.view(images.size(0), -1))
            probs = torch.softmax(outputs, dim=1)

            all_predictions.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_images.append(images.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = np.concatenate(all_images, axis=0)

    tabla_path = os.path.join(RUNS_PATH, "tabla_aciertos.png")

    tablaAciertos(
        predictions=all_predictions,
        labels=all_labels,
        images=all_images,
        class_names=class_names,
        save_path=tabla_path
    )
    print(f"Tabla de aciertos guardada en: {tabla_path}")