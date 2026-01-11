import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.v2  as F
from torchvision.datasets import ImageFolder


from models import BottleneckMLP, PatchMLP


def extract_embeddings(model, dataloader, device, dense=False):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            _, emb = model(inputs, return_embedding=True)
            
            embeddings.extend(list(emb.cpu().numpy()))
            labels.append(targets.numpy())

    # embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    return embeddings, labels


if __name__ == "__main__":
    torch.manual_seed(42)

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(225, 225), antialias=True),
                                ])
    TRAIN_PATH = "/export/home/group07/mcv/datasets/C3/2526/places_reduced/train"
    TEST_PATH = "/export/home/group07/mcv/datasets/C3/2526/places_reduced/val"
    MODEL_PATH = "/export/home/group07/week2/runs/week2_group07_97506/best.pt"

    data_train = ImageFolder(TRAIN_PATH, transform=transformation)
    data_test = ImageFolder(TEST_PATH, transform=transformation) 

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    # model = BottleneckMLP(input_d=C*H*W, hidden_dims=(600, 300, 150, 300, 600), output_d=11)
    model = PatchMLP(C, H, W, patch_size=45, hidden_d=1028, output_d=11)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

    np.savez_compressed(MODEL_PATH+"train_embeddings.npz", embeddings=train_embeddings, labels=train_labels)
    np.savez_compressed(MODEL_PATH+"test_embeddings.npz", embeddings=test_embeddings, labels=test_labels)
    # load with: data = np.load("path/to/file.npz")
    # embeddings = data['embeddings']
    # labels = data['labels']