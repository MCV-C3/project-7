import torch
import torchvision.transforms.v2 as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from models import PatchMLP, PatchMLP2

PATH_MODEL_1 = "/export/home/group07/week2/runs/week2_group07_97506/best.pt"
PATH_MODEL_2 = "/export/home/group07/week2/runs/week2_group07_97524/best.pt"
DATA_VAL_PATH = "/export/home/group07/mcv/datasets/C3/2526/places_reduced/val"

PATCH_SIZE = 45 
HIDDEN_DIM = 1028
NUM_CLASSES = 11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((225, 225), antialias=True)
])

dataset = ImageFolder(DATA_VAL_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
classes = dataset.classes

model1 = PatchMLP(3, 225, 225, PATCH_SIZE, HIDDEN_DIM, NUM_CLASSES).to(device)
ckpt1 = torch.load(PATH_MODEL_1, map_location=device, weights_only=True)
model1.load_state_dict(ckpt1['model_state_dict'])
model1.eval()

model2 = PatchMLP2(3, 225, 225, PATCH_SIZE, HIDDEN_DIM, NUM_CLASSES).to(device)
ckpt2 = torch.load(PATH_MODEL_2, map_location=device, weights_only=True)
model2.load_state_dict(ckpt2['model_state_dict'])
model2.eval()

target_class_name = 'water_ice_snow'
found = False
img_tensor, attn_weights = None, None
pred1_name, pred2_name = "", ""

with torch.no_grad():
    for img, label in dataloader:
        img = img.to(device)
        label_idx = label.item()
        real_class = classes[label_idx]
        
        if real_class != target_class_name:
            continue
            
        out1 = model1(img)
        pred1 = out1.argmax(dim=1).item()
        
        out2, _, weights = model2(img, return_embedding=True)
        pred2 = out2.argmax(dim=1).item()
        
        if pred1 != label_idx and pred2 == label_idx:
            print(f"Sample found: {real_class}")
            print(f"Model 1: {classes[pred1]} (Fail)")
            print(f"Model 2: {classes[pred2]} (Success)")
            
            img_tensor = img
            attn_weights = weights
            pred1_name = classes[pred1]
            pred2_name = classes[pred2]
            found = True
            break

if not found:
    print("No suitable example found in this batch.")
    exit()

img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
img_np = np.clip(img_np, 0, 1)

attn_map = attn_weights.squeeze().cpu().numpy()
grid_dim = int(np.sqrt(attn_map.shape[0]))
attn_map = attn_map.reshape(grid_dim, grid_dim)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(img_np)
axs[0].set_title(f"Model 1 (Mean Pooling)\nPred: {pred1_name} (Fail)\nReal: {target_class_name}", color='red', fontsize=12, fontweight='bold')
axs[0].axis('off')

axs[1].imshow(img_np)
im = axs[1].imshow(attn_map, cmap='jet', alpha=0.5, extent=[0, 225, 225, 0], interpolation='bicubic')
axs[1].set_title(f"Model 2 (Attention)\nPred: {pred2_name} (Success)", color='green', fontsize=12, fontweight='bold')
axs[1].axis('off')

cbar = plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
cbar.set_label('Attention Weight')

plt.tight_layout()
plt.savefig('attention_case_study.png', dpi=150)
plt.show()