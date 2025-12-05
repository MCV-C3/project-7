import os
import glob
import numpy as np
from PIL import Image
from typing import List, Tuple
import tqdm

def load_dataset(image_folder: str) -> List[Tuple[np.ndarray, int]]:
    # loader that returns numpy arrays
    return [(np.array(img), label) for img, label in Dataset(image_folder)]


def Dataset(ImageFolder: str = "data/MIT_split/train") -> List[Tuple[Image.Image, int]]:
    # Original Dataset implementation returning PIL images and integer labels
    map_classes = {clsi: idx for idx, clsi in enumerate(sorted(os.listdir(ImageFolder)))}
    dataset: List[Tuple[Image.Image, int]] = []

    for idx, cls_folder in enumerate(sorted(os.listdir(ImageFolder))):
        image_path = os.path.join(ImageFolder, cls_folder)
        if not os.path.isdir(image_path):
            continue
        images: List[str] = glob.glob(os.path.join(image_path, "*.jpg"))
        for img in images:
            try:
                img_pil = Image.open(img).convert("RGB")
                dataset.append((img_pil, map_classes[cls_folder]))
            except Exception as e:
                print(f"Error loading {img}: {e}")

    return dataset

def extract_descriptors(bovw, dataset: List[Tuple[np.ndarray, int]], 
                       desc: str = "Phase: Extracting descriptors"):
    all_descriptors = []
    all_labels = []
    
    for image, label in tqdm.tqdm(dataset, desc=desc):
        _, descriptors = bovw._extract_features(image)
        if descriptors is not None:
            all_descriptors.append(descriptors)
            all_labels.append(label)
    
    return all_descriptors, all_labels

def extract_bovw_histograms(bovw, descriptors_list: List[np.ndarray]) -> np.ndarray:
    return np.array([bovw._compute_codebook_descriptor(d, bovw.codebook_algo) for d in descriptors_list])

