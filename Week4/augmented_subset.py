from torch.utils.data import Dataset
import random

class AugmentedSubset(Dataset):
    """
    Dataset that samples from a base dataset but applies augmentation transforms.
    """

    def __init__(self, base_dataset, size):
        self.base_dataset = base_dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        real_idx = random.randint(0, len(self.base_dataset) - 1)
        return self.base_dataset[real_idx]
