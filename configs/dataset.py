import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

#####

class PALMDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transforms):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path, 'r')
        # img_data = np.array(img)
        mask = Image.open(mask_path, 'r')
        # mask_data = np.array(mask)

        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return (img, mask)
