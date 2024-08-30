import pickle
from torch.utils.data import Dataset
import numpy as np
import os

class CondSeqImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, conditional_offset=5):
        self.data_dir = data_dir
        self.transform = transform
        self.conditional_offset = conditional_offset
        self.cond_images = []
        self.target_images = []
        self._load_data()

    def _load_data(self):
        files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.mpy')])
        for file in files:
            with open(file, 'rb') as f:
                images = pickle.load(f)
                if isinstance(images, list):
                    images = np.array(images)
                    
                for img_idx in range(len(images) - self.conditional_offset):
                    self.cond_images.append(images[img_idx])
                    self.target_images.append(images[img_idx + self.conditional_offset])

    def __len__(self):
        return len(self.cond_images)

    def __getitem__(self, idx):
        cond_image = self.cond_images[idx]
        image = self.target_images[idx]

        if self.transform:
            image = self.transform(image)
            cond_image = self.transform(cond_image)

        return image, cond_image