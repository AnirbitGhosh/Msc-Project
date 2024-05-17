from torch.utils.data.dataset import Dataset
import os
import pickle
import torchvision

class WildfireDataset(Dataset):
    def __init__(self, img_path, img_ext="mpy"):
        self.img_ext = img_ext  # Correct variable name for consistency
        self.image_files = [os.path.join(img_path, f) for f in os.listdir(img_path)]
        self.images = []
        
        for file in self.image_files:
            filepath = os.path.join(img_path, file)
             # Load image on demand
            with open(filepath, 'rb') as f:
                img_arr = pickle.load(f)  # Assume images are stored as pickled numpy arrays
            
            self.images.extend(img_arr)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img = self.images[index]
        
        # Convert numpy array to tensor
        img_tensor = torchvision.transforms.ToTensor()(img)
        
        # Normalize the tensor
        img_tensor = (2 * img_tensor) - 1
        return img_tensor