from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import  ToTensor, Normalize, Compose, Resize, Grayscale

from a_utils import transforms

BATCH_SIZE = 64
transforms = transforms()

class create_dataset(Dataset):
    def __init__(self, root: str,
                 transform = transforms.feature_transform,
                 target_transform = transforms.target_transform):
        
        self.data = list(Path(root).glob("*/*.jpg"))
        
        # normalizing the input data
        self.age = [img.stem.split("_")[0] for img in self.data]
        self.age = torch.Tensor(list(map(int, self.age)))
        
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = self.data[index]
        image = Image.open(image_path)
        image = self.transform(image)
        age = self.target_transform(self.age[index])
        return image, age
    
    
    
def create_dataloader(train_dir,
                      test_dir,
                      feature_transform: Compose,
                      target_transform: Compose,
                      batch_size:int = BATCH_SIZE):
    
    train_data = create_dataset(train_dir,
                                feature_transform,
                                target_transform)
    
    test_data = create_dataset(test_dir,
                               feature_transform,
                               target_transform)
 
    train_dataloader = DataLoader(train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
    
    return train_dataloader, test_dataloader



