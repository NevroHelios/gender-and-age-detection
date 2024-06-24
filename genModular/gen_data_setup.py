from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import datasets
from torchvision.transforms import Compose

def create_dataloader(train_dir: str,
                      test_dir: str,
                      feature_transform: Compose,
                      batch_size:int):
    
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=feature_transform,
                                      target_transform=None)
    
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=feature_transform,
                                     target_transform=None)
    
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False)
    classes = train_data.classes
    
    return train_dataloader, test_dataloader, classes