from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision.transforms import  ToTensor, Normalize, Compose, Resize, Grayscale

custom_transform = Compose([
    Resize((128, 128)),
    Grayscale(),
    ToTensor(),
    # Normalize((0.5,), (0.5,))
])

age_list = [img.stem.split("_")[0] for img in Path("../data/train").glob("*/*.jpg")]
age_min = 1 # torch.Tensor(list(map(int, age_list))).min()
age_max = 117 # torch.Tensor(list(map(int, age_list))).max()
print(age_min, age_max)
target_transform = lambda x: (x - age_min)/(age_max - age_min)

class create_dataset(Dataset):
    def __init__(self, root: str, transform = custom_transform, target_transform = target_transform):
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
    
train_dir = "../data/train"
test_dir = "../data/test"

train_data = create_dataset(train_dir, custom_transform, target_transform)
test_data = create_dataset(test_dir, custom_transform, target_transform)

print(len(train_data), len(test_data))