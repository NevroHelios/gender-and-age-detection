from pathlib import Path
import torch
import os
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor

class transforms:
    def __init__(self):
        self.feature_transform = Compose([
                                    Resize((128, 128)),
                                    Grayscale(),
                                    ToTensor()
                                ])
        
        
        self.dct_true_t = {1 : "male", 0 : "female"}
        
        
        self.tgt_transform = lambda x: self.dct_true_t[x]

def save_model(model: torch.nn.Module, model_path: str, model_name: str):

    target_dir_path = Path(model_path)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name
    
    print(f"Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

def load_model(model: torch.nn.Module, model_path: str, model_name: str):

    model_save_path = Path(model_path) / model_name
    
    print(f"Loading model from: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path, map_location="cpu"))