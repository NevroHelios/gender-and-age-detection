from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from pathlib import Path
import torch

age_list = [img.stem.split("_")[0] for img in Path("../data/train").glob("*/*.jpg")]
age_min = 1 # torch.Tensor(list(map(int, age_list))).min()
age_max = 117 # torch.Tensor(list(map(int, age_list))).max()

class transforms:
    def __init__(self):
        
        self.feature_transform = Compose([
                                Resize((128, 128)),
                                Grayscale(),
                                ToTensor(),
                                ])
    
        self.target_transform = lambda x: (x - age_min)/(age_max - age_min)

        self.pred_transform = lambda x: x*(age_max - age_min) + age_min

# def target_transform(x, age_min = age_min, age_max = age_max):
#     return (x - age_min)/(age_max - age_min)

def save_model(model: torch.nn.Module, model_path: str, model_name: str):

    target_dir_path = Path(model_path)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name
    
    print(f"Saving model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)