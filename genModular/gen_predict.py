from PIL import Image
from pathlib import Path
import torch
import cv2
import os
import numpy as np

from gen_model_builder import GENV0
from gen_utils import transforms




def predict_gender(img,
                    model: torch.nn.Module,
                    transforms) -> int:
    
    if img.any() is None:
        raise ValueError("No image found")
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        
    img = transforms.feature_transform(img)
    
    img = img.unsqueeze(0)
    
    model.eval()
    with torch.inference_mode():
        pred = model(img)
        
    gender_pred = transforms.tgt_transform(int(pred.squeeze().item()))
    
    return gender_pred



if __name__ == "__main__":
    transforms = transforms()

    model_path = Path(os.path.dirname(__file__)).parent / "models"
    gender_model = GENV0(input_shape=1, output_shape=1)
    gender_model.load_state_dict(torch.load(model_path / "model_gen.pth"))

    image_path = Path(os.path.dirname(__file__)).parent / Path("data/test/male/35_0_1_20170117151527396.jpg.chip.jpg")
    img = cv2.imread(str(image_path))
    print(img.shape)
    if img.any() is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
    else:
        img = Image.open(image_path)
        

    gender_pred = predict_gender(img=img, model=gender_model, transforms=transforms)
    gender_true = transforms.tgt_transform(int(image_path.stem.split("_")[1]))
    print(f"Predicted Gender: {gender_pred}\nTrue Gender: {gender_true}")