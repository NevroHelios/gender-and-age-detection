from PIL import Image
from pathlib import Path
import torch
import cv2
import os
import numpy as np

from ageModular.age_model_builder import model_AGEV0
from ageModular.age_utils import transforms


transforms = transforms()

    
def predict_age(img,
                model: torch.nn.Module,
                transforms = transforms) -> int:
    
    # if img.any() is None:
    #     raise ValueError("No image found")
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        
    img = transforms.feature_transform(img)
    
    img = img.unsqueeze(0)
    
    model.eval()
    with torch.inference_mode():
        pred = model(img)
        
    age_pred = int(transforms.pred_transform(pred).squeeze())
    
    return age_pred

    

if __name__ == "__main__":
    
    
    model_path = Path(os.path.dirname(__file__)).parent / "models"
    age_model = model_AGEV0(input_shape=1, output_shape=1)
    age_model.load_state_dict(torch.load(model_path / "model_age.pt"))


    ### get sample image
    image_path = Path(os.path.dirname(__file__)).parent / Path("data/test/male/35_0_1_20170117151527396.jpg.chip.jpg")

    img = cv2.imread(str(image_path))
    print(img.shape)
    if img.any() is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
    else:
        img = Image.open(image_path)
    
    
    age_pred = predict_age(model=age_model,
                           img=img,
                           transforms=transforms)
    
    age_true = int(image_path.stem.split("_")[0])

    print(f"Predicted Age: {age_pred}\nTrue Age: {age_true}")

