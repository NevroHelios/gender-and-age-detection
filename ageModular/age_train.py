import os
import torch
import sys
from torch import nn
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, Grayscale
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ageModular.age_model_builder import model_AGEV0
from ageModular.age_data_setup import create_dataloader
from ageModular.age_engine import train
from ageModular.age_utils import transforms, save_model

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="AGEV1.pth", help="name of saved model")
parser.add_argument("-p", "--model_path", type=str, default="models", help="where to save the model")
parser.add_argument("-e", "--num_epochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="input batch size for training")

args = parser.parse_args()
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs

train_dir = os.path.join(os.path.dirname(__file__), "../data/train") # "../data/train"
test_dir = os.path.join(os.path.dirname(__file__), "../data/test") # "../data/test"

# create dataloaders
transforms = transforms()
train_dataloader, test_dataloader = create_dataloader(train_dir,
                                                      test_dir,
                                                      transforms.feature_transform,
                                                      transforms.target_transform,
                                                      batch_size=BATCH_SIZE)

# setup model
model_age = model_AGEV0(input_shape=1, output_shape=1).to(device)

# setup loss function and optimizer
age_loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=model_age.parameters(), lr=0.001)


from timeit import default_timer as timer

start_time = timer()

if __name__ == "__main__":
    model_results = train(model=model_age.to(device),
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=age_loss_fn,
                            epochs=NUM_EPOCHS,
                            device=device)

    end_time = timer()
    print(f"Total training time: {end_time - start_time}")
    
    save_model(model=model_age, model_path=args.model_path, model_name=args.model_name)
    