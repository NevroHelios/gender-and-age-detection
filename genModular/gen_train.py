import os
import torch
import sys
from torch import nn
from torchvision.transforms import  ToTensor, Normalize, Compose, Resize, Grayscale
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genModular.gen_data_setup import create_dataloader
from genModular.gen_model_builder import GENV0
from genModular.gen_engine import train
from genModular.gen_utils import save_model

torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="GENV1.pt", help="name of saved model")
parser.add_argument("-p", "--model_path", type=str, default="models", help="where to save the model")
parser.add_argument("-e", "--num_epochs", type=int, default=10, help="number of epochs to train for")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="input batch size for training")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs

train_dir = os.path.join(os.path.dirname(__file__), "../data/train")
test_dir = os.path.join(os.path.dirname(__file__), "../data/test")

feature_transform = Compose([
    Resize((128, 128)),
    Grayscale(),
    ToTensor()
])

# create dataloaders
train_dataloader, test_dataloader, class_names = create_dataloader(train_dir,
                                                                   test_dir,
                                                                   feature_transform,
                                                                   batch_size=BATCH_SIZE)

model = GENV0(input_shape=1, output_shape=1).to(device)

gen_loss_fn = torch.nn.BCELoss()

optimizer = torch.optim.SGD(params = model.parameters(),
                           lr = 0.01)

from timeit import default_timer as timer

start_time = timer()

if __name__ == "__main__":
    model_results = train(model=model.to(device),
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=gen_loss_fn,
                            epochs=NUM_EPOCHS,
                            device=device)

    end_time = timer()
    print(f"Total training time: {end_time - start_time}")
    
    save_model(model=model, model_path=args.model_path, model_name=args.model_name)
    print("Model saved!")