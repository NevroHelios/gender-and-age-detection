import torch
from torch import nn

class GENV0(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.gender_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128*16*16, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=output_shape)
        )
        
    def forward(self, x):
        gender = torch.sigmoid(self.gender_fc(self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))))
        return gender
    

# modelV0 = model_customV0(input_shape=1, output_shape=1).to(device)