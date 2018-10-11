import torch.nn as nn


class SimpleFullyConnected(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(25, 128)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x
