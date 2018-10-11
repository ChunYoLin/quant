import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(5, 64, 10),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, 5),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 32, 3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 16, 3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                )
        self.fc1 = nn.Linear(48, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
