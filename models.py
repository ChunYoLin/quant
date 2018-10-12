import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
                nn.Conv1d(5, 64, 10),
                nn.BatchNorm1d(64),
                nn.ReLU()
                )
        self.conv1 = nn.Sequential(
                nn.Conv1d(64, 64, 5),
                nn.BatchNorm1d(64),
                nn.ReLU()
                )
        self.conv2 = nn.Sequential(
                nn.Conv1d(64, 32, 3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                )
        self.conv3 = nn.Sequential(
                nn.Conv1d(32, 16, 3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = nn.AvgPool1d(3, 16)(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 1)
        return x
