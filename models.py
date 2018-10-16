import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
                nn.Conv1d(5, 64, 10),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 32, 3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 16, 3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                )
        self.conv1 = nn.Sequential(
                nn.Conv1d(5, 64, 5),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 32, 3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 16, 3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv1d(5, 64, 3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 32, 3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 16, 3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat((x0, x1, x2), dim=2)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 1)
        return x
