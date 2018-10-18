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

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv0(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 1)
        return x

class RNNModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(5, 32)
        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc0(x)
        r_out, (h_c, h_n) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out
