import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data

from torch_data import StockDataset
from models import SimpleFullyConnected


def main():
    dataset = StockDataset("AMD", "1993-01-01")
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

    net = SimpleFullyConnected().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    criterion = nn.MSELoss()
    for epoch in range(100000):
        losses = 0.
        for step, (batch_x, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.type(torch.FloatTensor).cuda(), batch_y.type(torch.FloatTensor).cuda()
            batch_x = batch_x.view(-1, 25)
            predict = net(batch_x)
            loss = criterion(batch_y, predict)
            loss.backward()
            losses += loss
            optimizer.step()
        print(f"epoch: {epoch}, losses: {losses/64.}")

if __name__ == "__main__":
    main()
