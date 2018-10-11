import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data

from torch_data import StockDataset
from models import SimpleModel


def main():
    dataset = StockDataset("AMD", "1993-01-01", data_len = 20)
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

    net = SimpleModel().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    for epoch in range(100000):
        losses = 0.
        for step, (batch_x, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.type(torch.FloatTensor).cuda(), batch_y.type(torch.FloatTensor).cuda()
            batch_x = batch_x.transpose(1, 2)
            predict = net(batch_x)
            loss = criterion(batch_y, predict)
            loss.backward()
            losses += loss
            optimizer.step()
        print(f"epoch: {epoch}, losses: {losses/64.}")
        test_data = torch.Tensor(dataset.get_test_datas()).cuda()
        test_data = test_data.transpose(0, 1)
        test_data = test_data.view(1, test_data.shape[0], test_data.shape[1])
        test_predict = net(test_data)
        print(test_data)
        print(test_predict)

if __name__ == "__main__":
    main()
