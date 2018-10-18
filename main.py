import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data

from torch_data import StockPriceRegression, StockDataset
from models import SimpleModel, RNNModel


def main():
    dataset = StockPriceRegression("AMD", "1993-01-01", data_len = 20)
    stock_list = ["AMD", "AAPL", "NVDA", "GOOG", "CDNS", "QCOM", "INTC", "MU"]
    datasets = StockDataset(stock_list, "1993-01-01", data_len = 20)
    dataloader = data.DataLoader(datasets, batch_size=32, shuffle=True, drop_last=True)

    net = RNNModel().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    for epoch in range(100000):
        losses = 0.
        for step, (batch_x, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.type(torch.FloatTensor).cuda(), batch_y.type(torch.FloatTensor).cuda()
            predict = net(batch_x)
            loss = criterion(batch_y, predict)
            loss.backward()
            losses += loss
            optimizer.step()
        print(f"epoch: {epoch}, losses: {losses/32.}")
        test_data = torch.Tensor(dataset.get_test_datas()).cuda()
        test_data = test_data.view(1, test_data.shape[0], test_data.shape[1])
        test_predict = net(test_data)
        for p_data in test_data[0]:
            recent_close = p_data[3].cpu().detach().numpy()
            print(dataset.denormalize(recent_close))
        #  print(test_data)
        print(dataset.denormalize(test_predict.cpu().detach().numpy()))

if __name__ == "__main__":
    main()
