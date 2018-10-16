from datetime import date, datetime, timedelta

from torch.utils import data
import numpy as np

from fetcher import Fetcher


def chunks(data, data_len):
    for idx in range(len(data) - data_len + 1):
        yield data[idx:idx+data_len]

class StockDataset(data.Dataset):
    def __init__(self, symbols, start, end=date.today(), data_len=5):
        assert isinstance(symbols, list)
        self.datasets = []
        for s in symbols:
            stock_dataset = StockPriceRegression(s, start, end, data_len)
            self.datasets.append(stock_dataset)
        self.data_x = None
        self.data_y = None
        for stock_dataset in self.datasets:
            data_x, data_y = stock_dataset.get_train_datas()
            if self.data_x is None:
                self.data_x, self.data_y = data_x, data_y
            else:
                self.data_x = np.concatenate((self.data_x, data_x))
                self.data_y = np.concatenate((self.data_y, data_y))

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)
        

class StockPriceRegression():
    def __init__(self, symbol, start, end=date.today(), data_len=5):
        usecols = ["open", "high", "low", "close", "volume"]
        self.data_df = Fetcher().fetch(symbol, start, end)[usecols]
        self.data_len = data_len
        self.data_x, self.data_y = self.get_train_datas()

    def get_train_datas(self):
        data = self.data_df.values
        data_len = self.data_len
        data_x = np.array(list(chunks(data, data_len))[:-1])
        data_y = np.zeros([data_x.shape[0], 1])
        for idx in range(len(data) - data_len):
            data_y[idx] = data[idx + data_len, 3]
        return data_x, data_y

    def get_test_datas(self):
        data = self.data_df.values
        data_len = self.data_len
        data_x = np.array(list(chunks(data, data_len))[-1])
        return data_x
