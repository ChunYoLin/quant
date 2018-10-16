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
            data_x = stock_dataset.get_train_datas()
            data_y = stock_dataset.get_train_targets()
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
        self.__data_df = Fetcher().fetch(symbol, start, end)[usecols]
        self.data_len = data_len

    def get_train_datas(self):
        data = self.__data_df.values
        data_x = np.array(list(chunks(data, self.data_len))[:-1])
        return data_x

    def get_train_targets(self):
        data = self.__data_df.values
        data_y = []
        for idx in range(len(data) - self.data_len):
            data_y.append(data[idx + self.data_len, 3])
        data_y = np.asarray(data_y)
        return data_y

    def get_test_datas(self):
        data = self.__data_df.values
        data_x = np.array(list(chunks(data, self.data_len))[-1])
        return data_x

    def get_raw_datas(self):
        return self.__data_df.values

class StockPriceChange():
    def __init__(self, symbol, start, end=date.today(), data_len=5):
        pass
