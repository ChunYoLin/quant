from datetime import date, datetime, timedelta

from torch.utils import data
import numpy as np

from fetcher import Fetcher


def chunks(data, data_len):
    for idx in range(len(data) - data_len + 1):
        yield data[idx:idx+data_len]

class StockDataset(data.Dataset):
    def __init__(self, symbol, start, end=date.today(), data_len=5):
        usecols = ["open", "high", "low", "close", "volume"]
        self.data_df = Fetcher().fetch(symbol, start, end)[usecols]
        self.data_x, self.data_y = self.get_train_datas(self.data_df.values, data_len)

    def __getitem__(self, idx):
        return self.data_x, self.data_y

    def __len__(self):
        return len(self.data_x)

    def get_train_datas(self, data, data_len):
        data_x = np.array(list(chunks(data, data_len))[:-1])
        data_y = np.zeros([data_x.shape[0], 1])
        for idx in range(len(data) - data_len):
            data_y[idx] = data[idx + data_len, 3]
        return data_x, data_y
