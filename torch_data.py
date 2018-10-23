from datetime import date, datetime, timedelta

import numpy as np
from torch.utils import data
from sklearn.preprocessing import MinMaxScaler

from fetcher import Fetcher


class StockDataset(data.Dataset):
    def __init__(self, symbols, start, end=date.today(), data_len=5, scale="D"):
        assert isinstance(symbols, list)
        self.datasets = []
        for s in symbols:
            stock_dataset = StockPriceRegression(s, start, end, data_len, scale)
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


def chunks(data, data_len):
    for idx in range(len(data) - data_len + 1):
        yield data[idx:idx+data_len]

class StockPriceData():

    def __init__(self, symbol, start, end=date.today(), data_len=5, scale="D"):
        usecols = ["open", "high", "low", "close", "volume"]
        self.__data_df = Fetcher().fetch(symbol, start, end)[usecols]
        if scale != "D":
            self.__data_df = self.__data_df.resample(
                    scale, 
                    how=
                    {"open": 'first',   
                     "high": 'max',
                     "low": 'min',
                     "close": 'last',
                     "volume": 'sum',
                     })[:-1]
        self.__data_df_norm = self.__data_df.copy()
        self.__data_df_norm['open'] = MinMaxScaler().fit_transform(self.__data_df.open.values.reshape(-1, 1))
        self.__data_df_norm['high'] = MinMaxScaler().fit_transform(self.__data_df.high.values.reshape(-1, 1))
        self.__data_df_norm['low'] = MinMaxScaler().fit_transform(self.__data_df.low.values.reshape(-1, 1))
        self.__data_df_norm['close'] = MinMaxScaler().fit_transform(self.__data_df.close.values.reshape(-1, 1))
        self.__data_df_norm['volume'] = MinMaxScaler().fit_transform(self.__data_df.volume.values.reshape(-1, 1))
        self.data_len = data_len

    def denormalize(self, norm_value):
        origin_values = self.__data_df["close"].values.reshape(-1, 1)
        norm_value = norm_value.reshape(-1, 1)
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit_transform(origin_values)
        denorm_value = min_max_scaler.inverse_transform(norm_value)
        return denorm_value

    def get_train_datas(self, scale="D"):
        data = self.__data_df_norm.values
        data_x = np.array(list(chunks(data, self.data_len))[:-1])
        return data_x

    def get_train_targets(self):
        pass

    def get_test_datas(self):
        data = self.__data_df_norm.values
        data_x = np.array(list(chunks(data, self.data_len))[-1])
        return data_x

    def get_raw_datas(self):
        return self.__data_df.values

    def get_norm_datas(self):
        return self.__data_df_norm.values


class StockPriceRegression(StockPriceData):

    def __init__(self, symbol, start, end=date.today(), data_len=5, scale="D"):
        super().__init__(symbol, start, end, data_len, scale)

    def get_train_targets(self):
        data = self.get_norm_datas()
        data_y = []
        for idx in range(len(data) - self.data_len):
            data_y.append(data[idx + self.data_len, 3])
        data_y = np.asarray(data_y).reshape(-1, 1)
        return data_y


class StockPriceChange(StockPriceData):
    def __init__(self, symbol, start, end=date.today(), data_len=5, scale="D"):
        super().__init__(symbol, start, end, data_len, scale)

    def get_train_targets(self):
        data = self.get_norm_datas()
        data_y = []
        for idx in range(len(data) - self.data_len):
            change = (data[idx + self.data_len, 3] - data[idx + self.data_len - 1, 3]) / data[idx + self.data_len - 1, 3]
            data_y.append(change)
        data_y = np.asarray(data_y).reshape(-1, 1)
        return data_y
