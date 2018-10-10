from datetime import date, datetime, timedelta

from fetcher import Fetcher
from torch.utils import data 


class StockDataset(data.Dataset):
    def __init__(self, symbol, start, end=date.today(), data_len=5):
        self.dataframe = Fetcher().fetch(symbol, start, end)
        print(self.dataframe)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
