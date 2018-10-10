import unittest

from fetcher import Fetcher
from torch_data import StockDataset
from torch_data import get_train_datas

class TestStockDataset(unittest.TestCase):

    def test_ctor(self):
        amd_dataset = StockDataset("AMD", "1993-01-01", "1994-01-05", data_len = 5)

    def test_len(self):
        data_len = 5
        amd_dataset = StockDataset("AMD", "1993-01-01", "1994-01-05", data_len = data_len)
        amd_df = amd_dataset.data_df
        self.assertEqual(len(amd_df)-data_len, len(amd_dataset))

    def test_get_train_datas(self):
        data_len = 5
        amd_data_df = StockDataset("AMD", "1993-01-01", "1994-01-05", data_len = data_len).data_df
        data = amd_data_df.values

        data_x, data_y = get_train_datas(data, data_len)
        self.assertEqual(amd_data_df.iloc[:data_len].values.tolist(), data_x[0].tolist())
        self.assertEqual(amd_data_df.iloc[data_len]["close"], data_y[0])
