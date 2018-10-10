import unittest
from torch_data import StockDataset


class TestStockDataset(unittest.TestCase):

    def test_ctor(self):
        amd_dataset = StockDataset("AMD", "1993-01-01", "1993-01-05", data_len = 10)
