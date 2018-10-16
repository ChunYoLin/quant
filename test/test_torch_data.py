import unittest

from fetcher import Fetcher
from torch_data import StockPriceRegression, StockPriceChange

class TestStockPriceRegression(unittest.TestCase):

    def test_ctor(self):
        amd_dataset = StockPriceRegression("AMD", "1993-01-01", "1994-01-05", data_len = 5)

    def test_GetTrainDatasTargets_XdataLenIsSameAsYdataLen(self):
        data_len = 5
        amd_dataset = StockPriceRegression("AMD", "1993-01-01", "1994-01-05", data_len = data_len)
        data = amd_dataset.get_raw_datas()

        data_x = amd_dataset.get_train_datas()
        data_y = amd_dataset.get_train_targets()

        self.assertEqual(len(data_x), len(data_y))

    def tess_GetTrainDatasTargets_YdataIsPreviousXdataClosePrice(self):
        data_len = 5
        amd_dataset = StockPriceRegression("AMD", "1993-01-01", "1994-01-05", data_len = data_len)
        data = amd_dataset.get_raw_datas()

        data_x = amd_dataset.get_train_datas()
        data_y = amd_dataset.get_train_targets()

        self.assertEqual(data_y[1], data_x[0, 3])

class TestStockPriceChange(unittest.TestCase):

    def test_ctor(self):
        amd_dataset = StockPriceChange("AMD", "1993-01-01", "1994-01-05", data_len = 5)
