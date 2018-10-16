import unittest

from fetcher import Fetcher
from torch_data import StockPriceData, StockPriceRegression, StockPriceChange

class TestStockPriceData(unittest.TestCase):
    def test_ctor(self):
        amd_dataset = StockPriceData("AMD", "1993-01-01", "1994-01-05", data_len = 5)

    def test_GetTrainDatas_LenOfOneXdataIsSameAsDataLen(self):
        data_len = 5
        amd_dataset = StockPriceData("AMD", "1993-01-01", "1994-01-05", data_len = data_len)

        data_x = amd_dataset.get_train_datas()

        self.assertEqual(len(data_x[-1]), data_len)

    def test_GetTrainDatas_XDataIsSliceOfRawData(self):
        data_len = 5
        amd_dataset = StockPriceData("AMD", "1993-01-01", "1994-01-05", data_len = data_len)
        data = amd_dataset.get_raw_datas()

        data_x = amd_dataset.get_train_datas()

        test_slice_idx = 4
        test_data_idx = 3
        self.assertEqual(data_x[test_slice_idx, test_data_idx, 3], data[test_slice_idx+test_data_idx, 3])
    

class TestStockPriceRegression(unittest.TestCase):

    def test_ctor(self):
        amd_dataset = StockPriceRegression("AMD", "1993-01-01", "1994-01-05", data_len = 5)

    def test_GetTrainTargets_YdataIsPreviousClosePrice(self):
        data_len = 5
        amd_dataset = StockPriceRegression("AMD", "1993-01-01", "1994-01-05", data_len = data_len)
        data = amd_dataset.get_raw_datas()

        data_y = amd_dataset.get_train_targets()

        self.assertEqual(data[data_len, 3], data_y[0])

    def test_GetTrainDatasTargets_XdataLenIsSameAsYdataLen(self):
        data_len = 5
        amd_dataset = StockPriceRegression("AMD", "1993-01-01", "1994-01-05", data_len = data_len)

        data_x = amd_dataset.get_train_datas()
        data_y = amd_dataset.get_train_targets()

        self.assertEqual(len(data_x), len(data_y))


class TestStockPriceChange(unittest.TestCase):

    def test_ctor(self):
        amd_dataset = StockPriceChange("AMD", "1993-01-01", "1994-01-05", data_len = 5)
