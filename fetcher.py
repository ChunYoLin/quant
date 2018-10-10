from datetime import date, datetime, timedelta

import pandas as pd
import pandas_datareader.data as web


class Fetcher:
    def __init__(self):
        pass

    def fetch(self, stock, start, end=date.today()):
        data = web.QuandlReader(stock, start, end, api_key="9rtGDsF3JKZLcWAxGbY5").read()
        data.rename(columns=lambda x: x.lower(), inplace=True)
        last_date_in_first_fetch = data.index[0].date()
        if end > data.index[0].date():
            start_date_in_second_fetch = last_date_in_first_fetch + timedelta(days=1)
            recent_data = web.DataReader(stock, 'iex', start_date_in_second_fetch, end)
        data = pd.concat([data, recent_data], axis=0, sort=False)
        data.index = pd.to_datetime(data.index)
        return data.sort_index()
