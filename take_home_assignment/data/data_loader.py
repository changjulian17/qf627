"""Data loading and preprocessing utilities."""
import yfinance as yf
import pandas as pd
import numpy as np

class DataLoader:
    """Handle data fetching and preprocessing."""
    
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self._data = None
    
    def load_data(self):
        """Download data from Yahoo Finance."""
        self._data = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True
        )
        return self._data
    
    def get_prices(self):
        """Return closing prices."""
        if self._data is None:
            self.load_data()
        return self._data['Close']
    
    def get_ohlcv(self):
        """Return OHLCV data."""
        if self._data is None:
            self.load_data()
        return self._data
    
    @staticmethod
    def clean_data(df, method='ffill'):
        """Handle missing values."""
        return df.fillna(method=method).dropna()