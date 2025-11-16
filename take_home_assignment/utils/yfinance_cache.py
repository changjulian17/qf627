"""Persistent cache for Yahoo Finance data to ensure reproducible results.

Yahoo Finance can return slightly different historical data between API calls
(due to splits, dividends adjustments, or data corrections). This cache stores
fetched data to disk to ensure consistency across runs.
"""
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime


class YFinanceCache:
    """Persistent cache for Yahoo Finance historical data."""
    
    def __init__(self, cache_dir='data/yfinance_cache'):
        """Initialize cache with storage directory.
        
        Args:
            cache_dir: Directory to store cached data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, ticker, start_date, end_date):
        """Generate cache key from ticker and date range.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache key string
        """
        # Normalize ticker for filename (remove special chars)
        clean_ticker = ticker.replace('^', 'idx_').replace('=', '_').replace('-', '_').replace('.', '_')
        start_str = pd.Timestamp(start_date).strftime('%Y%m%d')
        end_str = pd.Timestamp(end_date).strftime('%Y%m%d')
        return f"{clean_ticker}_{start_str}_{end_str}"
    
    def get(self, ticker, start_date, end_date):
        """Retrieve cached data if available.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Cached Series or None if not found
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                # print(f"[CACHE] Loaded cached data for {ticker} from {cache_file.name}")
                return data
            except Exception as e:
                print(f"[CACHE] Warning: Failed to load cache for {ticker}: {e}")
                return None
        return None
    
    def set(self, ticker, start_date, end_date, data):
        """Store data in cache.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            data: pandas Series to cache
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"[CACHE] Saved data for {ticker} to {cache_file.name}")
        except Exception as e:
            print(f"[CACHE] Warning: Failed to save cache for {ticker}: {e}")
    
    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob('*.pkl'):
            cache_file.unlink()
        print(f"[CACHE] Cleared all cached data from {self.cache_dir}")


# Global cache instance
_cache = YFinanceCache()


def get_cached_ticker_data(ticker, start_date, end_date, fetch_func):
    """Get ticker data from cache or fetch if not available.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
        fetch_func: Function to fetch data if not cached (should return pandas Series or DataFrame)
        
    Returns:
        pandas Series or DataFrame of price data
    """
    # Try to get from cache first
    data = _cache.get(ticker, start_date, end_date)
    
    if data is not None:
        return data
    
    # Fetch fresh data
    print(f"[CACHE] Fetching fresh data for {ticker}...")
    data = fetch_func()
    
    # Store in cache
    if data is not None and not (isinstance(data, pd.Series) and data.empty) and not (isinstance(data, pd.DataFrame) and data.empty):
        _cache.set(ticker, start_date, end_date, data)
    
    return data


def clear_yfinance_cache():
    """Clear all Yahoo Finance cached data."""
    _cache.clear()
