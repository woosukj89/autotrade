"""
Shared data fetch for the improvement/ classifier work: SPY/VIX prices plus
the macro series carried over from archive/bond_rate_v2 (credit spread proxy,
real yields) that validated well this session. Fetched once, cached to
parquet, and sliced by each signal/backtest as needed.
"""
import os
import sys
from datetime import datetime
from typing import Dict

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from providers import FREDProvider, CachedProvider  # live, unmodified provider module

CACHE_DIR = os.path.join(os.path.dirname(__file__), '.data_cache')
DATA_START = datetime(2003, 1, 1)
DATA_END = datetime(2025, 12, 31)


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f'{name}.csv')


def fetch_price_series(ticker: str) -> pd.Series:
    path = _cache_path(f'price_{ticker}')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)['Close']
    df = yf.download(ticker, start=DATA_START, end=DATA_END, progress=False, auto_adjust=True)
    closes = df['Close']
    if hasattr(closes, 'squeeze'):
        closes = closes.squeeze()
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    closes.to_frame(name='Close').to_csv(path)
    return closes


def fetch_fred_daily(friendly_name: str, fred_api_key: str = None) -> pd.Series:
    path = _cache_path(f'fred_{friendly_name}')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)['value']
    fred = FREDProvider(api_key=fred_api_key)  # CachedProvider skipped - also needs parquet
    series = fred.get_series(friendly_name, DATA_START, DATA_END, frequency='D')
    series.to_frame(name='value').to_csv(path)
    return series


def load_all(fred_api_key: str = None) -> Dict[str, pd.Series]:
    """Everything the signals module needs, aligned to daily frequency
    (forward-filled from whatever native frequency each series has)."""
    spy = fetch_price_series('SPY')
    vix = fetch_price_series('^VIX')
    baa10y = fetch_fred_daily('baa10y', fred_api_key)

    idx = spy.index
    return {
        'spy': spy,
        'vix': vix.reindex(idx, method='ffill'),
        'baa10y': baa10y.reindex(idx, method='ffill'),
    }


if __name__ == '__main__':
    data = load_all()
    for name, series in data.items():
        print(f'{name}: {len(series)} rows, {series.index.min().date()} -> {series.index.max().date()}, '
              f'{series.isna().sum()} NaN')
