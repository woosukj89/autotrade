"""
S&P 500 breadth signal: % of constituents trading above their own 200-day
SMA, on each date. Different information content than trend/drawdown/credit
(all of which are single-series measures) - breadth measures how WIDESPREAD
weakness is across individual stocks, which could distinguish a real,
broad-based bear (many stocks grinding down together, e.g. 2022) from a
narrower panic driven by a few headlines/names (e.g. 2011 debt-ceiling).

Reuses the existing curated S&P 500 ticker list from data/yahoo_data.py
(418 tickers) rather than fetching a fresh universe. Caveat, stated
plainly: this is the CURRENT constituent list applied across the whole
20-year history, so it has real survivorship bias (a company delisted or
removed from the index for poor performance before ~today won't be
counted during the period it was actually struggling). Accepted as a
reasonable, standard simplification - true point-in-time index membership
would require a much bigger data source than is available here.
"""
import os
import sys
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from yahoo_data import YahooDataProvider

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.data_cache', 'breadth.csv')
DATA_START = datetime(2003, 1, 1)
DATA_END = datetime(2025, 12, 31)


def fetch_breadth(force: bool = False) -> pd.Series:
    """Returns a daily Series: % of S&P 500 tickers above their own 200-day SMA."""
    if not force and os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)['breadth_pct']

    tickers = sorted(YahooDataProvider.SP500_TICKERS)
    print(f'Downloading {len(tickers)} tickers, {DATA_START.date()} to {DATA_END.date()}...')
    t0 = time.time()
    data = yf.download(tickers, start=DATA_START, end=DATA_END, progress=False,
                        auto_adjust=True, group_by='ticker', threads=True)
    print(f'  downloaded in {time.time()-t0:.1f}s')

    above_sma_flags = []
    n_ok = 0
    for ticker in tickers:
        try:
            closes = data[ticker]['Close']
        except (KeyError, TypeError):
            continue
        if closes.dropna().empty:
            continue
        sma200 = closes.rolling(200).mean()
        above = (closes > sma200)
        above_sma_flags.append(above.rename(ticker))
        n_ok += 1

    print(f'  {n_ok}/{len(tickers)} tickers had usable data')
    combined = pd.concat(above_sma_flags, axis=1)
    breadth_pct = combined.mean(axis=1, skipna=True) * 100
    breadth_pct.name = 'breadth_pct'

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    breadth_pct.to_frame().to_csv(CACHE_PATH)
    print(f'Saved {CACHE_PATH}')
    return breadth_pct


if __name__ == '__main__':
    breadth = fetch_breadth()
    print(f'\n{len(breadth)} rows, {breadth.index.min().date()} -> {breadth.index.max().date()}')
    print(f'Overall mean breadth: {breadth.mean():.1f}%')

    for name, (s, e) in {
        '2008 bear': ('2007-10-09', '2009-03-09'),
        '2020 bear': ('2020-02-19', '2020-03-23'),
        '2022 bear': ('2022-01-03', '2022-10-12'),
        '2011 false+': ('2011-08-04', '2011-10-27'),
        '2010 false+ a': ('2010-05-21', '2010-08-02'),
        '2018 corr': ('2018-09-20', '2018-12-24'),
        '2019-06 calm': ('2019-05-01', '2019-07-01'),
    }.items():
        seg = breadth[(breadth.index >= s) & (breadth.index <= e)]
        if len(seg) == 0:
            print(f'{name:18s} no data')
            continue
        print(f'{name:18s} min={seg.min():5.1f}%  mean={seg.mean():5.1f}%  max={seg.max():5.1f}%')
