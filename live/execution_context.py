"""
Shared live-trading ExecutionContext implementations.

Extracted from live_regime_trader.py so both LiveRegimeTrader and
LiveMomentumTrader (and any future live driver) can reuse the same
yfinance/broker-quote data-fetching logic instead of duplicating it.
These classes are strategy-agnostic - they only satisfy the same
duck-typed interface strategies.strategy.ExecutionContext exposes
(.date, .portfolio, get_price(), get_historical_prices(), get_fundamentals()).
"""
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from connectors.base import ExchangeConnector
from strategies.strategy import Portfolio


def _normalize_yf_columns(data: pd.DataFrame) -> pd.DataFrame:
    """yf.download() for a single ticker returns MultiIndex columns
    (field, ticker) on some yfinance versions and plain columns on
    others - CI installs whatever's newest since requirements.txt pins
    only a floor (yfinance>=0.2.0), so this can silently differ from a
    locally-cached version. Without this, data['Close'] returns a
    single-column DataFrame instead of a Series on the MultiIndex
    versions, and float(series.iloc[-1]) blows up with
    "TypeError: ... not 'Series'". strategies/regime_adaptive_strategy.py
    already works around this ad hoc with .squeeze() at each call site;
    normalizing once here means every caller of get_historical_prices()
    gets a consistently plain-columned DataFrame regardless of the
    installed yfinance version's default column shape.
    """
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    return data


class DryRunExecutionContext:
    """
    Execution context for dry-run mode.

    Uses only Yahoo Finance for all data - no Robinhood connection needed.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        date: datetime = None,
    ):
        self.portfolio = portfolio
        self.date = date or datetime.now()
        self._price_cache: Dict[str, float] = {}
        self._historical_cache: Dict[str, pd.DataFrame] = {}

    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price from Yahoo Finance."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(ticker, period='5d', progress=False)
                if data is not None and len(data) > 0:
                    data = _normalize_yf_columns(data)
                    price = float(data['Close'].iloc[-1])
                    self._price_cache[ticker] = price
                    return price
        except Exception:
            pass

        return None

    def get_historical_prices(self, ticker: str, periods: int = 30) -> Optional[pd.DataFrame]:
        """Get historical prices from yfinance."""
        cache_key = f"{ticker}_{periods}"
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Add buffer for weekends/holidays
                days = int(periods * 1.5) + 10
                end_date = self.date
                start_date = end_date - timedelta(days=days)

                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if data is not None and len(data) > 0:
                    data = _normalize_yf_columns(data)
                    # Keep only the requested number of periods
                    data = data.tail(periods)
                    self._historical_cache[cache_key] = data
                    return data
        except Exception:
            pass

        return None

    def get_fundamentals(self, ticker: str, field_name: str = None):
        """Get fundamentals - returns empty DataFrame for live trading."""
        return pd.DataFrame()


class LiveExecutionContext:
    """
    Execution context for live trading.

    Provides the same interface as backtest ExecutionContext but uses
    live data sources (Robinhood for prices, yfinance for historical data).
    """

    def __init__(
        self,
        connector: ExchangeConnector,
        portfolio: Portfolio,
        date: datetime = None,
    ):
        self.connector = connector
        self.portfolio = portfolio
        self.date = date or datetime.now()
        self._price_cache: Dict[str, float] = {}
        self._historical_cache: Dict[str, pd.DataFrame] = {}

    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price from connector or Yahoo Finance."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        try:
            quote = self.connector.get_quote(ticker)
            if quote and quote.last > 0:
                self._price_cache[ticker] = quote.last
                return quote.last
        except Exception:
            pass

        # Fallback to yfinance
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(ticker, period='5d', progress=False)
                if data is not None and len(data) > 0:
                    data = _normalize_yf_columns(data)
                    price = float(data['Close'].iloc[-1])
                    self._price_cache[ticker] = price
                    return price
        except Exception:
            pass

        return None

    def get_historical_prices(self, ticker: str, periods: int = 30) -> Optional[pd.DataFrame]:
        """Get historical prices from yfinance."""
        cache_key = f"{ticker}_{periods}"
        if cache_key in self._historical_cache:
            return self._historical_cache[cache_key]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                days = int(periods * 1.5) + 10
                end_date = self.date
                start_date = end_date - timedelta(days=days)

                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if data is not None and len(data) > 0:
                    data = _normalize_yf_columns(data)
                    data = data.tail(periods)
                    self._historical_cache[cache_key] = data
                    return data
        except Exception:
            pass

        return None

    def get_fundamentals(self, ticker: str, field_name: str = None):
        """Get fundamentals - returns empty DataFrame for live trading."""
        return pd.DataFrame()
