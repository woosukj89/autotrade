"""
Data Providers for Market Regime Indicators
============================================

Provides an interface-based approach to fetching economic and market data
from various sources (FRED, Yahoo Finance, Quandl, etc.).

Setup Requirements:
-------------------
1. FRED (Federal Reserve Economic Data) - FREE
   - Register at: https://fred.stlouisfed.org/docs/api/api_key.html
   - Set environment variable: FRED_API_KEY=your_key_here
   - Or pass api_key to FREDProvider

2. Yahoo Finance - FREE, NO KEY REQUIRED
   - Uses yfinance library

3. Quandl (now Nasdaq Data Link) - FREE TIER AVAILABLE
   - Register at: https://data.nasdaq.com/sign-up
   - Set environment variable: QUANDL_API_KEY=your_key_here

Install dependencies:
    pip install fredapi pandas-datareader yfinance
"""

import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'W'
    ) -> pd.Series:
        """
        Fetch a time series.

        Args:
            series_id: Identifier for the series (provider-specific)
            start_date: Start date for data
            end_date: End date for data
            frequency: 'D' (daily), 'W' (weekly), 'M' (monthly)

        Returns:
            pd.Series with DatetimeIndex
        """
        pass

    @abstractmethod
    def get_available_series(self) -> Dict[str, str]:
        """Return dict of {series_id: description}."""
        pass

    def resample_to_frequency(self, series: pd.Series, frequency: str = 'W') -> pd.Series:
        """Resample series to target frequency."""
        if frequency == 'W':
            return series.resample('W-FRI').last().dropna()
        elif frequency == 'M':
            return series.resample('ME').last().dropna()
        elif frequency == 'D':
            return series
        else:
            return series.resample(frequency).last().dropna()


class FREDProvider(DataProvider):
    """
    Federal Reserve Economic Data (FRED) provider.

    FREE to use with API key registration.
    Register at: https://fred.stlouisfed.org/docs/api/api_key.html

    Key Series IDs:
    - DGS10: 10-Year Treasury Constant Maturity Rate
    - DGS3MO: 3-Month Treasury Bill Rate
    - BAMLH0A0HYM2: ICE BofA US High Yield Index OAS
    - BAMLC0A0CM: ICE BofA US Corporate Index OAS
    - M2SL: M2 Money Stock
    - CPIAUCSL: Consumer Price Index
    - FEDFUNDS: Federal Funds Rate
    - T10YIE: 10-Year Breakeven Inflation Rate
    - VIXCLS: CBOE Volatility Index
    """

    SERIES_MAP = {
        'y10': 'DGS10',
        'y3m': 'DGS3MO',
        'y2': 'DGS2',
        'hy_oas': 'BAMLH0A0HYM2',
        'ig_oas': 'BAMLC0A0CM',
        'm2': 'M2SL',
        'cpi': 'CPIAUCSL',
        'fed_funds': 'FEDFUNDS',
        'breakeven_10y': 'T10YIE',
        'vix': 'VIXCLS',
        'unemployment': 'UNRATE',
        'real_gdp': 'GDPC1',
        'sp500': 'SP500',
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED provider.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        self._fred = None

    def _get_fred(self):
        """Lazy initialization of FRED client."""
        if self._fred is None:
            try:
                from fredapi import Fred
                if not self.api_key:
                    raise ValueError(
                        "FRED API key required. Register free at: "
                        "https://fred.stlouisfed.org/docs/api/api_key.html\n"
                        "Then set FRED_API_KEY environment variable."
                    )
                self._fred = Fred(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install fredapi: pip install fredapi")
        return self._fred

    def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'W'
    ) -> pd.Series:
        """Fetch series from FRED."""
        fred = self._get_fred()

        # Map friendly name to FRED ID if needed
        fred_id = self.SERIES_MAP.get(series_id, series_id)

        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 3)
        if end_date is None:
            end_date = datetime.now()

        series = fred.get_series(
            fred_id,
            observation_start=start_date,
            observation_end=end_date
        )

        return self.resample_to_frequency(series, frequency)

    def get_available_series(self) -> Dict[str, str]:
        return {
            'DGS10': '10-Year Treasury Rate',
            'DGS3MO': '3-Month Treasury Rate',
            'DGS2': '2-Year Treasury Rate',
            'BAMLH0A0HYM2': 'High Yield OAS Spread',
            'BAMLC0A0CM': 'Investment Grade OAS Spread',
            'M2SL': 'M2 Money Stock',
            'CPIAUCSL': 'Consumer Price Index',
            'FEDFUNDS': 'Federal Funds Rate',
            'T10YIE': '10-Year Breakeven Inflation',
            'VIXCLS': 'VIX Index',
            'UNRATE': 'Unemployment Rate',
            'SP500': 'S&P 500 Index',
        }


class YahooFinanceProvider(DataProvider):
    """
    Yahoo Finance data provider.

    FREE, NO API KEY REQUIRED.

    Key Series IDs:
    - ^VIX: CBOE Volatility Index
    - ^VIX3M: CBOE 3-Month Volatility Index
    - ^GSPC: S&P 500 Index
    - ^TNX: 10-Year Treasury Yield
    - ^IRX: 13-Week Treasury Bill
    """

    SERIES_MAP = {
        'vix': '^VIX',
        'vix_3m': '^VIX3M',
        'sp500': '^GSPC',
        'y10': '^TNX',
        'y3m': '^IRX',
        'nasdaq': '^IXIC',
        'russell2000': '^RUT',
    }

    def __init__(self):
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError("Please install yfinance: pip install yfinance")

    def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'W'
    ) -> pd.Series:
        """Fetch series from Yahoo Finance."""
        # Map friendly name to Yahoo ticker
        ticker = self.SERIES_MAP.get(series_id, series_id)

        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 3)
        if end_date is None:
            end_date = datetime.now()

        data = self._yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data found for {ticker}")

        series = data['Close']
        if hasattr(series, 'squeeze'):
            series = series.squeeze()

        series.name = series_id
        return self.resample_to_frequency(series, frequency)

    def get_available_series(self) -> Dict[str, str]:
        return {
            '^VIX': 'CBOE Volatility Index',
            '^VIX3M': 'CBOE 3-Month Volatility Index',
            '^GSPC': 'S&P 500 Index',
            '^TNX': '10-Year Treasury Yield',
            '^IRX': '13-Week Treasury Bill',
            '^IXIC': 'NASDAQ Composite',
            '^RUT': 'Russell 2000',
        }


class QuandlProvider(DataProvider):
    """
    Nasdaq Data Link (formerly Quandl) provider.

    FREE TIER available with registration.
    Register at: https://data.nasdaq.com/sign-up

    Key datasets:
    - MULTPL/SHILLER_PE_RATIO_MONTH: Shiller CAPE Ratio
    - FRED/*: FRED data via Quandl
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('QUANDL_API_KEY')
        try:
            import nasdaqdatalink
            self._quandl = nasdaqdatalink
            if self.api_key:
                nasdaqdatalink.ApiConfig.api_key = self.api_key
        except ImportError:
            try:
                import quandl
                self._quandl = quandl
                if self.api_key:
                    quandl.ApiConfig.api_key = self.api_key
            except ImportError:
                raise ImportError(
                    "Please install nasdaq-data-link: pip install nasdaq-data-link"
                )

    def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'W'
    ) -> pd.Series:
        """Fetch series from Quandl/Nasdaq Data Link."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 3)
        if end_date is None:
            end_date = datetime.now()

        data = self._quandl.get(
            series_id,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if isinstance(data, pd.DataFrame):
            series = data.iloc[:, 0]
        else:
            series = data

        series.name = series_id
        return self.resample_to_frequency(series, frequency)

    def get_available_series(self) -> Dict[str, str]:
        return {
            'MULTPL/SHILLER_PE_RATIO_MONTH': 'Shiller CAPE Ratio',
            'FRED/DGS10': '10-Year Treasury (via Quandl)',
        }


class CompositeProvider(DataProvider):
    """
    Combines multiple providers with fallback logic.

    Tries providers in order until one succeeds.
    """

    def __init__(self, providers: list[DataProvider]):
        self.providers = providers

    def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'W'
    ) -> pd.Series:
        """Try each provider until one succeeds."""
        errors = []
        for provider in self.providers:
            try:
                return provider.get_series(series_id, start_date, end_date, frequency)
            except Exception as e:
                errors.append(f"{provider.__class__.__name__}: {e}")

        raise ValueError(
            f"All providers failed for {series_id}:\n" +
            "\n".join(errors)
        )

    def get_available_series(self) -> Dict[str, str]:
        combined = {}
        for provider in self.providers:
            combined.update(provider.get_available_series())
        return combined


class CachedProvider(DataProvider):
    """
    Wraps a provider with local caching to reduce API calls.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache_dir: str = '.data_cache',
        cache_hours: int = 24
    ):
        self.provider = provider
        self.cache_dir = cache_dir
        self.cache_hours = cache_hours
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, series_id: str, frequency: str) -> str:
        safe_id = series_id.replace('/', '_').replace('^', '_')
        return os.path.join(self.cache_dir, f"{safe_id}_{frequency}.parquet")

    def _is_cache_valid(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return (datetime.now() - mtime).total_seconds() < self.cache_hours * 3600

    def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'W'
    ) -> pd.Series:
        """Fetch from cache if valid, otherwise from provider."""
        cache_path = self._cache_path(series_id, frequency)

        if self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                series = df.iloc[:, 0]
                # Filter by date range if specified
                if start_date:
                    series = series[series.index >= start_date]
                if end_date:
                    series = series[series.index <= end_date]
                return series
            except Exception:
                pass  # Fall through to fetch

        # Fetch from provider
        series = self.provider.get_series(series_id, start_date, end_date, frequency)

        # Cache result
        try:
            df = series.to_frame()
            df.to_parquet(cache_path)
        except Exception:
            pass  # Caching failure is not critical

        return series

    def get_available_series(self) -> Dict[str, str]:
        return self.provider.get_available_series()


# =============================================================================
# Regime Data Fetcher - Combines all indicators
# =============================================================================

class RegimeDataFetcher:
    """
    Fetches all indicators needed for bear score calculation.

    Uses FRED for economic data and Yahoo Finance for market data.
    """

    # Default FRED series mappings
    FRED_SERIES = {
        'y10': 'DGS10',           # 10-Year Treasury
        'y3m': 'DGS3MO',          # 3-Month Treasury
        'hy_oas': 'BAMLH0A0HYM2', # High Yield OAS
        'ig_oas': 'BAMLC0A0CM',   # Investment Grade OAS
        'm2': 'M2SL',             # M2 Money Stock
        'cpi': 'CPIAUCSL',        # CPI
        'fed_funds': 'FEDFUNDS',  # Fed Funds Rate
    }

    # Default Yahoo series mappings
    YAHOO_SERIES = {
        'vix': '^VIX',
        'vix_3m': '^VIX3M',
        'sp500': '^GSPC',
    }

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_hours: int = 24
    ):
        """
        Initialize the regime data fetcher.

        Args:
            fred_api_key: FRED API key (or set FRED_API_KEY env var)
            use_cache: Whether to cache data locally
            cache_hours: Hours before cache expires
        """
        self.fred_api_key = fred_api_key

        # Initialize providers
        self._yahoo = YahooFinanceProvider()

        # FRED provider (may fail if no API key)
        self._fred = None
        try:
            self._fred = FREDProvider(api_key=fred_api_key)
        except Exception as e:
            print(f"Warning: FRED provider not available: {e}")

        # Apply caching if requested
        if use_cache:
            self._yahoo = CachedProvider(self._yahoo, cache_hours=cache_hours)
            if self._fred:
                self._fred = CachedProvider(self._fred, cache_hours=cache_hours)

    def fetch_all(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'W'
    ) -> Dict[str, pd.Series]:
        """
        Fetch all regime indicators.

        Returns dict compatible with compute_bear_score().
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 3)

        result = {}

        # --- Treasury Yields ---
        try:
            if self._fred:
                result['y10'] = self._fred.get_series('DGS10', start_date, end_date, frequency)
                result['y3m'] = self._fred.get_series('DGS3MO', start_date, end_date, frequency)
            else:
                # Fallback to Yahoo (less reliable for rates)
                result['y10'] = self._yahoo.get_series('^TNX', start_date, end_date, frequency)
                result['y3m'] = self._yahoo.get_series('^IRX', start_date, end_date, frequency)
        except Exception as e:
            print(f"Warning: Could not fetch treasury yields: {e}")

        # --- Credit Spread ---
        try:
            if self._fred:
                hy = self._fred.get_series('BAMLH0A0HYM2', start_date, end_date, frequency)
                ig = self._fred.get_series('BAMLC0A0CM', start_date, end_date, frequency)
                result['credit_spread'] = hy - ig  # HY-IG spread
            else:
                # No good Yahoo alternative for credit spreads
                print("Warning: Credit spread requires FRED API key")
        except Exception as e:
            print(f"Warning: Could not fetch credit spread: {e}")

        # --- M2 YoY Growth ---
        try:
            if self._fred:
                m2 = self._fred.get_series('M2SL', start_date, end_date, 'M')
                # Calculate YoY growth
                m2_yoy = m2.pct_change(periods=12) * 100
                # Resample to weekly
                result['m2_yoy'] = m2_yoy.resample('W-FRI').ffill().dropna()
            else:
                print("Warning: M2 data requires FRED API key")
        except Exception as e:
            print(f"Warning: Could not fetch M2: {e}")

        # --- Real Rate ---
        try:
            if self._fred:
                fed = self._fred.get_series('FEDFUNDS', start_date, end_date, 'M')
                cpi = self._fred.get_series('CPIAUCSL', start_date, end_date, 'M')
                cpi_yoy = cpi.pct_change(periods=12) * 100
                real_rate = fed - cpi_yoy
                result['real_rate'] = real_rate.resample('W-FRI').ffill().dropna()
            else:
                print("Warning: Real rate requires FRED API key")
        except Exception as e:
            print(f"Warning: Could not fetch real rate: {e}")

        # --- VIX ---
        try:
            result['vix'] = self._yahoo.get_series('^VIX', start_date, end_date, frequency)
        except Exception as e:
            print(f"Warning: Could not fetch VIX: {e}")

        # --- VIX 3-Month ---
        try:
            result['vix_3m'] = self._yahoo.get_series('^VIX3M', start_date, end_date, frequency)
        except Exception as e:
            # Fallback: use VIX with slight adjustment
            if 'vix' in result:
                result['vix_3m'] = result['vix'] * 0.95
            print(f"Warning: Could not fetch VIX3M, using approximation")

        # --- Breadth (% above 200 DMA) ---
        try:
            # Calculate from S&P 500
            sp500 = self._yahoo.get_series('^GSPC', start_date, end_date, 'D')
            ma200 = sp500.rolling(200).mean()
            pct_above = ((sp500 > ma200).rolling(20).mean() * 100)
            result['pct_above_200dma'] = pct_above.resample('W-FRI').last().dropna()
        except Exception as e:
            print(f"Warning: Could not calculate breadth: {e}")

        # --- CAPE Percentile ---
        try:
            # CAPE is harder to get - use S&P 500 P/E as proxy
            # For real CAPE, consider Quandl: MULTPL/SHILLER_PE_RATIO_MONTH
            sp500 = self._yahoo.get_series('^GSPC', start_date, end_date, frequency)
            # Approximate CAPE percentile based on price level
            # This is a rough proxy - real CAPE requires earnings data
            pct_rank = sp500.rank(pct=True) * 100
            result['cape_percentile'] = pct_rank
        except Exception as e:
            print(f"Warning: Could not calculate CAPE percentile: {e}")

        return result

    def get_aligned_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = 'W'
    ) -> Dict[str, pd.Series]:
        """
        Fetch all data and align to common dates.

        Returns dict with all series having the same index.
        """
        data = self.fetch_all(start_date, end_date, frequency)

        if not data:
            raise ValueError("No data could be fetched")

        # Find common date range
        all_indices = [s.index for s in data.values()]
        common_start = max(idx.min() for idx in all_indices)
        common_end = min(idx.max() for idx in all_indices)

        # Align all series
        aligned = {}
        for key, series in data.items():
            aligned[key] = series[(series.index >= common_start) & (series.index <= common_end)]

        return aligned


# =============================================================================
# Factory function for easy setup
# =============================================================================

def create_default_fetcher(
    fred_api_key: Optional[str] = None,
    use_cache: bool = True
) -> RegimeDataFetcher:
    """
    Create a regime data fetcher with sensible defaults.

    Args:
        fred_api_key: FRED API key. If None, reads from FRED_API_KEY env var.
        use_cache: Whether to cache API responses (recommended)

    Returns:
        RegimeDataFetcher instance
    """
    return RegimeDataFetcher(
        fred_api_key=fred_api_key,
        use_cache=use_cache,
        cache_hours=24
    )
