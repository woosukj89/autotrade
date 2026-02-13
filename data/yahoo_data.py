"""
Yahoo Finance Data Provider
===========================

Unified data source using Yahoo Finance API for both ticker discovery
and fundamental data. This ensures consistency between ticker symbols
and their corresponding financial data.

Usage:
    from data.yahoo_data import YahooDataProvider

    provider = YahooDataProvider()

    # Get universe of tickers
    tickers = provider.get_us_equity_universe()

    # Get fundamentals for a ticker
    fundamentals = provider.get_fundamentals("AAPL")

    # Get fundamentals for multiple tickers
    all_funds = provider.get_fundamentals_batch(tickers)
"""

import os
import sqlite3
import warnings
import logging
import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

# Suppress yfinance warnings
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*YFPricesMissingError.*')
warnings.filterwarnings('ignore', message='.*possibly delisted.*')


@dataclass
class StockFundamentals:
    """Fundamental data for a stock."""
    ticker: str
    sector: str
    industry: str
    market_cap: float

    # Profitability
    roe: Optional[float] = None
    roa: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    profit_margin: Optional[float] = None

    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None

    # Balance Sheet
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None

    # Cash Flow
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None

    # Dividends
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

    # Valuation
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None

    # Other
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None


class YahooDataProvider:
    """
    Unified data provider using Yahoo Finance API.

    Provides both ticker discovery and fundamental data from a single source
    to ensure consistency.
    """

    # Major US indices and their Yahoo Finance symbols for constituent lookup
    INDEX_SYMBOLS = {
        'sp500': '^GSPC',
        'nasdaq100': '^NDX',
        'dow30': '^DJI',
        'russell2000': '^RUT',
    }

    # S&P 500 constituents (curated list - updated periodically)
    # This serves as the primary universe when Wikipedia scraping fails
    SP500_TICKERS = {
        # Technology
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'ORCL', 'CSCO', 'CRM',
        'AMD', 'ADBE', 'ACN', 'INTC', 'IBM', 'QCOM', 'TXN', 'AMAT', 'NOW', 'INTU',
        'LRCX', 'MU', 'ADI', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'APH', 'FTNT', 'PANW',
        'CRWD', 'MSI', 'NXPI', 'KEYS', 'ON', 'MPWR', 'ANSS', 'CDW', 'CTSH', 'HPQ',
        'ANET', 'TEL', 'IT', 'ZBRA', 'GLW', 'STX', 'WDC', 'JNPR', 'NTAP', 'TYL',
        'ENPH', 'SEDG', 'FSLR', 'GEN', 'AKAM', 'FFIV', 'SWKS', 'QRVO', 'EPAM', 'LDOS',

        # Healthcare
        'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'VRTX', 'ISRG', 'MDT', 'REGN', 'BSX', 'SYK', 'CVS', 'CI',
        'ELV', 'HUM', 'MCK', 'ZTS', 'BDX', 'DXCM', 'IDXX', 'IQV', 'MTD', 'EW',
        'A', 'RMD', 'ALGN', 'WST', 'HOLX', 'BAX', 'CAH', 'BIIB', 'MRNA', 'ILMN',
        'TECH', 'MOH', 'CNC', 'HCA', 'LH', 'DGX', 'PKI', 'VTRS', 'XRAY', 'HSIC',

        # Financials
        'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK',
        'C', 'AXP', 'SCHW', 'CB', 'MMC', 'PGR', 'CME', 'ICE', 'AON', 'USB',
        'PNC', 'TFC', 'AIG', 'MET', 'PRU', 'TRV', 'AFL', 'AMP', 'ALL', 'MSCI',
        'MCO', 'COF', 'BK', 'STT', 'TROW', 'NTRS', 'DFS', 'FDS', 'NDAQ', 'CBOE',
        'FITB', 'KEY', 'CFG', 'RF', 'HBAN', 'MTB', 'SIVB', 'ZION', 'CMA', 'WRB',

        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
        'ORLY', 'AZO', 'ROST', 'MAR', 'HLT', 'YUM', 'DHI', 'LEN', 'GM', 'F',
        'APTV', 'EBAY', 'ETSY', 'BBY', 'DRI', 'POOL', 'GPC', 'LVS', 'WYNN', 'MGM',
        'CCL', 'RCL', 'EXPE', 'ULTA', 'DG', 'DLTR', 'TSCO', 'WSM', 'GRMN', 'PHM',
        'NVR', 'TPR', 'VFC', 'PVH', 'HAS', 'BWA', 'LEG', 'WHR', 'AAP', 'KMX',

        # Consumer Staples
        'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
        'EL', 'GIS', 'K', 'SYY', 'HSY', 'KHC', 'STZ', 'ADM', 'KR', 'MKC',
        'CHD', 'HRL', 'CPB', 'SJM', 'CAG', 'BG', 'TSN', 'LW', 'TAP', 'CLX',

        # Industrials
        'CAT', 'DE', 'UNP', 'UPS', 'RTX', 'HON', 'BA', 'LMT', 'GE', 'MMM',
        'ADP', 'ITW', 'EMR', 'ETN', 'PH', 'CTAS', 'NSC', 'CSX', 'JCI', 'PCAR',
        'CMI', 'ROK', 'GD', 'NOC', 'TT', 'CARR', 'OTIS', 'FDX', 'WM', 'RSG',
        'FAST', 'VRSK', 'IR', 'AME', 'DOV', 'GWW', 'SWK', 'IEX', 'PNR', 'XYL',
        'GNRC', 'LDOS', 'J', 'PWR', 'WAB', 'EXPD', 'CHRW', 'DAL', 'UAL', 'LUV',

        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'PSX', 'VLO', 'OXY',
        'WMB', 'KMI', 'HAL', 'DVN', 'FANG', 'HES', 'BKR', 'OKE', 'TRGP', 'APA',
        'MRO', 'CTRA', 'EQT', 'MTDR',

        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'WEC',
        'ES', 'ED', 'AWK', 'DTE', 'PPL', 'AEE', 'EIX', 'FE', 'CNP', 'CMS',
        'NI', 'EVRG', 'ATO', 'NRG', 'PNW', 'LNT', 'CEG', 'AES',

        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
        'EQR', 'VICI', 'VTR', 'ARE', 'ESS', 'MAA', 'UDR', 'PEAK', 'HST', 'KIM',
        'REG', 'SLG', 'BXP', 'CPT', 'IRM', 'SBAC', 'INVH', 'WY', 'EXR', 'CBRE',

        # Communication Services
        'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR',
        'ATVI', 'EA', 'TTWO', 'MTCH', 'WBD', 'PARA', 'FOX', 'FOXA', 'NWS', 'NWSA',
        'LYV', 'OMC', 'IPG',

        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DD', 'DOW', 'PPG',
        'VMC', 'MLM', 'ALB', 'CTVA', 'FMC', 'IFF', 'CE', 'EMN', 'CF', 'MOS',
        'BALL', 'PKG', 'IP', 'WRK', 'SEE', 'AVY', 'AMCR',
    }

    # Priority tickers for high beta strategy (known outperformers)
    HIGH_BETA_PRIORITY = {
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'AVGO', 'CRM',
        'NFLX', 'ADBE', 'NOW', 'LRCX', 'ANET', 'KLAC', 'CDNS', 'FTNT', 'MU', 'PANW',
        'AMAT', 'SNPS', 'MRVL', 'CRWD', 'INTU', 'ISRG', 'REGN', 'VRTX', 'MSCI', 'CTAS',
    }

    # Priority tickers for bear beta strategy (known defensive stocks)
    DEFENSIVE_PRIORITY = {
        # Consumer Staples
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL', 'KMB', 'GIS',
        'K', 'CPB', 'SJM', 'MKC', 'CHD', 'HSY', 'HRL', 'TSN', 'CAG', 'KHC',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
        'GILD', 'MDT', 'ISRG', 'CVS', 'CI', 'HUM', 'MCK', 'CAH', 'VTRS', 'ZTS',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'WEC',
        'ES', 'ED', 'DTE', 'PPL', 'AEE', 'CNP', 'CMS', 'NI', 'EVRG', 'ATO',
        # Gold/Safe Haven
        'NEM', 'GOLD', 'AEM', 'FNV', 'WPM', 'RGLD',
    }

    def __init__(self, cache_db: str = None, cache_days: int = 7):
        """
        Initialize the Yahoo data provider.

        Args:
            cache_db: Path to SQLite cache database (optional)
            cache_days: Number of days to cache data
        """
        self.cache_days = cache_days
        self.cache_db = cache_db

        if cache_db:
            self._init_cache()

    def _init_cache(self):
        """Initialize cache database tables."""
        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS yf_fundamentals_cache (
                ticker TEXT PRIMARY KEY,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                roe REAL,
                roa REAL,
                operating_margin REAL,
                gross_margin REAL,
                profit_margin REAL,
                revenue_growth REAL,
                earnings_growth REAL,
                debt_to_equity REAL,
                current_ratio REAL,
                free_cash_flow REAL,
                operating_cash_flow REAL,
                dividend_yield REAL,
                payout_ratio REAL,
                pe_ratio REAL,
                forward_pe REAL,
                pb_ratio REAL,
                beta REAL,
                fifty_two_week_high REAL,
                fifty_two_week_low REAL,
                last_updated TEXT
            )
        """)
        conn.commit()
        conn.close()

    def get_us_equity_universe(
        self,
        include_sp500: bool = True,
        min_market_cap: float = 1_000_000_000,  # $1B minimum
        sectors: List[str] = None,
    ) -> Set[str]:
        """
        Get a universe of US equity tickers from Yahoo Finance.

        Uses curated S&P 500 list as the base universe.

        Args:
            include_sp500: Include S&P 500 constituents
            min_market_cap: Minimum market cap filter (applied during fundamentals fetch)
            sectors: Optional list of sectors to filter

        Returns:
            Set of ticker symbols
        """
        tickers = set()

        if include_sp500:
            tickers.update(self.SP500_TICKERS)

        # Could add more sources here (e.g., scrape other indices)

        print(f"[YahooData] Universe: {len(tickers)} tickers")
        return tickers

    def get_high_beta_universe(self) -> Set[str]:
        """Get tickers suitable for high beta strategy."""
        universe = self.get_us_equity_universe()
        # Add priority tickers
        universe.update(self.HIGH_BETA_PRIORITY)
        return universe

    def get_defensive_universe(self) -> Set[str]:
        """Get tickers suitable for defensive/bear beta strategy."""
        universe = self.get_us_equity_universe()
        # Add priority defensive tickers
        universe.update(self.DEFENSIVE_PRIORITY)
        return universe

    def get_fundamentals(self, ticker: str, use_cache: bool = True) -> Optional[StockFundamentals]:
        """
        Get fundamental data for a single ticker.

        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached data if available

        Returns:
            StockFundamentals object or None if data unavailable
        """
        # Check cache first
        if use_cache and self.cache_db:
            cached = self._get_from_cache(ticker)
            if cached:
                return cached

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = yf.Ticker(ticker).info

            if not info or 'symbol' not in info:
                return None

            fundamentals = StockFundamentals(
                ticker=ticker,
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=info.get('marketCap') or 0,

                # Profitability
                roe=info.get('returnOnEquity'),
                roa=info.get('returnOnAssets'),
                operating_margin=info.get('operatingMargins'),
                gross_margin=info.get('grossMargins'),
                profit_margin=info.get('profitMargins'),

                # Growth
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),

                # Balance Sheet
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),

                # Cash Flow
                free_cash_flow=info.get('freeCashflow'),
                operating_cash_flow=info.get('operatingCashflow'),

                # Dividends
                dividend_yield=info.get('dividendYield'),
                payout_ratio=info.get('payoutRatio'),

                # Valuation
                pe_ratio=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                pb_ratio=info.get('priceToBook'),

                # Other
                beta=info.get('beta'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
                fifty_two_week_low=info.get('fiftyTwoWeekLow'),
            )

            # Cache the result
            if self.cache_db:
                self._save_to_cache(fundamentals)

            return fundamentals

        except Exception as e:
            return None

    def get_fundamentals_batch(
        self,
        tickers: List[str],
        max_workers: int = 15,
        use_cache: bool = True,
    ) -> Dict[str, StockFundamentals]:
        """
        Get fundamentals for multiple tickers in parallel.

        Args:
            tickers: List of ticker symbols
            max_workers: Max parallel threads
            use_cache: Whether to use cached data

        Returns:
            Dict mapping ticker to StockFundamentals
        """
        results = {}
        tickers_to_fetch = []

        # Check cache first
        if use_cache and self.cache_db:
            for ticker in tickers:
                cached = self._get_from_cache(ticker)
                if cached:
                    results[ticker] = cached
                else:
                    tickers_to_fetch.append(ticker)
        else:
            tickers_to_fetch = list(tickers)

        if tickers_to_fetch:
            print(f"[YahooData] Fetching fundamentals for {len(tickers_to_fetch)} tickers "
                  f"(cached: {len(results)})...")

            def fetch_one(ticker):
                return ticker, self.get_fundamentals(ticker, use_cache=False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for ticker, fund in executor.map(fetch_one, tickers_to_fetch):
                    if fund:
                        results[ticker] = fund

        print(f"[YahooData] Got fundamentals for {len(results)} tickers")
        return results

    def _get_from_cache(self, ticker: str) -> Optional[StockFundamentals]:
        """Get fundamentals from cache if fresh enough."""
        if not self.cache_db:
            return None

        try:
            conn = sqlite3.connect(self.cache_db)
            cutoff = (datetime.now() - timedelta(days=self.cache_days)).strftime("%Y-%m-%d")

            cursor = conn.execute(
                "SELECT * FROM yf_fundamentals_cache WHERE ticker = ? AND last_updated >= ?",
                (ticker, cutoff)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                columns = [
                    'ticker', 'sector', 'industry', 'market_cap',
                    'roe', 'roa', 'operating_margin', 'gross_margin', 'profit_margin',
                    'revenue_growth', 'earnings_growth',
                    'debt_to_equity', 'current_ratio',
                    'free_cash_flow', 'operating_cash_flow',
                    'dividend_yield', 'payout_ratio',
                    'pe_ratio', 'forward_pe', 'pb_ratio',
                    'beta', 'fifty_two_week_high', 'fifty_two_week_low',
                    'last_updated'
                ]
                data = dict(zip(columns, row))
                del data['last_updated']
                return StockFundamentals(**data)
        except Exception:
            pass

        return None

    def _save_to_cache(self, fund: StockFundamentals):
        """Save fundamentals to cache."""
        if not self.cache_db:
            return

        try:
            conn = sqlite3.connect(self.cache_db)
            conn.execute("""
                INSERT OR REPLACE INTO yf_fundamentals_cache
                (ticker, sector, industry, market_cap,
                 roe, roa, operating_margin, gross_margin, profit_margin,
                 revenue_growth, earnings_growth,
                 debt_to_equity, current_ratio,
                 free_cash_flow, operating_cash_flow,
                 dividend_yield, payout_ratio,
                 pe_ratio, forward_pe, pb_ratio,
                 beta, fifty_two_week_high, fifty_two_week_low,
                 last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fund.ticker, fund.sector, fund.industry, fund.market_cap,
                fund.roe, fund.roa, fund.operating_margin, fund.gross_margin, fund.profit_margin,
                fund.revenue_growth, fund.earnings_growth,
                fund.debt_to_equity, fund.current_ratio,
                fund.free_cash_flow, fund.operating_cash_flow,
                fund.dividend_yield, fund.payout_ratio,
                fund.pe_ratio, fund.forward_pe, fund.pb_ratio,
                fund.beta, fund.fifty_two_week_high, fund.fifty_two_week_low,
                datetime.now().strftime("%Y-%m-%d")
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def get_historical_prices(
        self,
        ticker: str,
        days: int = 365,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                end = datetime.now()
                start = end - timedelta(days=days)
                data = yf.download(ticker, start=start, end=end, progress=False)

            if data is not None and len(data) > 0:
                return data
        except Exception:
            pass

        return None


# Singleton instance for convenience
_provider_instance = None

def get_yahoo_provider(cache_db: str = None) -> YahooDataProvider:
    """Get or create a Yahoo data provider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = YahooDataProvider(cache_db=cache_db)
    return _provider_instance
