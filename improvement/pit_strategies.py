"""
Point-in-time-aware research strategy variants.

The live-shared HighBetaGrowthStrategy/QualityDefensiveStrategy fetch
fundamentals via self._yahoo_provider.get_fundamentals_batch() - no date
parameter, so a backtest simulating March 2010 scores stocks using
TODAY's (2026) ROE/margins/growth. That's correct for LIVE trading
(today's data IS what you'd use) but a real lookahead bias for every
backtest built on these classes this session.

PitHighBetaGrowthStrategy / PitQualityDefensiveStrategy override just the
fundamentals-fetching step to pull from the SEC-EDGAR-derived point-in-
time `fundamentals` table (improvement/edgar_pipeline.py) instead,
filtered to only what was actually filed as of the simulated date.
Everything else (scoring formula, position sizing, sector caps,
rebalance cadence) is inherited unchanged from the live classes.
"""
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from high_beta_strategy import HighBetaGrowthStrategy
from yahoo_data import StockFundamentals

sys.path.insert(0, os.path.dirname(__file__))
from quality_strategy import QualityDefensiveStrategy

EDGAR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'edgar_fundamentals.sqlite')


class PitFundamentalsStore:
    """Loads the whole point-in-time `fundamentals` + `companies` tables
    into memory once, and computes StockFundamentals-compatible ratios
    as-of any simulated date via fast in-memory lookups (no per-ticker
    SQL query per rebalance - that would be ~400 tickers x ~80
    rebalances x however many fields, too slow against a live SQL
    connection).
    """

    def __init__(self, db_path: str = EDGAR_DB_PATH):
        conn = sqlite3.connect(db_path)
        self._facts = pd.read_sql_query(
            "SELECT ticker, field, date, value FROM fundamentals", conn,
            parse_dates=['date'])
        try:
            self._sectors = dict(conn.execute(
                "SELECT ticker, sector FROM companies WHERE sector IS NOT NULL").fetchall())
        except sqlite3.OperationalError:
            self._sectors = {}
        conn.close()

        # index: (ticker, field) -> DataFrame sorted by date, for fast asof lookups
        self._index: Dict[tuple, pd.DataFrame] = {}
        for (ticker, field), grp in self._facts.groupby(['ticker', 'field']):
            self._index[(ticker, field)] = grp.sort_values('date').reset_index(drop=True)

    def _asof(self, ticker: str, field: str, asof: pd.Timestamp, n: int = 1):
        """Returns up to the last `n` values known as of `asof`, most
        recent first: [(date, value), ...]."""
        df = self._index.get((ticker, field))
        if df is None or df.empty:
            return []
        pos = df['date'].searchsorted(asof, side='right')
        if pos <= 0:
            return []
        rows = df.iloc[max(0, pos - n):pos]
        return list(zip(rows['date'], rows['value']))[::-1]

    def get_fundamentals_asof(self, ticker: str, asof: datetime) -> Optional[StockFundamentals]:
        asof_ts = pd.Timestamp(asof)

        def latest(field):
            vals = self._asof(ticker, field, asof_ts, n=1)
            return vals[0][1] if vals else None

        def yoy_growth(field):
            vals = self._asof(ticker, field, asof_ts, n=2)
            if len(vals) < 2:
                return None
            (_, cur), (_, prev) = vals
            if prev is None or prev == 0 or cur is None:
                return None
            return (cur - prev) / abs(prev)

        revenue = latest('Revenue')
        net_income = latest('NetIncome')
        gross_profit = latest('GrossProfit')
        operating_income = latest('OperatingIncome')
        equity = latest('Equity')
        total_debt = latest('TotalDebt')
        cash_from_ops = latest('CashFromOperations')
        capex = latest('CapitalExpenditures')
        dividends_paid = latest('DividendsPaid')

        if revenue is None and net_income is None and equity is None:
            return None  # no data at all for this ticker as of this date

        roe = (net_income / equity) if (net_income is not None and equity and equity > 0) else None
        operating_margin = (operating_income / revenue) if (operating_income is not None and revenue and revenue > 0) else None
        gross_margin = (gross_profit / revenue) if (gross_profit is not None and revenue and revenue > 0) else None
        debt_to_equity = ((total_debt / equity) * 100) if (total_debt is not None and equity and equity > 0) else None
        free_cash_flow = (cash_from_ops - capex) if (cash_from_ops is not None and capex is not None) else None
        # No shares-outstanding tag ingested -> can't compute a precise
        # per-share dividend yield; PitQualityDefensiveStrategy checks
        # "pays a dividend" directly off dividends_paid_positive instead
        # of a yield threshold (see its _score_stock override).
        dividends_paid_positive = bool(dividends_paid and dividends_paid > 0)

        return StockFundamentals(
            ticker=ticker,
            sector=self._sectors.get(ticker, 'Unknown'),
            industry='Unknown',
            market_cap=0,  # not point-in-time tracked; unused by scoring beyond breakdown
            roe=roe,
            operating_margin=operating_margin,
            gross_margin=gross_margin,
            revenue_growth=yoy_growth('Revenue'),
            earnings_growth=yoy_growth('NetIncome'),
            debt_to_equity=debt_to_equity,
            free_cash_flow=free_cash_flow,
            dividend_yield=(0.02 if dividends_paid_positive else 0.0),  # boolean proxy, see note above
        )


def _fetch_betas_only(strategy, tickers, context):
    """Reimplements just the BETA half of
    HighBetaGrowthStrategy._batch_fetch_betas_and_fundamentals() - not a
    call to the parent method, because that method also fetches
    fundamentals via the live yfinance `.info` path, which would both
    (a) reintroduce the exact lookahead bias/reliability risk this class
    exists to remove, and (b) waste network calls fetching data that's
    about to be overwritten by the point-in-time store anyway. Beta
    itself is unaffected - context.get_historical_prices() is already
    point-in-time correct (only serves data up to the simulated date),
    so reusing HighBetaGrowthStrategy._calculate_beta() directly is fine.
    """
    conn = sqlite3.connect(strategy.db_path)
    cutoff_date = (datetime.now() - pd.Timedelta(days=strategy.BETA_CACHE_DAYS)).strftime("%Y-%m-%d")
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        cached_betas = pd.read_sql_query(
            f"SELECT ticker, beta FROM beta_cache WHERE ticker IN ({placeholders}) AND last_updated >= ?",
            conn, params=(*tickers, cutoff_date),
        )
        for _, row in cached_betas.iterrows():
            if row['beta'] is not None:
                strategy._beta_cache[row['ticker']] = row['beta']

    need_beta = [t for t in tickers if t not in strategy._beta_cache]
    if need_beta:
        beta_results = []
        for ticker in need_beta:
            beta = strategy._calculate_beta(ticker, context)
            if beta is not None:
                strategy._beta_cache[ticker] = beta
                beta_results.append((ticker, beta, datetime.now().strftime("%Y-%m-%d")))
        if beta_results:
            conn.executemany(
                "INSERT OR REPLACE INTO beta_cache (ticker, beta, last_updated) VALUES (?, ?, ?)",
                beta_results,
            )
            conn.commit()
    conn.close()


class PitHighBetaGrowthStrategy(HighBetaGrowthStrategy):
    def __init__(self, pit_store: PitFundamentalsStore, **kwargs):
        super().__init__(**kwargs)
        self._pit_store = pit_store

    def _batch_fetch_betas_and_fundamentals(self, tickers, context):
        _fetch_betas_only(self, tickers, context)
        for ticker in tickers:
            fund = self._pit_store.get_fundamentals_asof(ticker, context.date)
            if fund is not None:
                self._fundamentals_cache[ticker] = fund
            else:
                self._fundamentals_cache.pop(ticker, None)


class PitQualityDefensiveStrategy(QualityDefensiveStrategy):
    def __init__(self, pit_store: PitFundamentalsStore, **kwargs):
        super().__init__(**kwargs)
        self._pit_store = pit_store

    def _batch_fetch_betas_and_fundamentals(self, tickers, context):
        _fetch_betas_only(self, tickers, context)
        for ticker in tickers:
            fund = self._pit_store.get_fundamentals_asof(ticker, context.date)
            if fund is not None:
                self._fundamentals_cache[ticker] = fund
            else:
                self._fundamentals_cache.pop(ticker, None)
