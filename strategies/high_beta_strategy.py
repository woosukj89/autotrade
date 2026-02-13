"""
High Beta Growth Strategy
=========================

Based on analysis of S&P 500 outperformers (2014-2026), this strategy identifies
stocks with characteristics that historically led to market-beating returns.

Key findings from analysis:
- Outperformers had avg Beta of 1.26 vs 0.83 for underperformers
- 92% of top performers had Beta > 1.0
- Technology sector represented 64% of outperformers
- High profitability (ROE > 25%) combined with high growth (>15%)

Quantifiable selection criteria:
1. Beta > 1.0 (captures market upside)
2. ROE > 15% (prefer > 25%)
3. Operating Margin > 15%
4. Gross Margin > 40%
5. Revenue Growth > 10%
6. Sector: Technology, Health Care, Consumer Discretionary preferred

Data Source:
- All ticker and fundamental data from Yahoo Finance API for consistency
"""

import os
import sys
import sqlite3
import logging
import re
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

# Handle imports for both package and direct execution
try:
    from strategies.strategy import Strategy, Portfolio, Position, ExecutionContext
except ImportError:
    from strategy import Strategy, Portfolio, Position, ExecutionContext

# Import Yahoo data provider
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from data.yahoo_data import YahooDataProvider, StockFundamentals
except ImportError:
    from yahoo_data import YahooDataProvider, StockFundamentals

# Suppress yfinance download warnings
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*YFPricesMissingError.*')
warnings.filterwarnings('ignore', message='.*possibly delisted.*')


class HighBetaGrowthStrategy(Strategy):
    """
    High Beta Growth Strategy targeting market outperformers.

    Based on 12-year analysis showing that high-beta, high-growth,
    high-profitability stocks significantly outperformed the market.

    Data Source: Yahoo Finance API (unified source for tickers and fundamentals)
    """

    BETA_CACHE_DAYS = 30
    FUNDAMENTALS_CACHE_DAYS = 7

    # Sector scores based on historical outperformance
    SECTOR_SCORES = {
        'Technology': 15,
        'Health Care': 10,
        'Consumer Discretionary': 10,
        'Consumer Cyclical': 10,
        'Financials': 5,
        'Financial Services': 5,
        'Industrials': 5,
        'Communication Services': 5,
    }

    # Priority tickers - large cap stocks that should always be considered
    # Based on S&P 500 top holdings and historical outperformers
    PRIORITY_TICKERS = {
        # Top S&P 500 by weight
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'BRK.B', 'LLY', 'AVGO',
        'JPM', 'TSLA', 'UNH', 'XOM', 'V', 'MA', 'COST', 'HD', 'PG', 'JNJ',
        'WMT', 'NFLX', 'CRM', 'BAC', 'ABBV', 'CVX', 'MRK', 'KO', 'PEP', 'AMD',
        'TMO', 'ORCL', 'CSCO', 'ACN', 'MCD', 'LIN', 'ABT', 'ADBE', 'DHR', 'WFC',
        # Historical outperformers from analysis
        'LRCX', 'ANET', 'KLAC', 'CDNS', 'FTNT', 'MU', 'PANW', 'MSCI', 'CTAS',
        'AMAT', 'SNPS', 'MRVL', 'CRWD', 'NOW', 'INTU', 'ISRG', 'REGN', 'VRTX',
    }

    def __init__(
        self,
        db_path: str = "data/fundamentals.sqlite",  # Used only for beta cache
        max_positions: int = 15,
        min_beta: float = 1.0,
        min_beta_quality: float = 0.7,  # Lower beta allowed for high-quality stocks
        min_quality_score: int = 45,    # Profitability + Growth score required for low-beta
        min_score: float = 50,
        rebalance_days: int = 90,
        max_sector_weight: float = 0.50,  # Allow tech concentration
        max_position_weight: float = 0.15,
    ):
        self.db_path = db_path
        self.max_positions = max_positions
        self.min_beta = min_beta
        self.min_beta_quality = min_beta_quality
        self.min_quality_score = min_quality_score
        self.min_score = min_score
        self.rebalance_days = rebalance_days
        self.max_sector_weight = max_sector_weight
        self.max_position_weight = max_position_weight

        # Initialize Yahoo data provider (unified data source)
        self._yahoo_provider = YahooDataProvider(cache_db=db_path, cache_days=self.FUNDAMENTALS_CACHE_DAYS)

        self._eligible_tickers: Optional[set] = None
        self._ticker_sectors: Dict[str, str] = {}
        self._beta_cache: Dict[str, float] = {}
        self._fundamentals_cache: Dict[str, StockFundamentals] = {}
        self._holdings: Dict[str, dict] = {}
        self._last_rebalance: Optional[datetime] = None
        self._init_cache_tables()

    def _init_cache_tables(self) -> None:
        """Initialize cache tables for beta (fundamentals cached by YahooDataProvider)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS beta_cache (
                ticker TEXT PRIMARY KEY,
                beta REAL,
                volatility REAL,
                last_updated TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _is_valid_ticker(self, ticker: str) -> bool:
        """Filter out warrants, units, and other special securities."""
        if not ticker or len(ticker) > 6:
            return False
        # Exclude warrants (ending in W), units (ending in U), rights (ending in R)
        if ticker.endswith('W') or ticker.endswith('WS'):
            return False
        if ticker.endswith('U') and len(ticker) > 2:
            return False
        if ticker.endswith('R') and len(ticker) > 3:
            return False
        # Allow tickers with letters, periods, and hyphens (e.g., BRK.B, BRK-B)
        if not re.match(r'^[A-Z][A-Z0-9.\-]*$', ticker):
            return False
        return True

    def _load_eligible_tickers(self) -> None:
        """Load tickers from Yahoo Finance data provider (unified source)."""
        # Get universe from Yahoo data provider
        universe = self._yahoo_provider.get_high_beta_universe()

        # Filter to valid tickers only
        valid_tickers = [t for t in universe if self._is_valid_ticker(t)]
        self._eligible_tickers = set(valid_tickers)

        print(f"[HighBetaGrowth] Loaded {len(self._eligible_tickers)} tickers from Yahoo Finance")

    def _calculate_beta(self, ticker: str, context: ExecutionContext) -> Optional[float]:
        """Calculate beta from historical prices."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Get stock and SPY historical data
                stock_hist = context.get_historical_prices(ticker, 504)  # ~2 years
                spy_hist = context.get_historical_prices("SPY", 504)

            if stock_hist is None or spy_hist is None:
                return None
            if len(stock_hist) < 252 or len(spy_hist) < 252:
                return None

            # Calculate returns
            stock_ret = stock_hist['Close'].pct_change().dropna()
            spy_ret = spy_hist['Close'].pct_change().dropna()

            # Align dates
            common_idx = stock_ret.index.intersection(spy_ret.index)
            if len(common_idx) < 200:
                return None

            stock_ret = stock_ret.loc[common_idx].values
            spy_ret = spy_ret.loc[common_idx].values

            # Calculate beta
            covariance = np.cov(stock_ret, spy_ret)[0, 1]
            variance = np.var(spy_ret)

            if variance > 0:
                return covariance / variance
            return None
        except Exception:
            return None

    def _batch_fetch_betas_and_fundamentals(
        self,
        tickers: List[str],
        context: ExecutionContext
    ) -> None:
        """Fetch betas and fundamentals using Yahoo Finance (unified source)."""
        conn = sqlite3.connect(self.db_path)
        cutoff_date = (datetime.now() - timedelta(days=self.BETA_CACHE_DAYS)).strftime("%Y-%m-%d")

        # Load cached betas from local cache
        if tickers:
            placeholders = ",".join("?" * len(tickers))
            cached_betas = pd.read_sql_query(
                f"""
                SELECT ticker, beta, volatility
                FROM beta_cache
                WHERE ticker IN ({placeholders}) AND last_updated >= ?
                """,
                conn,
                params=(*tickers, cutoff_date),
            )
            for _, row in cached_betas.iterrows():
                self._beta_cache[row['ticker']] = row['beta']

        # Determine what needs fetching
        need_beta = [t for t in tickers if t not in self._beta_cache]
        need_funds = [t for t in tickers if t not in self._fundamentals_cache]

        # Fetch betas (calculated from price history)
        if need_beta:
            print(f"[HighBetaGrowth] Calculating beta for {len(need_beta)} tickers...")
            beta_results = []
            for ticker in need_beta:
                beta = self._calculate_beta(ticker, context)
                if beta is not None:
                    self._beta_cache[ticker] = beta
                    beta_results.append((ticker, beta, datetime.now().strftime("%Y-%m-%d")))

            if beta_results:
                conn.executemany(
                    "INSERT OR REPLACE INTO beta_cache (ticker, beta, last_updated) VALUES (?, ?, ?)",
                    beta_results,
                )
                conn.commit()

        conn.close()

        # Fetch fundamentals using Yahoo provider (handles its own caching)
        if need_funds:
            print(f"[HighBetaGrowth] Fetching fundamentals for {len(need_funds)} tickers...")
            fundamentals = self._yahoo_provider.get_fundamentals_batch(need_funds)
            self._fundamentals_cache.update(fundamentals)

    def _score_stock(self, ticker: str) -> Optional[dict]:
        """
        Score a stock based on outperformer characteristics.
        Returns score (0-100) and component breakdown.

        Uses StockFundamentals from Yahoo Finance (unified data source).
        """
        beta = self._beta_cache.get(ticker)
        fund = self._fundamentals_cache.get(ticker)

        if beta is None or fund is None:
            return None

        score = 0
        breakdown = {}

        # Extract values from StockFundamentals object
        roe = fund.roe or 0
        op_margin = fund.operating_margin or 0
        gross_margin = fund.gross_margin or 0
        rev_growth = fund.revenue_growth or 0
        earn_growth = fund.earnings_growth or 0

        # PROFITABILITY (max 30 points)
        prof_score = 0
        if roe > 0.15:
            prof_score += 10
        if roe > 0.25:
            prof_score += 5
        if op_margin > 0.15:
            prof_score += 10
        if gross_margin > 0.40:
            prof_score += 5

        # GROWTH (max 25 points)
        growth_score = 0
        if rev_growth > 0.10:
            growth_score += 10
        if rev_growth > 0.20:
            growth_score += 5
        if earn_growth > 0.15:
            growth_score += 10

        quality_score = prof_score + growth_score

        # Beta requirement: normal min_beta OR lower min_beta_quality if high quality
        if beta < self.min_beta:
            # Allow low-beta stocks only if they have exceptional quality
            if beta < self.min_beta_quality or quality_score < self.min_quality_score:
                return None
            # Low beta quality stock - no beta points awarded
            beta_score = 0
        else:
            # Normal high-beta stock
            beta_score = 10 if beta > 1.0 else 0
            if beta > 1.2:
                beta_score += 5

        # Add profitability and growth to score
        score += prof_score
        breakdown['profitability'] = prof_score
        score += growth_score
        breakdown['growth'] = growth_score

        # BALANCE SHEET (max 15 points)
        debt_equity = fund.debt_to_equity or 0
        fcf = fund.free_cash_flow or 0

        balance_score = 0
        if debt_equity < 100:
            balance_score += 5
        if debt_equity < 50:
            balance_score += 5
        if fcf > 0:
            balance_score += 5

        score += balance_score
        breakdown['balance_sheet'] = balance_score

        # MARKET EXPOSURE (max 15 points) - already calculated above
        score += beta_score
        breakdown['beta'] = beta_score

        # SECTOR (max 15 points)
        sector = fund.sector or 'Unknown'
        sector_score = self.SECTOR_SCORES.get(sector, 0)

        score += sector_score
        breakdown['sector'] = sector_score

        return {
            'ticker': ticker,
            'score': score,
            'breakdown': breakdown,
            'beta': beta,
            'roe': roe,
            'operating_margin': op_margin,
            'gross_margin': gross_margin,
            'revenue_growth': rev_growth,
            'earnings_growth': earn_growth,
            'debt_to_equity': debt_equity,
            'sector': sector,
            'market_cap': fund.market_cap or 0,
        }

    def execute(self, context: ExecutionContext) -> Portfolio:
        """Execute the high beta growth strategy."""
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        # Check if rebalance needed
        needs_rebalance = (
            self._last_rebalance is None
            or (context.date - self._last_rebalance).days >= self.rebalance_days
        )

        if needs_rebalance:
            print(f"[HighBetaGrowth] Rebalancing on {context.date.strftime('%Y-%m-%d')}...")

            # Build candidate list: priority tickers first, then others sorted by market cap
            priority_in_db = [t for t in self.PRIORITY_TICKERS if t in self._eligible_tickers]
            other_tickers = [t for t in self._eligible_tickers if t not in self.PRIORITY_TICKERS]

            # Start with priority tickers
            candidates = list(priority_in_db)

            # Add other tickers up to limit
            remaining_slots = 500 - len(candidates)
            if remaining_slots > 0:
                candidates.extend(other_tickers[:remaining_slots])

            print(f"[HighBetaGrowth] Evaluating {len(candidates)} candidates ({len(priority_in_db)} priority)")

            # Fetch betas and fundamentals
            self._batch_fetch_betas_and_fundamentals(candidates, context)

            # Score all candidates
            scored = []
            for ticker in candidates:
                result = self._score_stock(ticker)
                if result and result['score'] >= self.min_score:
                    price = context.get_price(ticker)
                    if price and price > 0:
                        result['price'] = price
                        scored.append(result)

            # Sort by score (highest first)
            scored.sort(key=lambda x: x['score'], reverse=True)

            print(f"[HighBetaGrowth] Found {len(scored)} stocks meeting criteria (score >= {self.min_score})")

            # Select top candidates with sector diversification
            self._holdings = {}
            sector_weights = defaultdict(float)

            for candidate in scored:
                if len(self._holdings) >= self.max_positions:
                    break

                ticker = candidate['ticker']
                sector = candidate['sector']

                # Check sector weight limit
                if sector_weights[sector] >= self.max_sector_weight:
                    continue

                # Calculate position weight (higher score = higher weight)
                base_weight = 0.90 / self.max_positions
                score_bonus = (candidate['score'] - self.min_score) / 50 * 0.05  # Up to 5% bonus
                weight = min(base_weight + score_bonus, self.max_position_weight)

                alloc = weight * total_value
                shares = int(alloc / candidate['price'])

                if shares > 0:
                    self._holdings[ticker] = {
                        'shares': shares,
                        'entry_price': candidate['price'],
                        'score': candidate['score'],
                        'beta': candidate['beta'],
                        'sector': sector,
                    }
                    sector_weights[sector] += weight

            self._last_rebalance = context.date

            # Log selected stocks
            if self._holdings:
                print(f"[HighBetaGrowth] Selected {len(self._holdings)} positions:")
                for ticker, data in list(self._holdings.items())[:10]:
                    print(f"  {ticker}: Score={data['score']}, Beta={data['beta']:.2f}, Sector={data['sector']}")

        # Build portfolio
        positions = {}
        invested = 0.0
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if price and holding['shares'] > 0:
                positions[ticker] = Position(
                    ticker=ticker,
                    shares=float(holding['shares']),
                    avg_cost=holding['entry_price'],
                )
                invested += holding['shares'] * price

        return Portfolio(cash=total_value - invested, positions=positions)


class HighBetaMomentumStrategy(HighBetaGrowthStrategy):
    """
    High Beta strategy with momentum overlay.

    Adds price momentum to the selection criteria:
    - 12-month price momentum > 0
    - Stock above 200-day moving average
    """

    def __init__(
        self,
        db_path: str = "data/fundamentals.sqlite",
        max_positions: int = 12,
        min_beta: float = 1.0,
        min_score: float = 45,
        rebalance_days: int = 60,
        **kwargs
    ):
        super().__init__(
            db_path=db_path,
            max_positions=max_positions,
            min_beta=min_beta,
            min_score=min_score,
            rebalance_days=rebalance_days,
            **kwargs
        )

    def _score_stock(self, ticker: str, context: ExecutionContext = None) -> Optional[dict]:
        """Score with momentum overlay."""
        result = super()._score_stock(ticker)
        if result is None:
            return None

        # Add momentum score if context available
        if context:
            try:
                hist = context.get_historical_prices(ticker, 252)
                if hist is not None and len(hist) >= 200:
                    current = float(hist['Close'].iloc[-1])
                    year_ago = float(hist['Close'].iloc[0])
                    ma_200 = float(hist['Close'].tail(200).mean())

                    # 12-month momentum
                    momentum = (current - year_ago) / year_ago

                    # Momentum score (max 10 points bonus)
                    mom_score = 0
                    if momentum > 0:
                        mom_score += 5
                    if momentum > 0.20:
                        mom_score += 3
                    if current > ma_200:
                        mom_score += 2

                    result['score'] += mom_score
                    result['breakdown']['momentum'] = mom_score
                    result['momentum'] = momentum
            except:
                pass

        return result

    def execute(self, context: ExecutionContext) -> Portfolio:
        """Execute with momentum-aware scoring."""
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        needs_rebalance = (
            self._last_rebalance is None
            or (context.date - self._last_rebalance).days >= self.rebalance_days
        )

        if needs_rebalance:
            print(f"[HighBetaMomentum] Rebalancing on {context.date.strftime('%Y-%m-%d')}...")

            # Build candidate list: priority tickers first, then others
            priority_in_db = [t for t in self.PRIORITY_TICKERS if t in self._eligible_tickers]
            other_tickers = [t for t in self._eligible_tickers if t not in self.PRIORITY_TICKERS]
            candidates = list(priority_in_db)
            remaining_slots = 500 - len(candidates)
            if remaining_slots > 0:
                candidates.extend(other_tickers[:remaining_slots])

            print(f"[HighBetaMomentum] Evaluating {len(candidates)} candidates ({len(priority_in_db)} priority)")
            self._batch_fetch_betas_and_fundamentals(candidates, context)

            # Score with momentum
            scored = []
            for ticker in candidates:
                result = self._score_stock(ticker, context)
                if result and result['score'] >= self.min_score:
                    price = context.get_price(ticker)
                    if price and price > 0:
                        result['price'] = price
                        scored.append(result)

            scored.sort(key=lambda x: x['score'], reverse=True)

            print(f"[HighBetaMomentum] Found {len(scored)} stocks meeting criteria")

            # Select positions
            self._holdings = {}
            sector_weights = defaultdict(float)

            for candidate in scored:
                if len(self._holdings) >= self.max_positions:
                    break

                ticker = candidate['ticker']
                sector = candidate['sector']

                if sector_weights[sector] >= self.max_sector_weight:
                    continue

                weight = min(0.90 / self.max_positions, self.max_position_weight)
                alloc = weight * total_value
                shares = int(alloc / candidate['price'])

                if shares > 0:
                    self._holdings[ticker] = {
                        'shares': shares,
                        'entry_price': candidate['price'],
                        'score': candidate['score'],
                        'beta': candidate['beta'],
                        'sector': sector,
                    }
                    sector_weights[sector] += weight

            self._last_rebalance = context.date

        # Build portfolio
        positions = {}
        invested = 0.0
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if price and holding['shares'] > 0:
                positions[ticker] = Position(
                    ticker=ticker,
                    shares=float(holding['shares']),
                    avg_cost=holding['entry_price'],
                )
                invested += holding['shares'] * price

        return Portfolio(cash=total_value - invested, positions=positions)


def get_high_beta_strategies(db_path: str = "data/fundamentals.sqlite") -> dict:
    """Return all high beta strategies for testing."""
    return {
        "HighBetaGrowth": HighBetaGrowthStrategy(db_path),
        "HighBetaMomentum": HighBetaMomentumStrategy(db_path),
    }
