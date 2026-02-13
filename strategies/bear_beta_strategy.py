"""
Bear Beta Strategy
==================

Identifies stocks with low or negative "bear beta" - stocks that move
relatively inversely (or less negatively) to the market during down days.

Bear Beta Definition:
    β_bear = Cov(stock, market | market < -1%) / Var(market | market < -1%)

A stock with β_bear < 0 moves up when market is down (true hedge).
A stock with β_bear near 0 is defensive (doesn't move with market crashes).
A stock with β_bear > 1 amplifies losses during downturns.

This strategy finds stocks that:
1. Have low bear beta (defensive during crashes)
2. Still have positive overall returns (not just shorting the market)
3. Have quality fundamentals (sustainable business)

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
from typing import Optional, Dict, List, Tuple

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


class BearBetaStrategy(Strategy):
    """
    Bear Beta Strategy - finds defensive stocks for downturns.

    Selects stocks with low bear beta (perform well when market crashes)
    combined with quality fundamentals and positive long-term returns.

    Data Source: Yahoo Finance API (unified source for tickers and fundamentals)
    """

    CACHE_DAYS = 30
    FUNDAMENTALS_CACHE_DAYS = 7

    # Defensive sectors that typically have low bear betas
    DEFENSIVE_SECTOR_SCORES = {
        'Consumer Defensive': 15,
        'Consumer Staples': 15,
        'Healthcare': 12,
        'Health Care': 12,
        'Utilities': 12,
        'Communication Services': 8,
        'Real Estate': 5,
    }

    # Large-cap defensive stocks to prioritize
    PRIORITY_DEFENSIVE = {
        # Consumer Staples
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL', 'KMB', 'GIS',
        'K', 'CPB', 'SJM', 'MKC', 'CHD', 'HSY', 'HRL', 'TSN', 'CAG', 'KHC',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
        'GILD', 'MDT', 'ISRG', 'CVS', 'CI', 'HUM', 'MCK', 'CAH', 'VTRS', 'ZTS',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'WEC',
        'ES', 'ED', 'DTE', 'PPL', 'AEE', 'CNP', 'CMS', 'NI', 'EVRG', 'ATO',
        # Low-volatility tech
        'MSFT', 'AAPL', 'CSCO', 'IBM', 'ORCL', 'ACN', 'INTU', 'ADBE',
        # Gold miners (negative bear beta candidates)
        'NEM', 'GOLD', 'AEM', 'FNV', 'WPM', 'RGLD',
    }

    def __init__(
        self,
        db_path: str = "data/fundamentals.sqlite",  # Used only for beta cache
        max_positions: int = 20,
        max_bear_beta: float = 0.8,  # Allow stocks with bear beta < 0.8
        min_total_return: float = -0.10,  # Allow slightly negative returns
        min_score: float = 40,
        rebalance_days: int = 90,
        max_sector_weight: float = 0.40,
        max_position_weight: float = 0.10,
    ):
        self.db_path = db_path
        self.max_positions = max_positions
        self.max_bear_beta = max_bear_beta
        self.min_total_return = min_total_return
        self.min_score = min_score
        self.rebalance_days = rebalance_days
        self.max_sector_weight = max_sector_weight
        self.max_position_weight = max_position_weight

        # Initialize Yahoo data provider (unified data source)
        self._yahoo_provider = YahooDataProvider(cache_db=db_path, cache_days=self.FUNDAMENTALS_CACHE_DAYS)

        self._eligible_tickers: Optional[set] = None
        self._ticker_sectors: Dict[str, str] = {}
        self._bear_beta_cache: Dict[str, dict] = {}
        self._fundamentals_cache: Dict[str, StockFundamentals] = {}
        self._holdings: Dict[str, dict] = {}
        self._last_rebalance: Optional[datetime] = None
        self._init_cache_tables()

    def _init_cache_tables(self) -> None:
        """Initialize cache tables for bear beta (fundamentals cached by YahooDataProvider)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bear_beta_cache (
                ticker TEXT PRIMARY KEY,
                bear_beta REAL,
                bull_beta REAL,
                total_beta REAL,
                down_capture REAL,
                up_capture REAL,
                total_return REAL,
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
        if ticker.endswith('W') or ticker.endswith('WS'):
            return False
        if ticker.endswith('U') and len(ticker) > 2:
            return False
        if ticker.endswith('R') and len(ticker) > 3:
            return False
        if not re.match(r'^[A-Z][A-Z0-9.\-]*$', ticker):
            return False
        return True

    def _load_eligible_tickers(self) -> None:
        """Load tickers from Yahoo Finance data provider (unified source)."""
        # Get defensive universe from Yahoo data provider
        universe = self._yahoo_provider.get_defensive_universe()

        # Filter to valid tickers only
        valid_tickers = [t for t in universe if self._is_valid_ticker(t)]
        self._eligible_tickers = set(valid_tickers)

        print(f"[BearBeta] Loaded {len(self._eligible_tickers)} tickers from Yahoo Finance")

    def _calculate_bear_beta(
        self,
        ticker: str,
        context: ExecutionContext,
        down_threshold: float = -0.01  # Market down > 1%
    ) -> Optional[dict]:
        """
        Calculate bear beta and related metrics.

        β_bear = Cov(stock, market | market < -1%) / Var(market | market < -1%)

        Also calculates:
        - Bull beta (upside capture)
        - Down/Up capture ratios
        - Total return
        - Volatility
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stock_hist = context.get_historical_prices(ticker, 756)  # ~3 years
                spy_hist = context.get_historical_prices("SPY", 756)

            if stock_hist is None or spy_hist is None:
                return None
            if len(stock_hist) < 504 or len(spy_hist) < 504:
                return None

            # Calculate daily returns
            stock_ret = stock_hist['Close'].pct_change().dropna()
            spy_ret = spy_hist['Close'].pct_change().dropna()

            # Align dates
            common_idx = stock_ret.index.intersection(spy_ret.index)
            if len(common_idx) < 400:
                return None

            stock_ret = stock_ret.loc[common_idx]
            spy_ret = spy_ret.loc[common_idx]

            # Identify down days (market < -1%)
            down_days = spy_ret < down_threshold
            up_days = spy_ret > -down_threshold  # > +1%

            # Calculate Bear Beta
            if down_days.sum() < 20:
                return None

            stock_down = stock_ret[down_days].values
            spy_down = spy_ret[down_days].values

            cov_down = np.cov(stock_down, spy_down)[0, 1]
            var_down = np.var(spy_down)

            if var_down > 0:
                bear_beta = cov_down / var_down
            else:
                return None

            # Calculate Bull Beta
            if up_days.sum() >= 20:
                stock_up = stock_ret[up_days].values
                spy_up = spy_ret[up_days].values
                cov_up = np.cov(stock_up, spy_up)[0, 1]
                var_up = np.var(spy_up)
                bull_beta = cov_up / var_up if var_up > 0 else None
            else:
                bull_beta = None

            # Calculate total (normal) beta
            cov_total = np.cov(stock_ret.values, spy_ret.values)[0, 1]
            var_total = np.var(spy_ret.values)
            total_beta = cov_total / var_total if var_total > 0 else None

            # Down capture ratio
            avg_stock_down = stock_ret[down_days].mean()
            avg_spy_down = spy_ret[down_days].mean()
            down_capture = avg_stock_down / avg_spy_down if avg_spy_down != 0 else None

            # Up capture ratio
            if up_days.sum() >= 20:
                avg_stock_up = stock_ret[up_days].mean()
                avg_spy_up = spy_ret[up_days].mean()
                up_capture = avg_stock_up / avg_spy_up if avg_spy_up != 0 else None
            else:
                up_capture = None

            # Total return (annualized)
            total_return = (stock_hist['Close'].iloc[-1] / stock_hist['Close'].iloc[0]) ** (252 / len(stock_hist)) - 1

            # Volatility (annualized)
            volatility = stock_ret.std() * np.sqrt(252)

            return {
                'ticker': ticker,
                'bear_beta': bear_beta,
                'bull_beta': bull_beta,
                'total_beta': total_beta,
                'down_capture': down_capture,
                'up_capture': up_capture,
                'total_return': total_return,
                'volatility': volatility,
                'down_days_count': down_days.sum(),
            }
        except Exception as e:
            return None


    def _batch_calculate_bear_betas(
        self,
        tickers: List[str],
        context: ExecutionContext
    ) -> None:
        """Calculate bear betas with caching."""
        conn = sqlite3.connect(self.db_path)
        cutoff_date = (datetime.now() - timedelta(days=self.CACHE_DAYS)).strftime("%Y-%m-%d")

        # Load cached bear betas
        if tickers:
            placeholders = ",".join("?" * len(tickers))
            cached = pd.read_sql_query(
                f"""
                SELECT ticker, bear_beta, bull_beta, total_beta,
                       down_capture, up_capture, total_return, volatility
                FROM bear_beta_cache
                WHERE ticker IN ({placeholders}) AND last_updated >= ?
                """,
                conn,
                params=(*tickers, cutoff_date),
            )
            for _, row in cached.iterrows():
                self._bear_beta_cache[row['ticker']] = row.to_dict()

        # Determine what needs calculation
        need_calc = [t for t in tickers if t not in self._bear_beta_cache]

        if need_calc:
            print(f"[BearBeta] Calculating bear betas for {len(need_calc)} tickers...")
            results = []

            for i, ticker in enumerate(need_calc):
                if i % 50 == 0:
                    print(f"  Progress: {i}/{len(need_calc)}")

                result = self._calculate_bear_beta(ticker, context)
                if result is not None:
                    self._bear_beta_cache[ticker] = result
                    results.append((
                        ticker,
                        result['bear_beta'],
                        result.get('bull_beta'),
                        result.get('total_beta'),
                        result.get('down_capture'),
                        result.get('up_capture'),
                        result.get('total_return'),
                        result.get('volatility'),
                        datetime.now().strftime("%Y-%m-%d"),
                    ))

            if results:
                conn.executemany(
                    """INSERT OR REPLACE INTO bear_beta_cache
                       (ticker, bear_beta, bull_beta, total_beta,
                        down_capture, up_capture, total_return, volatility, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    results,
                )
                conn.commit()

        conn.close()

    def _batch_fetch_fundamentals(self, tickers: List[str]) -> None:
        """Fetch fundamentals using Yahoo Finance (unified source)."""
        need_funds = [t for t in tickers if t not in self._fundamentals_cache]

        if need_funds:
            print(f"[BearBeta] Fetching fundamentals for {len(need_funds)} tickers...")
            fundamentals = self._yahoo_provider.get_fundamentals_batch(need_funds)
            self._fundamentals_cache.update(fundamentals)

    def _score_stock(self, ticker: str) -> Optional[dict]:
        """
        Score a stock for defensive characteristics.

        Higher score = better defensive stock.
        Uses StockFundamentals from Yahoo Finance (unified data source).
        """
        beta_data = self._bear_beta_cache.get(ticker)
        fund = self._fundamentals_cache.get(ticker)

        if beta_data is None:
            return None

        bear_beta = beta_data.get('bear_beta')
        if bear_beta is None:
            return None

        # Filter by max bear beta
        if bear_beta > self.max_bear_beta:
            return None

        # Filter by total return requirement
        total_return = beta_data.get('total_return', 0) or 0
        if total_return < self.min_total_return:
            return None

        score = 0
        breakdown = {}

        # BEAR BETA SCORE (max 35 points) - lower is better
        if bear_beta < 0:
            # Negative bear beta - moves opposite to market on down days (excellent!)
            bb_score = 35
        elif bear_beta < 0.3:
            bb_score = 30
        elif bear_beta < 0.5:
            bb_score = 25
        elif bear_beta < 0.7:
            bb_score = 15
        else:
            bb_score = 5

        score += bb_score
        breakdown['bear_beta'] = bb_score

        # DOWN CAPTURE SCORE (max 20 points) - lower capture is better
        down_capture = beta_data.get('down_capture')
        if down_capture is not None:
            if down_capture < 0.5:
                dc_score = 20  # Loses less than half of market decline
            elif down_capture < 0.7:
                dc_score = 15
            elif down_capture < 0.9:
                dc_score = 10
            else:
                dc_score = 5
            score += dc_score
            breakdown['down_capture'] = dc_score

        # ASYMMETRY SCORE (max 15 points) - high up_capture / low down_capture
        up_capture = beta_data.get('up_capture')
        if up_capture and down_capture and down_capture > 0:
            asymmetry = up_capture / down_capture
            if asymmetry > 1.5:
                asym_score = 15
            elif asymmetry > 1.2:
                asym_score = 10
            elif asymmetry > 1.0:
                asym_score = 5
            else:
                asym_score = 0
            score += asym_score
            breakdown['asymmetry'] = asym_score

        # RETURN SCORE (max 10 points) - positive returns still matter
        if total_return > 0.15:
            ret_score = 10
        elif total_return > 0.10:
            ret_score = 8
        elif total_return > 0.05:
            ret_score = 5
        elif total_return > 0:
            ret_score = 3
        else:
            ret_score = 0

        score += ret_score
        breakdown['return'] = ret_score

        # SECTOR SCORE (max 15 points)
        sector = 'Unknown'
        if fund:
            sector = fund.sector or 'Unknown'
        sector_score = self.DEFENSIVE_SECTOR_SCORES.get(sector, 0)
        score += sector_score
        breakdown['sector'] = sector_score

        # QUALITY SCORE (max 10 points) - fundamentals if available
        quality_score = 0
        if fund:
            div_yield = fund.dividend_yield or 0
            if div_yield > 0.02:
                quality_score += 3
            if div_yield > 0.03:
                quality_score += 2

            profit_margin = fund.profit_margin or 0
            if profit_margin > 0.10:
                quality_score += 3

            current_ratio = fund.current_ratio or 0
            if current_ratio > 1.5:
                quality_score += 2

        score += quality_score
        breakdown['quality'] = quality_score

        # MARKET CAP BONUS (max 10 points) - prefer large-cap for stability
        market_cap = (fund.market_cap or 0) if fund else 0
        if market_cap > 100_000_000_000:  # > $100B
            cap_score = 10
        elif market_cap > 50_000_000_000:  # > $50B
            cap_score = 8
        elif market_cap > 10_000_000_000:  # > $10B
            cap_score = 5
        elif market_cap > 1_000_000_000:  # > $1B
            cap_score = 2
        else:
            cap_score = 0

        score += cap_score
        breakdown['market_cap'] = cap_score

        # PRIORITY STOCK BONUS (10 points) - known defensive stocks
        if ticker in self.PRIORITY_DEFENSIVE:
            score += 10
            breakdown['priority'] = 10

        return {
            'ticker': ticker,
            'score': score,
            'breakdown': breakdown,
            'bear_beta': bear_beta,
            'bull_beta': beta_data.get('bull_beta'),
            'total_beta': beta_data.get('total_beta'),
            'down_capture': beta_data.get('down_capture'),
            'up_capture': beta_data.get('up_capture'),
            'total_return': total_return,
            'volatility': beta_data.get('volatility'),
            'sector': sector,
            'market_cap': market_cap,
        }

    def execute(self, context: ExecutionContext) -> Portfolio:
        """Execute the bear beta strategy."""
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
            print(f"[BearBeta] Rebalancing on {context.date.strftime('%Y-%m-%d')}...")

            # ALWAYS include priority defensive stocks regardless of database
            # These are large-cap defensive stocks we know are good candidates
            candidates = list(self.PRIORITY_DEFENSIVE)

            # Add some from database if available
            if self._eligible_tickers:
                other_tickers = [t for t in self._eligible_tickers
                                if t not in self.PRIORITY_DEFENSIVE]
                remaining_slots = 200 - len(candidates)
                if remaining_slots > 0:
                    candidates.extend(other_tickers[:remaining_slots])

            print(f"[BearBeta] Evaluating {len(candidates)} candidates ({len(self.PRIORITY_DEFENSIVE)} priority defensive)")

            # Calculate bear betas
            self._batch_calculate_bear_betas(candidates, context)
            self._batch_fetch_fundamentals(candidates)

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

            print(f"[BearBeta] Found {len(scored)} defensive stocks (score >= {self.min_score})")

            # Log top candidates with bear beta info
            print("\n[BearBeta] Top defensive candidates:")
            for s in scored[:10]:
                print(f"  {s['ticker']:6s} Score={s['score']:2.0f}  "
                      f"BearBeta={s['bear_beta']:5.2f}  "
                      f"Down%={s.get('down_capture', 0)*100:4.0f}%  "
                      f"Return={s['total_return']*100:5.1f}%  "
                      f"Sector={s['sector']}")

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
                        'bear_beta': candidate['bear_beta'],
                        'sector': sector,
                    }
                    sector_weights[sector] += weight

            self._last_rebalance = context.date

            if self._holdings:
                print(f"\n[BearBeta] Selected {len(self._holdings)} defensive positions")

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


def analyze_bear_betas(
    tickers: List[str] = None,
    lookback_days: int = 756,
    down_threshold: float = -0.01
) -> pd.DataFrame:
    """
    Analyze bear betas for a list of tickers.

    Standalone function for research/analysis.
    Uses Yahoo Finance directly for this analysis utility.
    """
    import yfinance as yf  # Local import for standalone analysis function

    if tickers is None:
        # Default to S&P 500 sectors ETFs and some individual stocks
        tickers = [
            'SPY', 'QQQ', 'IWM',  # Benchmarks
            'XLK', 'XLV', 'XLP', 'XLU', 'XLE', 'XLF', 'XLI', 'XLY', 'XLB',  # Sectors
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN',  # Tech
            'JNJ', 'PG', 'KO', 'PEP', 'WMT',  # Defensive
            'NEM', 'GOLD', 'GLD', 'TLT', 'SHY',  # Safe havens
        ]

    print(f"Fetching {lookback_days} days of data for {len(tickers)} tickers...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    # Fetch SPY for market returns
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)['Close']
    if hasattr(spy, 'squeeze'):
        spy = spy.squeeze()
    spy_ret = spy.pct_change().dropna()

    results = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
            if hasattr(data, 'squeeze'):
                data = data.squeeze()
            stock_ret = data.pct_change().dropna()

            # Align
            common_idx = stock_ret.index.intersection(spy_ret.index)
            if len(common_idx) < 200:
                continue

            stock = stock_ret.loc[common_idx]
            market = spy_ret.loc[common_idx]

            # Down days
            down_days = market < down_threshold
            up_days = market > -down_threshold

            if down_days.sum() < 20:
                continue

            # Bear beta
            stock_down = stock[down_days].values
            mkt_down = market[down_days].values
            bear_beta = np.cov(stock_down, mkt_down)[0, 1] / np.var(mkt_down)

            # Bull beta
            if up_days.sum() >= 20:
                stock_up = stock[up_days].values
                mkt_up = market[up_days].values
                bull_beta = np.cov(stock_up, mkt_up)[0, 1] / np.var(mkt_up)
            else:
                bull_beta = None

            # Total beta
            total_beta = np.cov(stock.values, market.values)[0, 1] / np.var(market.values)

            # Capture ratios
            down_capture = stock[down_days].mean() / market[down_days].mean()
            up_capture = stock[up_days].mean() / market[up_days].mean() if up_days.sum() >= 20 else None

            # Total return
            total_return = (data.iloc[-1] / data.iloc[0]) - 1

            results.append({
                'ticker': ticker,
                'bear_beta': bear_beta,
                'bull_beta': bull_beta,
                'total_beta': total_beta,
                'down_capture': down_capture,
                'up_capture': up_capture,
                'total_return': total_return,
                'volatility': stock.std() * np.sqrt(252),
                'down_days': down_days.sum(),
            })
        except Exception as e:
            print(f"  Error for {ticker}: {e}")

    df = pd.DataFrame(results)
    df = df.sort_values('bear_beta')

    return df


if __name__ == "__main__":
    print("=" * 70)
    print("BEAR BETA ANALYSIS")
    print("=" * 70)
    print("\nβ_bear = Cov(stock, market | market < -1%) / Var(market | market < -1%)")
    print("\nLower bear beta = better performance during market crashes\n")

    # Analyze some tickers
    df = analyze_bear_betas()

    print("\nBear Beta Rankings (lowest = most defensive):")
    print("-" * 70)
    print(df.to_string(index=False))

    print("\n\nInterpretation:")
    print("- β_bear < 0: Moves UP when market crashes (true hedge)")
    print("- β_bear ≈ 0: Doesn't move with market crashes (defensive)")
    print("- β_bear < 1: Loses less than market during crashes")
    print("- β_bear > 1: Amplifies losses during crashes (avoid for defense)")
