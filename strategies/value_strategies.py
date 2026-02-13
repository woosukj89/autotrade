"""
Multiple value investing strategy variations for backtesting.

Strategies:
1. BuyAndHoldValueStrategy - True Buffett style: buy undervalued, hold for years
2. MomentumValueStrategy - Value + price momentum to time entries
3. ConcentratedValueStrategy - Fewer positions, higher conviction
4. QualityGrowthStrategy - Emphasize moat and growth over deep discount
5. AdaptiveValueStrategy - Dynamic allocation based on opportunity quality
"""

import sqlite3
import concurrent.futures
from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from strategies.strategy import Strategy, Portfolio, Position, ExecutionContext


class BaseValueStrategy(Strategy):
    """Base class with shared valuation logic."""

    DISCOUNT_RATE = 0.09
    TERMINAL_GROWTH = 0.02
    TAX_RETENTION = 0.79
    SHARES_CACHE_DAYS = 30  # Refresh shares outstanding every 30 days

    def __init__(self, db_path: str = "fundamentals.sqlite"):
        self.db_path = db_path
        self._eligible_tickers: Optional[set[str]] = None
        self._shares_outstanding: dict[str, Optional[float]] = {}
        self._valuations: dict[str, dict] = {}
        self._last_valuation_date: Optional[datetime] = None
        self._init_shares_cache()

    def _init_shares_cache(self) -> None:
        """Initialize the shares_outstanding cache table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shares_cache (
                ticker TEXT PRIMARY KEY,
                shares_outstanding REAL,
                last_updated TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_eligible_tickers(self) -> None:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            """
            SELECT ticker, COUNT(DISTINCT fy) AS years
            FROM fundamentals
            WHERE statement_type = 'income'
            GROUP BY ticker
            HAVING years >= 7
            """,
            conn,
        )
        conn.close()
        self._eligible_tickers = set(df["ticker"])

    def _batch_fetch_shares_outstanding(self, tickers: list[str]) -> None:
        """Fetch shares outstanding with SQLite caching.

        Only fetches from yfinance if:
        1. Ticker not in cache, OR
        2. Cache entry is older than SHARES_CACHE_DAYS
        """
        to_fetch = [t for t in tickers if t not in self._shares_outstanding]
        if not to_fetch:
            return

        conn = sqlite3.connect(self.db_path)
        cutoff_date = (datetime.now() - pd.Timedelta(days=self.SHARES_CACHE_DAYS)).strftime("%Y-%m-%d")

        # Load from cache
        placeholders = ",".join("?" * len(to_fetch))
        cached = pd.read_sql_query(
            f"""
            SELECT ticker, shares_outstanding, last_updated
            FROM shares_cache
            WHERE ticker IN ({placeholders}) AND last_updated >= ?
            """,
            conn,
            params=(*to_fetch, cutoff_date),
        )

        # Add cached values to memory
        cached_tickers = set()
        for _, row in cached.iterrows():
            ticker = row["ticker"]
            shares = row["shares_outstanding"]
            self._shares_outstanding[ticker] = shares if shares and shares > 0 else None
            cached_tickers.add(ticker)

        # Determine which tickers need fresh fetch
        need_fetch = [t for t in to_fetch if t not in cached_tickers]

        if need_fetch:
            print(f"Fetching shares outstanding for {len(need_fetch)} tickers (cached: {len(cached_tickers)})...")

            def _fetch_one(ticker: str):
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        info = yf.Ticker(ticker).info
                        if info is None:
                            return ticker, None
                        shares = info.get("sharesOutstanding")
                        return ticker, shares if shares and shares > 0 else None
                except Exception:
                    return ticker, None

            # Fetch in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
                for ticker, shares in pool.map(_fetch_one, need_fetch):
                    self._shares_outstanding[ticker] = shares
                    results.append((ticker, shares, datetime.now().strftime("%Y-%m-%d")))

            # Save to cache
            if results:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO shares_cache (ticker, shares_outstanding, last_updated)
                    VALUES (?, ?, ?)
                    """,
                    results,
                )
                conn.commit()

        conn.close()

    def _compute_all_valuations(self, as_of_date: datetime) -> None:
        """Compute valuations with Revenue as OPTIONAL.

        CHANGED: Revenue is now optional. When missing:
        - Use GrossProfit as revenue proxy for margin calculations
        - Use ROE (NetIncome/Equity) as ROIC proxy
        - This allows companies like Apple (missing Revenue 2012-2017) to be valued.
        """
        conn = sqlite3.connect(self.db_path)
        date_str = as_of_date.strftime("%Y-%m-%d")

        df = pd.read_sql_query(
            """
            SELECT ticker, fy, field, value
            FROM fundamentals
            WHERE date <= ?
            ORDER BY ticker, fy
            """,
            conn,
            params=(date_str,),
        )
        conn.close()

        if df.empty:
            self._valuations = {}
            self._last_valuation_date = as_of_date
            return

        df = df[df["ticker"].isin(self._eligible_tickers)]
        preliminary: dict[str, dict] = {}

        for ticker, tg in df.groupby("ticker"):
            if tg["fy"].nunique() < 7:
                continue

            year_metrics: list[dict] = []
            for fy, yg in tg.groupby("fy"):
                vals = dict(zip(yg["field"], yg["value"]))

                cfo = vals.get("CashFromOperations")
                capex = vals.get("CapitalExpenditures")
                dep = vals.get("Depreciation", 0)
                op_inc = vals.get("OperatingIncome")
                net_inc = vals.get("NetIncome")
                equity = vals.get("Equity")
                debt = vals.get("TotalDebt")
                cash = vals.get("CashAndEquivalents")
                revenue = vals.get("Revenue")
                gross_profit = vals.get("GrossProfit")

                # CHANGED: Only require OperatingIncome and Equity
                # Both Revenue AND CFO are now OPTIONAL
                if any(v is None for v in (op_inc, equity)):
                    continue
                if equity <= 0:
                    continue

                # Handle optional fields
                capex = capex if capex else 0
                debt = debt if debt else 0
                cash = cash if cash else 0
                net_inc = net_inc if net_inc else op_inc * 0.75

                capex_abs = abs(capex)
                maint_capex = min(capex_abs, dep * 0.8) if dep else capex_abs * 0.5

                # CHANGED: CFO is now optional - use NetIncome + Depreciation as proxy
                if cfo is not None:
                    owner_earnings = cfo - maint_capex
                else:
                    # Proxy: NetIncome + Depreciation - maintenance CapEx
                    dep_val = dep if dep else 0
                    owner_earnings = net_inc + dep_val - maint_capex
                invested_capital = equity + debt - cash

                # CHANGED: Calculate metrics based on available data
                if revenue and revenue > 0:
                    roic = (
                        (op_inc * self.TAX_RETENTION) / invested_capital
                        if invested_capital > 0
                        else 0.0
                    )
                    operating_margin = op_inc / revenue
                    capex_intensity = capex_abs / revenue
                    revenue_for_calc = revenue
                else:
                    # Use ROE as ROIC proxy when revenue is missing
                    roic = net_inc / equity if equity > 0 else 0.0

                    if gross_profit and gross_profit > 0:
                        # Estimate revenue from gross profit
                        revenue_for_calc = gross_profit * 2.5
                        operating_margin = op_inc / revenue_for_calc
                        capex_intensity = capex_abs / revenue_for_calc
                    else:
                        revenue_for_calc = equity * 1.5
                        operating_margin = op_inc / revenue_for_calc
                        capex_intensity = capex_abs / revenue_for_calc

                year_metrics.append({
                    "fy": fy,
                    "owner_earnings": owner_earnings,
                    "roic": roic,
                    "operating_margin": operating_margin,
                    "capex_intensity": capex_intensity,
                    "revenue": revenue_for_calc,
                    "has_revenue": revenue is not None and revenue > 0,
                })

            if len(year_metrics) < 7:
                continue

            mdf = pd.DataFrame(year_metrics).sort_values("fy")

            # Moat score - adjust ROIC threshold when using ROE proxy
            has_real_revenue = mdf["has_revenue"].any()
            roic_threshold = 0.15 if has_real_revenue else 0.20

            roic_persistence = float((mdf["roic"] > roic_threshold).mean())
            margin_stability = 1.0 / (1.0 + float(mdf["operating_margin"].std()))
            capex_light = max(0.0, 1.0 - float(mdf["capex_intensity"].mean()))
            moat = 0.5 * roic_persistence + 0.3 * margin_stability + 0.2 * capex_light

            cap_years = 30 if moat > 0.8 else 20 if moat > 0.6 else 10 if moat > 0.4 else 5

            last_oe = float(mdf.iloc[-1]["owner_earnings"])
            if last_oe <= 0:
                continue

            # Revenue growth (CAGR) - use earnings growth as proxy when revenue missing
            revenues = mdf["revenue"].values
            if len(revenues) >= 5 and revenues[0] > 0:
                revenue_cagr = (revenues[-1] / revenues[0]) ** (1.0 / (len(revenues) - 1)) - 1
            else:
                revenue_cagr = 0.0

            # CHANGED: Allow higher growth for high-ROE companies
            avg_roic = float(mdf["roic"].mean())
            if has_real_revenue:
                growth = min(avg_roic * 0.5, 0.05)
            else:
                growth = min(avg_roic * 0.4, 0.06)

            r = self.DISCOUNT_RATE
            g_term = self.TERMINAL_GROWTH

            pv = sum(
                last_oe * ((1 + growth) ** t) / ((1 + r) ** t)
                for t in range(1, cap_years + 1)
            )
            terminal = last_oe * (1 + growth) ** cap_years * (1 + g_term) / (r - g_term)
            pv += terminal / ((1 + r) ** cap_years)

            if pv <= 0:
                continue

            preliminary[ticker] = {
                "pv": pv,
                "moat_score": moat,
                "growth_rate": growth,
                "revenue_cagr": revenue_cagr,
                "avg_roic": avg_roic,
                "last_owner_earnings": last_oe,
            }

        self._batch_fetch_shares_outstanding(list(preliminary.keys()))

        self._valuations = {}
        for ticker, data in preliminary.items():
            shares_out = self._shares_outstanding.get(ticker)
            if not shares_out:
                continue
            self._valuations[ticker] = {
                "intrinsic_per_share": data["pv"] / shares_out,
                "moat_score": data["moat_score"],
                "growth_rate": data["growth_rate"],
                "revenue_cagr": data["revenue_cagr"],
                "avg_roic": data["avg_roic"],
            }

        self._last_valuation_date = as_of_date


# =============================================================================
# Strategy 1: Buy and Hold Value Strategy (True Buffett Style)
# =============================================================================

class BuyAndHoldValueStrategy(BaseValueStrategy):
    """
    True Buffett-style: Buy undervalued quality companies and HOLD.

    Key differences from FundamentalStrategy:
    - Much longer holding period (only revalue annually)
    - Only sell if margin of safety becomes VERY negative (>30% overvalued)
    - Or if fundamentals deteriorate (moat score drops significantly)
    - Never sell just because price rose to fair value
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 5,
        min_margin_of_safety: float = 0.25,
        rebalance_days: int = 365,  # Annual rebalancing
        sell_threshold: float = -0.30,  # Only sell if 30% overvalued
        max_position_weight: float = 0.35,
    ):
        super().__init__(db_path)
        self.max_positions = max_positions
        self.min_mos = min_margin_of_safety
        self.rebalance_days = rebalance_days
        self.sell_threshold = sell_threshold
        self.max_position_weight = max_position_weight
        self._holdings: dict[str, dict] = {}  # ticker -> {shares, entry_price, entry_date}

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price is not None else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        if needs_rebalance:
            self._compute_all_valuations(context.date)

        # Check existing holdings - only sell if VERY overvalued
        for ticker in list(self._holdings.keys()):
            val = self._valuations.get(ticker)
            price = context.get_price(ticker)
            if val is None or price is None:
                continue

            mos = (val["intrinsic_per_share"] - price) / val["intrinsic_per_share"]
            # Only sell if significantly overvalued
            if mos < self.sell_threshold:
                del self._holdings[ticker]

        # On rebalance, look for new opportunities
        if needs_rebalance:
            candidates = []
            for ticker, val in self._valuations.items():
                if ticker in self._holdings:
                    continue  # Already own it
                price = context.get_price(ticker)
                if not price or price <= 0:
                    continue
                intrinsic = val["intrinsic_per_share"]
                mos = (intrinsic - price) / intrinsic
                if mos < self.min_mos:
                    continue
                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "intrinsic": intrinsic,
                    "margin_of_safety": mos,
                    "moat_score": val["moat_score"],
                    "score": mos * val["moat_score"] * (1 + val["growth_rate"]),
                })

            candidates.sort(key=lambda c: c["score"], reverse=True)

            # Add new positions up to max
            available_slots = self.max_positions - len(self._holdings)
            for cand in candidates[:available_slots]:
                alloc = min(self.max_position_weight, 1.0 / self.max_positions) * total_value
                shares = int(alloc / cand["price"])
                if shares > 0:
                    self._holdings[cand["ticker"]] = {
                        "shares": shares,
                        "entry_price": cand["price"],
                        "entry_date": context.date,
                    }

        # Build portfolio from holdings
        positions: dict[str, Position] = {}
        invested = 0.0
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if price is not None and holding["shares"] > 0:
                positions[ticker] = Position(
                    ticker=ticker,
                    shares=float(holding["shares"]),
                    avg_cost=holding["entry_price"],
                )
                invested += holding["shares"] * price

        return Portfolio(cash=total_value - invested, positions=positions)


# =============================================================================
# Strategy 2: Momentum + Value Strategy
# =============================================================================

class MomentumValueStrategy(BaseValueStrategy):
    """
    Combine value with momentum to avoid catching falling knives.

    Only buy when:
    - Stock is undervalued (margin of safety > threshold)
    - AND price is showing positive momentum (above 50-day or 200-day MA)

    Sell when momentum turns negative while overvalued.
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 5,
        min_margin_of_safety: float = 0.20,
        rebalance_days: int = 30,  # Monthly check
        momentum_lookback: int = 50,  # 50-day momentum
        max_position_weight: float = 0.30,
    ):
        super().__init__(db_path)
        self.max_positions = max_positions
        self.min_mos = min_margin_of_safety
        self.rebalance_days = rebalance_days
        self.momentum_lookback = momentum_lookback
        self.max_position_weight = max_position_weight
        self._target_shares: dict[str, int] = {}

    def _check_momentum(self, context: ExecutionContext, ticker: str) -> bool:
        """Return True if price is above moving average (positive momentum)."""
        hist = context.get_historical_prices(ticker, self.momentum_lookback + 10)
        if hist is None or len(hist) < self.momentum_lookback:
            return False

        closes = hist["Close"].values
        ma = np.mean(closes[-self.momentum_lookback:])
        current_price = closes[-1]
        return current_price > ma

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price is not None else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        if needs_rebalance:
            self._compute_all_valuations(context.date)

            # Find candidates with value AND momentum
            candidates = []
            for ticker, val in self._valuations.items():
                price = context.get_price(ticker)
                if not price or price <= 0:
                    continue

                intrinsic = val["intrinsic_per_share"]
                mos = (intrinsic - price) / intrinsic

                if mos < self.min_mos:
                    continue

                # Check momentum
                if not self._check_momentum(context, ticker):
                    continue

                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "margin_of_safety": mos,
                    "moat_score": val["moat_score"],
                    "score": mos * val["moat_score"],
                })

            candidates.sort(key=lambda c: c["score"], reverse=True)
            top = candidates[:self.max_positions]

            # Allocate
            self._target_shares = {}
            if top:
                weight = min(self.max_position_weight, 0.8 / len(top))
                for cand in top:
                    alloc = weight * total_value
                    shares = int(alloc / cand["price"])
                    if shares > 0:
                        self._target_shares[cand["ticker"]] = shares
        else:
            # Check if any holding lost momentum while overvalued
            for ticker in list(self._target_shares.keys()):
                val = self._valuations.get(ticker)
                if val is None:
                    continue
                price = context.get_price(ticker)
                if price is None:
                    continue

                mos = (val["intrinsic_per_share"] - price) / val["intrinsic_per_share"]
                has_momentum = self._check_momentum(context, ticker)

                # Sell if overvalued AND lost momentum
                if mos < 0 and not has_momentum:
                    del self._target_shares[ticker]

        # Build portfolio
        positions: dict[str, Position] = {}
        invested = 0.0
        for ticker, shares in self._target_shares.items():
            price = context.get_price(ticker)
            if price is not None and shares > 0:
                positions[ticker] = Position(ticker=ticker, shares=float(shares), avg_cost=price)
                invested += shares * price

        return Portfolio(cash=total_value - invested, positions=positions)


# =============================================================================
# Strategy 3: Concentrated Value Strategy
# =============================================================================

class ConcentratedValueStrategy(BaseValueStrategy):
    """
    Highly concentrated portfolio (2-3 positions) with highest conviction.

    Like Buffett's biggest bets - when you find something great, bet big.
    Requires very high margin of safety and moat score.
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 3,
        min_margin_of_safety: float = 0.35,  # Higher threshold
        min_moat_score: float = 0.6,  # Must be high quality
        rebalance_days: int = 180,  # Semi-annual
        max_position_weight: float = 0.50,  # Can go up to 50% in one stock
    ):
        super().__init__(db_path)
        self.max_positions = max_positions
        self.min_mos = min_margin_of_safety
        self.min_moat = min_moat_score
        self.rebalance_days = rebalance_days
        self.max_position_weight = max_position_weight
        self._holdings: dict[str, dict] = {}

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price is not None else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        if needs_rebalance:
            self._compute_all_valuations(context.date)

            # Very strict filtering
            candidates = []
            for ticker, val in self._valuations.items():
                price = context.get_price(ticker)
                if not price or price <= 0:
                    continue

                intrinsic = val["intrinsic_per_share"]
                mos = (intrinsic - price) / intrinsic

                # Strict criteria
                if mos < self.min_mos:
                    continue
                if val["moat_score"] < self.min_moat:
                    continue

                # Composite score emphasizing quality
                score = mos * (val["moat_score"] ** 2) * (1 + val["growth_rate"])

                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "margin_of_safety": mos,
                    "moat_score": val["moat_score"],
                    "score": score,
                })

            candidates.sort(key=lambda c: c["score"], reverse=True)

            # Take only the very best
            self._holdings = {}
            for cand in candidates[:self.max_positions]:
                # Size position by conviction (score-weighted)
                weight = min(self.max_position_weight, 0.9 / max(1, len(candidates[:self.max_positions])))
                alloc = weight * total_value
                shares = int(alloc / cand["price"])
                if shares > 0:
                    self._holdings[cand["ticker"]] = {"shares": shares, "price": cand["price"]}

        # Build portfolio
        positions: dict[str, Position] = {}
        invested = 0.0
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if price is not None and holding["shares"] > 0:
                positions[ticker] = Position(ticker=ticker, shares=float(holding["shares"]), avg_cost=holding["price"])
                invested += holding["shares"] * price

        return Portfolio(cash=total_value - invested, positions=positions)


# =============================================================================
# Strategy 4: Quality Growth Strategy
# =============================================================================

class QualityGrowthStrategy(BaseValueStrategy):
    """
    Focus on quality and growth rather than deep value.

    Willing to pay fair price for wonderful company (Buffett's later style).
    Emphasizes: high ROIC, strong moat, revenue growth.
    Lower margin of safety requirement but higher quality bar.
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 5,
        min_margin_of_safety: float = 0.10,  # Lower - willing to pay near fair value
        min_moat_score: float = 0.65,
        min_roic: float = 0.15,  # Must have 15%+ ROIC
        rebalance_days: int = 90,
        max_position_weight: float = 0.30,
    ):
        super().__init__(db_path)
        self.max_positions = max_positions
        self.min_mos = min_margin_of_safety
        self.min_moat = min_moat_score
        self.min_roic = min_roic
        self.rebalance_days = rebalance_days
        self.max_position_weight = max_position_weight
        self._target_shares: dict[str, int] = {}

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price is not None else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        if needs_rebalance:
            self._compute_all_valuations(context.date)

            candidates = []
            for ticker, val in self._valuations.items():
                price = context.get_price(ticker)
                if not price or price <= 0:
                    continue

                intrinsic = val["intrinsic_per_share"]
                mos = (intrinsic - price) / intrinsic

                # Quality filters
                if mos < self.min_mos:
                    continue
                if val["moat_score"] < self.min_moat:
                    continue
                if val["avg_roic"] < self.min_roic:
                    continue

                # Score emphasizes quality over discount
                # Quality factors weighted more heavily
                quality_score = (
                    val["moat_score"] * 0.4 +
                    min(val["avg_roic"], 0.30) / 0.30 * 0.3 +  # Normalize ROIC
                    min(val["revenue_cagr"], 0.15) / 0.15 * 0.2 +  # Normalize growth
                    mos * 0.1  # Small weight on discount
                )

                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "margin_of_safety": mos,
                    "moat_score": val["moat_score"],
                    "roic": val["avg_roic"],
                    "score": quality_score,
                })

            candidates.sort(key=lambda c: c["score"], reverse=True)
            top = candidates[:self.max_positions]

            self._target_shares = {}
            if top:
                weight = min(self.max_position_weight, 0.85 / len(top))
                for cand in top:
                    alloc = weight * total_value
                    shares = int(alloc / cand["price"])
                    if shares > 0:
                        self._target_shares[cand["ticker"]] = shares

        # Build portfolio
        positions: dict[str, Position] = {}
        invested = 0.0
        for ticker, shares in self._target_shares.items():
            price = context.get_price(ticker)
            if price is not None and shares > 0:
                positions[ticker] = Position(ticker=ticker, shares=float(shares), avg_cost=price)
                invested += shares * price

        return Portfolio(cash=total_value - invested, positions=positions)


# =============================================================================
# Strategy 5: Adaptive Value Strategy
# =============================================================================

class AdaptiveValueStrategy(BaseValueStrategy):
    """
    Dynamically adjust allocation based on market conditions.

    - When many opportunities exist (high avg margin of safety): deploy more capital
    - When few opportunities: hold more cash
    - Longer holding periods, less frequent trading
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 8,
        min_margin_of_safety: float = 0.20,
        rebalance_days: int = 120,  # Quarterly-ish
        sell_threshold: float = -0.15,  # Only sell if 15% overvalued
    ):
        super().__init__(db_path)
        self.max_positions = max_positions
        self.min_mos = min_margin_of_safety
        self.rebalance_days = rebalance_days
        self.sell_threshold = sell_threshold
        self._holdings: dict[str, dict] = {}

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price is not None else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        if needs_rebalance:
            self._compute_all_valuations(context.date)

        # Check existing holdings
        for ticker in list(self._holdings.keys()):
            val = self._valuations.get(ticker)
            price = context.get_price(ticker)
            if val is None or price is None:
                continue
            mos = (val["intrinsic_per_share"] - price) / val["intrinsic_per_share"]
            if mos < self.sell_threshold:
                del self._holdings[ticker]

        if needs_rebalance:
            # Gather all opportunities
            candidates = []
            for ticker, val in self._valuations.items():
                if ticker in self._holdings:
                    continue
                price = context.get_price(ticker)
                if not price or price <= 0:
                    continue
                intrinsic = val["intrinsic_per_share"]
                mos = (intrinsic - price) / intrinsic
                if mos < self.min_mos:
                    continue
                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "margin_of_safety": mos,
                    "moat_score": val["moat_score"],
                    "score": mos * val["moat_score"],
                })

            candidates.sort(key=lambda c: c["score"], reverse=True)

            # Adaptive allocation: more opportunities = more invested
            if candidates:
                avg_mos = np.mean([c["margin_of_safety"] for c in candidates[:10]])
                # Scale from 30% to 90% based on opportunity quality
                target_invested = min(0.90, max(0.30, 0.20 + avg_mos * 2))
            else:
                target_invested = 0.30

            # Fill available slots
            available_slots = self.max_positions - len(self._holdings)
            for cand in candidates[:available_slots]:
                per_position = target_invested / self.max_positions
                alloc = per_position * total_value
                shares = int(alloc / cand["price"])
                if shares > 0:
                    self._holdings[cand["ticker"]] = {"shares": shares, "price": cand["price"]}

        # Build portfolio
        positions: dict[str, Position] = {}
        invested = 0.0
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if price is not None and holding["shares"] > 0:
                positions[ticker] = Position(ticker=ticker, shares=float(holding["shares"]), avg_cost=holding["price"])
                invested += holding["shares"] * price

        return Portfolio(cash=total_value - invested, positions=positions)


# =============================================================================
# Utility: Compare multiple strategies
# =============================================================================

def get_all_strategies(db_path: str = "fundamentals.sqlite") -> dict[str, Strategy]:
    """Return a dict of all available strategies for easy testing."""
    return {
        "BuyAndHold": BuyAndHoldValueStrategy(db_path),
        "MomentumValue": MomentumValueStrategy(db_path),
        "Concentrated": ConcentratedValueStrategy(db_path),
        "QualityGrowth": QualityGrowthStrategy(db_path),
        "Adaptive": AdaptiveValueStrategy(db_path),
    }
