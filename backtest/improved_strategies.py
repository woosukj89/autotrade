"""
Improved value investing strategies based on performance analysis.

Key changes from original strategies:
1. Less conservative valuation (higher growth allowance for quality companies)
2. Focus on compounders, not just cheap stocks
3. Truly long-term holding (only sell on fundamental deterioration)
4. Sector diversification
5. Market regime awareness
"""

import sqlite3
import concurrent.futures
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from strategies.strategy import Strategy, Portfolio, Position, ExecutionContext


class ImprovedBaseStrategy(Strategy):
    """Base class with improved valuation logic."""

    SHARES_CACHE_DAYS = 30  # Refresh shares outstanding every 30 days

    def __init__(self, db_path: str = "fundamentals.sqlite"):
        self.db_path = db_path
        self._eligible_tickers: Optional[set[str]] = None
        self._shares_outstanding: dict[str, Optional[float]] = {}
        self._valuations: dict[str, dict] = {}
        self._last_valuation_date: Optional[datetime] = None
        self._ticker_sectors: dict[str, str] = {}
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
            SELECT f.ticker, COUNT(DISTINCT f.fy) AS years, c.industry
            FROM fundamentals f
            LEFT JOIN companies c ON f.ticker = c.ticker
            WHERE f.statement_type = 'income'
            GROUP BY f.ticker
            HAVING years >= 5
            """,
            conn,
        )
        conn.close()
        self._eligible_tickers = set(df["ticker"])
        self._ticker_sectors = dict(zip(df["ticker"], df["industry"].fillna("Unknown")))

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

    def _compute_valuations_improved(self, as_of_date: datetime) -> None:
        """Improved valuation with dynamic growth and quality metrics.

        Key change: Revenue is now OPTIONAL. When missing, we use:
        - GrossProfit as a revenue proxy for margin calculations
        - ROE (NetIncome/Equity) instead of ROIC
        - Earnings growth instead of revenue growth

        This allows companies like Apple (missing Revenue 2012-2017) to be valued.
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
            if tg["fy"].nunique() < 5:
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

                # Use NetIncome or derive from OperatingIncome
                net_inc = net_inc if net_inc else op_inc * 0.75

                capex = capex if capex else 0
                debt = debt if debt else 0
                cash = cash if cash else 0

                capex_abs = abs(capex)

                # CHANGED: CFO is now optional - use NetIncome + Depreciation as proxy
                if cfo is not None:
                    owner_earnings = cfo - min(capex_abs, dep * 0.8 if dep else capex_abs * 0.5)
                else:
                    # Proxy: NetIncome + Depreciation - CapEx (approximates CFO - CapEx)
                    dep_val = dep if dep else 0
                    owner_earnings = net_inc + dep_val - capex_abs * 0.5  # Conservative capex
                invested_capital = max(equity + debt - cash, equity)

                # CHANGED: Calculate ROIC only if we have revenue, otherwise use ROE
                # For margin, use gross_profit as a proxy when revenue is missing
                if revenue and revenue > 0:
                    roic = (op_inc * 0.75) / invested_capital if invested_capital > 0 else 0
                    operating_margin = op_inc / revenue
                    revenue_for_calc = revenue
                else:
                    # When revenue is missing, use ROE as quality metric
                    # and estimate operating margin from gross profit if available
                    roic = net_inc / equity if equity > 0 else 0  # Use ROE as ROIC proxy
                    if gross_profit and gross_profit > 0:
                        # Estimate margin from gross profit and operating income ratio
                        operating_margin = op_inc / (gross_profit * 2.5)  # Typical revenue/GP ratio
                        revenue_for_calc = gross_profit * 2.5
                    else:
                        operating_margin = op_inc / (equity * 1.5)  # Use equity as proxy
                        revenue_for_calc = equity * 1.5

                roe = net_inc / equity if equity > 0 else 0

                year_metrics.append({
                    "fy": fy,
                    "revenue": revenue_for_calc,
                    "net_income": net_inc,
                    "owner_earnings": owner_earnings,
                    "roic": roic,
                    "roe": roe,
                    "operating_margin": operating_margin,
                    "equity": equity,
                    "has_revenue": revenue is not None and revenue > 0,
                })

            if len(year_metrics) < 5:
                continue

            mdf = pd.DataFrame(year_metrics).sort_values("fy")

            # Growth metrics - use earnings when revenue is missing
            revenues = mdf["revenue"].values
            earnings = mdf["net_income"].values
            has_real_revenue = mdf["has_revenue"].any()

            if revenues[0] > 0 and revenues[-1] > 0:
                revenue_cagr = (revenues[-1] / revenues[0]) ** (1.0 / (len(revenues) - 1)) - 1
            else:
                revenue_cagr = 0

            if earnings[0] > 0 and earnings[-1] > 0:
                earnings_cagr = (earnings[-1] / earnings[0]) ** (1.0 / (len(earnings) - 1)) - 1
            else:
                earnings_cagr = 0

            # CHANGED: When revenue is missing, rely more heavily on earnings growth
            if not has_real_revenue:
                revenue_cagr = earnings_cagr  # Use earnings growth as proxy

            # Quality metrics
            avg_roic = float(mdf["roic"].mean())
            avg_roe = float(mdf["roe"].mean())

            # CHANGED: Use higher threshold for ROE-based ROIC proxy (ROE is typically higher)
            roic_threshold = 0.12 if has_real_revenue else 0.18
            roic_consistency = float((mdf["roic"] > roic_threshold).mean())

            margin_stability = 1.0 / (1.0 + float(mdf["operating_margin"].std()))

            # CHANGED: Adjust moat formula when using ROE as ROIC proxy
            if has_real_revenue:
                moat = (
                    0.35 * min(avg_roic / 0.20, 1.0) +
                    0.25 * roic_consistency +
                    0.20 * margin_stability +
                    0.20 * min(max(revenue_cagr, 0) / 0.10, 1.0)
                )
            else:
                # When using ROE proxy, weight ROE more heavily
                moat = (
                    0.30 * min(avg_roe / 0.25, 1.0) +  # ROE normalized to 25%
                    0.30 * roic_consistency +
                    0.20 * margin_stability +
                    0.20 * min(max(earnings_cagr, 0) / 0.10, 1.0)
                )

            # Dynamic growth estimate based on quality
            base_growth = max(0, min(revenue_cagr, earnings_cagr, 0.15))

            # CHANGED: For companies with high ROE (like Apple), allow higher growth
            high_quality = (avg_roic > 0.15 and moat > 0.6) or (avg_roe > 0.25 and moat > 0.55)
            if high_quality:
                growth_allowance = min(base_growth * 1.5, 0.12)
            else:
                growth_allowance = min(base_growth, 0.06)

            # DCF with improved parameters
            last_oe = float(mdf.iloc[-1]["owner_earnings"])
            if last_oe <= 0:
                last_oe = float(mdf.iloc[-1]["net_income"]) * 0.8
                if last_oe <= 0:
                    continue

            # Discount rate based on quality
            if moat > 0.7 and (avg_roic > 0.15 or avg_roe > 0.25):
                discount_rate = 0.08
            elif moat > 0.5:
                discount_rate = 0.09
            else:
                discount_rate = 0.10

            cap_years = 15 if moat > 0.7 else 10 if moat > 0.5 else 7

            pv = 0
            for t in range(1, cap_years + 1):
                year_growth = growth_allowance * (1 - t / (cap_years * 2))
                pv += last_oe * ((1 + year_growth) ** t) / ((1 + discount_rate) ** t)

            terminal_growth = 0.025
            terminal = last_oe * (1 + growth_allowance) ** cap_years * (1 + terminal_growth)
            terminal /= (discount_rate - terminal_growth)
            pv += terminal / ((1 + discount_rate) ** cap_years)

            preliminary[ticker] = {
                "pv": pv,
                "moat_score": moat,
                "growth_rate": growth_allowance,
                "revenue_cagr": revenue_cagr,
                "earnings_cagr": earnings_cagr,
                "avg_roic": avg_roic,
                "avg_roe": avg_roe,
                "last_earnings": float(mdf.iloc[-1]["net_income"]),
                "has_revenue_data": has_real_revenue,
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
                "avg_roe": data["avg_roe"],
                "eps": data["last_earnings"] / shares_out,
                "sector": self._ticker_sectors.get(ticker, "Unknown"),
            }

        self._last_valuation_date = as_of_date


# =============================================================================
# Strategy 1: Compounders Strategy
# =============================================================================

class CompoundersStrategy(ImprovedBaseStrategy):
    """
    Focus on high-quality compounders - companies that can grow earnings
    at high rates for extended periods.

    Key insight: It's better to buy a wonderful company at a fair price
    than a fair company at a wonderful price.

    Selection criteria:
    - High and consistent ROIC (>15%)
    - Strong revenue/earnings growth
    - High moat score
    - Willing to pay up to fair value (not requiring deep discount)
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 10,
        min_roic: float = 0.15,
        min_moat: float = 0.55,
        max_premium: float = 0.20,  # Will pay up to 20% above intrinsic value
        rebalance_days: int = 180,
        max_sector_weight: float = 0.35,
    ):
        super().__init__(db_path)
        self.max_positions = max_positions
        self.min_roic = min_roic
        self.min_moat = min_moat
        self.max_premium = max_premium
        self.rebalance_days = rebalance_days
        self.max_sector_weight = max_sector_weight
        self._holdings: dict[str, dict] = {}

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        if needs_rebalance:
            self._compute_valuations_improved(context.date)

            # Find compounders
            candidates = []
            for ticker, val in self._valuations.items():
                price = context.get_price(ticker)
                if not price or price <= 0:
                    continue

                # Quality filters
                if val["avg_roic"] < self.min_roic:
                    continue
                if val["moat_score"] < self.min_moat:
                    continue

                intrinsic = val["intrinsic_per_share"]
                premium = (price - intrinsic) / intrinsic if intrinsic > 0 else 1

                # Willing to pay slight premium for quality
                if premium > self.max_premium:
                    continue

                # Score emphasizes quality and growth over discount
                score = (
                    val["moat_score"] * 0.3 +
                    min(val["avg_roic"] / 0.25, 1.0) * 0.3 +
                    min(max(val["revenue_cagr"], 0) / 0.15, 1.0) * 0.25 +
                    max(-premium, 0) * 0.15  # Bonus for discount, not required
                )

                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "intrinsic": intrinsic,
                    "premium": premium,
                    "moat_score": val["moat_score"],
                    "roic": val["avg_roic"],
                    "growth": val["revenue_cagr"],
                    "sector": val["sector"],
                    "score": score,
                })

            # Sort by score
            candidates.sort(key=lambda c: c["score"], reverse=True)

            # Sector-aware selection
            self._holdings = {}
            sector_weights = defaultdict(float)

            for cand in candidates:
                if len(self._holdings) >= self.max_positions:
                    break

                sector = cand["sector"]
                current_sector_weight = sector_weights[sector]

                if current_sector_weight >= self.max_sector_weight:
                    continue

                # Equal weight positions
                weight = 0.9 / self.max_positions
                alloc = weight * total_value
                shares = int(alloc / cand["price"])

                if shares > 0:
                    self._holdings[cand["ticker"]] = {
                        "shares": shares,
                        "entry_price": cand["price"],
                        "sector": sector,
                    }
                    sector_weights[sector] += weight

        # Build portfolio
        positions = {}
        invested = 0.0
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if price and holding["shares"] > 0:
                positions[ticker] = Position(ticker, float(holding["shares"]), holding["entry_price"])
                invested += holding["shares"] * price

        return Portfolio(cash=total_value - invested, positions=positions)


# =============================================================================
# Strategy 2: True Buffett Strategy
# =============================================================================

class TrueBuffettStrategy(ImprovedBaseStrategy):
    """
    Mimics Buffett's actual approach more closely:

    1. Concentrated portfolio (5-8 positions)
    2. Hold for very long periods (only sell on fundamental deterioration)
    3. Focus on businesses with durable competitive advantages
    4. Prefer companies with pricing power and strong brands
    5. Require margin of safety but accept fair price for exceptional quality
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 6,
        min_roic: float = 0.15,
        min_moat: float = 0.60,
        min_margin_of_safety: float = 0.10,
        fundamental_deterioration_threshold: float = 0.25,  # Sell if ROIC drops 25%
        rebalance_days: int = 365,  # Annual review
        max_position_weight: float = 0.40,
    ):
        super().__init__(db_path)
        self.max_positions = max_positions
        self.min_roic = min_roic
        self.min_moat = min_moat
        self.min_mos = min_margin_of_safety
        self.deterioration_threshold = fundamental_deterioration_threshold
        self.rebalance_days = rebalance_days
        self.max_position_weight = max_position_weight
        self._holdings: dict[str, dict] = {}
        self._entry_fundamentals: dict[str, dict] = {}

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        if needs_rebalance:
            self._compute_valuations_improved(context.date)

            # Check existing holdings for fundamental deterioration
            for ticker in list(self._holdings.keys()):
                val = self._valuations.get(ticker)
                entry = self._entry_fundamentals.get(ticker)

                if val is None or entry is None:
                    continue

                # Sell only if fundamentals deteriorated significantly
                roic_decline = (entry["roic"] - val["avg_roic"]) / entry["roic"] if entry["roic"] > 0 else 0
                moat_decline = (entry["moat"] - val["moat_score"]) / entry["moat"] if entry["moat"] > 0 else 0

                if roic_decline > self.deterioration_threshold or moat_decline > self.deterioration_threshold:
                    del self._holdings[ticker]
                    del self._entry_fundamentals[ticker]

            # Look for new opportunities if we have room
            if len(self._holdings) < self.max_positions:
                candidates = []
                for ticker, val in self._valuations.items():
                    if ticker in self._holdings:
                        continue

                    price = context.get_price(ticker)
                    if not price or price <= 0:
                        continue

                    # Quality requirements
                    if val["avg_roic"] < self.min_roic:
                        continue
                    if val["moat_score"] < self.min_moat:
                        continue

                    intrinsic = val["intrinsic_per_share"]
                    mos = (intrinsic - price) / intrinsic if intrinsic > 0 else -1

                    if mos < self.min_mos:
                        continue

                    # Buffett scoring: quality first, then value
                    score = (
                        val["moat_score"] ** 2 * 0.4 +  # Moat squared - he loves moats
                        min(val["avg_roic"] / 0.25, 1.0) * 0.3 +
                        mos * 0.2 +
                        min(val["avg_roe"] / 0.20, 1.0) * 0.1
                    )

                    candidates.append({
                        "ticker": ticker,
                        "price": price,
                        "intrinsic": intrinsic,
                        "mos": mos,
                        "roic": val["avg_roic"],
                        "moat": val["moat_score"],
                        "score": score,
                    })

                candidates.sort(key=lambda c: c["score"], reverse=True)

                # Add new positions
                available_slots = self.max_positions - len(self._holdings)
                for cand in candidates[:available_slots]:
                    # Position sizing: higher conviction = larger position
                    base_weight = 0.85 / self.max_positions
                    conviction_multiplier = 1 + (cand["score"] - 0.5) * 0.5
                    weight = min(base_weight * conviction_multiplier, self.max_position_weight)

                    alloc = weight * total_value
                    shares = int(alloc / cand["price"])

                    if shares > 0:
                        self._holdings[cand["ticker"]] = {
                            "shares": shares,
                            "entry_price": cand["price"],
                            "entry_date": context.date,
                        }
                        self._entry_fundamentals[cand["ticker"]] = {
                            "roic": cand["roic"],
                            "moat": cand["moat"],
                        }

        # Build portfolio
        positions = {}
        invested = 0.0
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if price and holding["shares"] > 0:
                positions[ticker] = Position(ticker, float(holding["shares"]), holding["entry_price"])
                invested += holding["shares"] * price

        return Portfolio(cash=total_value - invested, positions=positions)


# =============================================================================
# Strategy 3: Market Regime Strategy
# =============================================================================

class MarketRegimeStrategy(ImprovedBaseStrategy):
    """
    Adjusts exposure based on market conditions:

    - Bull market (SPY above 200-day MA): Full investment in quality stocks
    - Bear market (SPY below 200-day MA): Reduce exposure, hold more cash
    - Crash (SPY down >20% from peak): Deploy cash aggressively

    This captures Buffett's "be greedy when others are fearful" approach.
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 8,
        min_moat: float = 0.50,
        min_margin_of_safety: float = 0.15,
        rebalance_days: int = 30,
        bull_allocation: float = 0.90,
        bear_allocation: float = 0.50,
        crash_allocation: float = 1.00,  # Go all-in during crashes
    ):
        super().__init__(db_path)
        self.max_positions = max_positions
        self.min_moat = min_moat
        self.min_mos = min_margin_of_safety
        self.rebalance_days = rebalance_days
        self.bull_alloc = bull_allocation
        self.bear_alloc = bear_allocation
        self.crash_alloc = crash_allocation
        self._target_shares: dict[str, int] = {}
        self._spy_peak: float = 0

    def _get_market_regime(self, context: ExecutionContext) -> str:
        """Determine current market regime based on SPY."""
        spy_hist = context.get_historical_prices("SPY", 250)
        if spy_hist is None or len(spy_hist) < 200:
            return "neutral"

        current_price = float(spy_hist["Close"].iloc[-1])
        ma_200 = float(spy_hist["Close"].tail(200).mean())
        recent_peak = float(spy_hist["Close"].max())

        # Update all-time high tracking
        if current_price > self._spy_peak:
            self._spy_peak = current_price

        drawdown = (self._spy_peak - current_price) / self._spy_peak if self._spy_peak > 0 else 0

        if drawdown > 0.20:
            return "crash"
        elif current_price < ma_200:
            return "bear"
        else:
            return "bull"

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        # Get market regime
        regime = self._get_market_regime(context)
        if regime == "crash":
            target_allocation = self.crash_alloc
        elif regime == "bear":
            target_allocation = self.bear_alloc
        else:
            target_allocation = self.bull_alloc

        if needs_rebalance:
            self._compute_valuations_improved(context.date)

            candidates = []
            for ticker, val in self._valuations.items():
                price = context.get_price(ticker)
                if not price or price <= 0:
                    continue

                if val["moat_score"] < self.min_moat:
                    continue

                intrinsic = val["intrinsic_per_share"]
                mos = (intrinsic - price) / intrinsic if intrinsic > 0 else -1

                # In crash, accept lower margin of safety (buying at better prices)
                required_mos = self.min_mos * 0.5 if regime == "crash" else self.min_mos
                if mos < required_mos:
                    continue

                score = val["moat_score"] * 0.4 + mos * 0.3 + min(val["avg_roic"] / 0.20, 1.0) * 0.3

                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "mos": mos,
                    "score": score,
                })

            candidates.sort(key=lambda c: c["score"], reverse=True)
            top = candidates[:self.max_positions]

            self._target_shares = {}
            if top:
                per_position = target_allocation / len(top)
                for cand in top:
                    alloc = per_position * total_value
                    shares = int(alloc / cand["price"])
                    if shares > 0:
                        self._target_shares[cand["ticker"]] = shares

        # Build portfolio
        positions = {}
        invested = 0.0
        for ticker, shares in self._target_shares.items():
            price = context.get_price(ticker)
            if price and shares > 0:
                positions[ticker] = Position(ticker, float(shares), price)
                invested += shares * price

        return Portfolio(cash=total_value - invested, positions=positions)


# =============================================================================
# Strategy 4: Earnings Growth Strategy
# =============================================================================

class EarningsGrowthStrategy(ImprovedBaseStrategy):
    """
    Focus purely on earnings growth - buy companies growing earnings fast
    regardless of traditional value metrics.

    This captures what actually drove S&P returns: earnings growth,
    not cheap valuation multiples.

    CHANGED: Now uses ROE as quality metric when ROIC is ROE-based (missing revenue).
    """

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 10,
        min_earnings_growth: float = 0.10,  # 10% earnings CAGR minimum
        min_revenue_growth: float = 0.08,   # 8% revenue CAGR minimum
        min_roic: float = 0.12,
        min_roe: float = 0.18,  # Alternative for companies without revenue data
        rebalance_days: int = 90,
        max_pe_ratio: float = 40,  # Avoid extreme valuations
    ):
        super().__init__(db_path)
        self.max_positions = max_positions
        self.min_earnings_growth = min_earnings_growth
        self.min_revenue_growth = min_revenue_growth
        self.min_roic = min_roic
        self.min_roe = min_roe
        self.rebalance_days = rebalance_days
        self.max_pe = max_pe_ratio
        self._target_shares: dict[str, int] = {}

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        if needs_rebalance:
            self._compute_valuations_improved(context.date)

            candidates = []
            for ticker, val in self._valuations.items():
                price = context.get_price(ticker)
                if not price or price <= 0:
                    continue

                # Growth requirements - use earnings growth (works with/without revenue)
                if val["revenue_cagr"] < self.min_revenue_growth:
                    continue

                # CHANGED: Quality check uses ROIC or ROE depending on data availability
                # When revenue is missing, avg_roic is actually ROE (see _compute_valuations_improved)
                quality_metric = max(val["avg_roic"], val["avg_roe"])
                if quality_metric < self.min_roic:
                    continue

                # Valuation sanity check
                eps = val["eps"]
                if eps <= 0:
                    continue
                pe_ratio = price / eps
                if pe_ratio > self.max_pe or pe_ratio < 0:
                    continue

                # Score based on growth and quality
                score = (
                    min(val["revenue_cagr"] / 0.20, 1.0) * 0.35 +  # Revenue/Earnings growth
                    min(quality_metric / 0.25, 1.0) * 0.30 +      # Quality (ROIC or ROE)
                    val["moat_score"] * 0.20 +                      # Moat
                    max(0, (self.max_pe - pe_ratio) / self.max_pe) * 0.15  # Reasonable valuation
                )

                candidates.append({
                    "ticker": ticker,
                    "price": price,
                    "growth": val["revenue_cagr"],
                    "roic": val["avg_roic"],
                    "roe": val["avg_roe"],
                    "pe": pe_ratio,
                    "score": score,
                })

            candidates.sort(key=lambda c: c["score"], reverse=True)
            top = candidates[:self.max_positions]

            self._target_shares = {}
            if top:
                weight = 0.90 / len(top)
                for cand in top:
                    alloc = weight * total_value
                    shares = int(alloc / cand["price"])
                    if shares > 0:
                        self._target_shares[cand["ticker"]] = shares

        # Build portfolio
        positions = {}
        invested = 0.0
        for ticker, shares in self._target_shares.items():
            price = context.get_price(ticker)
            if price and shares > 0:
                positions[ticker] = Position(ticker, float(shares), price)
                invested += shares * price

        return Portfolio(cash=total_value - invested, positions=positions)


# =============================================================================
# Utility function
# =============================================================================

def get_improved_strategies(db_path: str = "fundamentals.sqlite") -> dict[str, Strategy]:
    """Return all improved strategies for testing."""
    return {
        "Compounders": CompoundersStrategy(db_path),
        "TrueBuffett": TrueBuffettStrategy(db_path),
        "MarketRegime": MarketRegimeStrategy(db_path),
        "EarningsGrowth": EarningsGrowthStrategy(db_path),
    }
