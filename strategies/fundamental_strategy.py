import sqlite3
import concurrent.futures
from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from strategies.strategy import Strategy, Portfolio, Position, ExecutionContext


class FundamentalStrategy(Strategy):
    """Buffett-style value investing strategy.

    Periodically recomputes intrinsic valuations from SEC fundamental data,
    compares them with market prices, selects the top undervalued stocks,
    and optimises portfolio allocation via constrained optimisation.

    Between rebalance dates the strategy monitors held positions and exits
    any that have appreciated past their intrinsic value.  A cash buffer
    is always maintained — its size scales inversely with opportunity
    quality so the strategy never blindly deploys all capital.
    """

    DISCOUNT_RATE = 0.09
    TERMINAL_GROWTH = 0.02
    TAX_RETENTION = 0.79  # 1 − 21 % corporate tax
    SHARES_CACHE_DAYS = 30  # Refresh shares outstanding every 30 days

    def __init__(
        self,
        db_path: str = "fundamentals.sqlite",
        max_positions: int = 5,
        min_margin_of_safety: float = 0.20,
        rebalance_days: int = 90,
        max_position_weight: float = 0.35,
    ):
        self.db_path = db_path
        self.max_positions = max_positions
        self.min_mos = min_margin_of_safety
        self.rebalance_days = rebalance_days
        self.max_position_weight = max_position_weight

        self._eligible_tickers: Optional[set[str]] = None
        self._shares_outstanding: dict[str, Optional[float]] = {}
        self._valuations: dict[str, dict] = {}
        self._last_valuation_date: Optional[datetime] = None
        self._target_shares: dict[str, int] = {}
        self._init_shares_cache()

    # ── one-time setup ────────────────────────────────────────────────

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
        """Tickers with >= 7 distinct fiscal years in the income statement."""
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

    # ── shares outstanding ────────────────────────────────────────────

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

    # ── valuation engine ──────────────────────────────────────────────

    def _compute_all_valuations(self, as_of_date: datetime) -> None:
        """Recompute intrinsic values using only data filed before *as_of_date*.

        CHANGED: Revenue is now OPTIONAL. When missing, we use:
        - GrossProfit as a revenue proxy for margin calculations
        - ROE (NetIncome/Equity) as ROIC proxy when invested capital calc fails
        - This allows companies like Apple (missing Revenue 2012-2017) to be valued.
        """

        conn = sqlite3.connect(self.db_path)
        date_str = as_of_date.strftime("%Y-%m-%d")

        # One bulk read — filtered only by date, ticker filtering in Python.
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

        # Keep only eligible tickers.
        df = df[df["ticker"].isin(self._eligible_tickers)]

        # ── Phase 1: compute fundamental metrics (no shares needed) ───
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
                else:
                    # When revenue missing: use ROE as quality metric, estimate margins
                    roe = net_inc / equity if equity > 0 else 0
                    roic = roe  # Use ROE as ROIC proxy

                    if gross_profit and gross_profit > 0:
                        # Estimate revenue from gross profit (typical GP ratio ~40%)
                        est_revenue = gross_profit * 2.5
                        operating_margin = op_inc / est_revenue
                        capex_intensity = capex_abs / est_revenue
                    else:
                        # Last resort: use equity as denominator
                        operating_margin = op_inc / (equity * 1.5)
                        capex_intensity = capex_abs / (equity * 1.5)

                year_metrics.append(
                    {
                        "fy": fy,
                        "owner_earnings": owner_earnings,
                        "roic": roic,
                        "operating_margin": operating_margin,
                        "capex_intensity": capex_intensity,
                        "has_revenue": revenue is not None and revenue > 0,
                    }
                )

            if len(year_metrics) < 7:
                continue

            mdf = pd.DataFrame(year_metrics).sort_values("fy")

            # Moat score (mirrors fundamental2.py)
            # CHANGED: Adjust ROIC threshold when using ROE proxy (ROE is typically higher)
            has_real_revenue = mdf["has_revenue"].any()
            roic_threshold = 0.15 if has_real_revenue else 0.20  # Higher bar for ROE

            roic_persistence = float((mdf["roic"] > roic_threshold).mean())
            margin_stability = 1.0 / (1.0 + float(mdf["operating_margin"].std()))
            capex_light = max(0.0, 1.0 - float(mdf["capex_intensity"].mean()))
            moat = 0.5 * roic_persistence + 0.3 * margin_stability + 0.2 * capex_light

            cap_years = (
                30 if moat > 0.8 else 20 if moat > 0.6 else 10 if moat > 0.4 else 5
            )

            last_oe = float(mdf.iloc[-1]["owner_earnings"])
            if last_oe <= 0:
                continue

            # CHANGED: Allow higher growth for high-ROE companies
            avg_roic = float(mdf["roic"].mean())
            if has_real_revenue:
                growth = min(avg_roic * 0.5, 0.05)
            else:
                # For ROE-based metrics, allow slightly higher growth (ROE typically > ROIC)
                growth = min(avg_roic * 0.4, 0.06)  # Up to 6% for high-ROE companies

            r = self.DISCOUNT_RATE
            g_term = self.TERMINAL_GROWTH

            pv = sum(
                last_oe * ((1 + growth) ** t) / ((1 + r) ** t)
                for t in range(1, cap_years + 1)
            )
            terminal = (
                last_oe * (1 + growth) ** cap_years * (1 + g_term) / (r - g_term)
            )
            pv += terminal / ((1 + r) ** cap_years)

            if pv <= 0:
                continue

            preliminary[ticker] = {
                "pv": pv,
                "moat_score": moat,
                "growth_rate": growth,
            }

        # ── Phase 2: batch-fetch shares outstanding ───────────────────
        self._batch_fetch_shares_outstanding(list(preliminary.keys()))

        # ── Phase 3: intrinsic per share ──────────────────────────────
        self._valuations = {}
        for ticker, data in preliminary.items():
            shares_out = self._shares_outstanding.get(ticker)
            if not shares_out:
                continue
            self._valuations[ticker] = {
                "intrinsic_per_share": data["pv"] / shares_out,
                "moat_score": data["moat_score"],
                "growth_rate": data["growth_rate"],
            }

        self._last_valuation_date = as_of_date

    # ── candidate ranking ─────────────────────────────────────────────

    def _rank_candidates(self, context: ExecutionContext) -> list[dict]:
        """Return undervalued stocks sorted by composite score (descending)."""
        candidates: list[dict] = []

        for ticker, val in self._valuations.items():
            price = context.get_price(ticker)
            if not price or price <= 0:
                continue

            intrinsic = val["intrinsic_per_share"]
            mos = (intrinsic - price) / intrinsic
            if mos < self.min_mos:
                continue

            candidates.append(
                {
                    "ticker": ticker,
                    "price": price,
                    "intrinsic": intrinsic,
                    "margin_of_safety": mos,
                    "moat_score": val["moat_score"],
                    "growth_rate": val["growth_rate"],
                    "score": mos * val["moat_score"] * (1.0 + val["growth_rate"]),
                }
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    # ── portfolio optimisation ────────────────────────────────────────

    def _optimize_weights(self, candidates: list[dict]) -> np.ndarray:
        """Optimise allocation weights for the selected candidates.

        The total invested fraction scales with opportunity quality:
        stronger average margin-of-safety → more capital deployed,
        but always at least 10 % cash.  Each position is capped at
        ``max_position_weight``.
        """
        n = len(candidates)
        if n == 0:
            return np.array([])

        scores = np.array([c["score"] for c in candidates])
        mos_values = np.array([c["margin_of_safety"] for c in candidates])

        avg_mos = float(np.mean(mos_values))
        max_total = min(0.90, max(0.30, 0.20 + 1.5 * avg_mos))
        cap = self.max_position_weight

        def objective(w):
            expected = np.dot(w, scores)
            concentration = np.sum(w ** 2)
            return -(expected - 0.5 * concentration)

        bounds = [(0.0, cap)] * n
        constraints = [{"type": "ineq", "fun": lambda w: max_total - np.sum(w)}]
        x0 = np.full(n, min(max_total / n, cap))

        try:
            from scipy.optimize import minimize

            res = minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )
            if res.success:
                return np.clip(res.x, 0.0, cap)
        except Exception:
            pass

        # Fallback: score-proportional allocation.
        total_score = scores.sum()
        if total_score > 0:
            return np.clip(scores / total_score * max_total, 0.0, cap)
        return np.full(n, max_total / n)

    # ── main entry point ──────────────────────────────────────────────

    def execute(self, context: ExecutionContext) -> Portfolio:
        if self._eligible_tickers is None:
            self._load_eligible_tickers()

        portfolio = context.portfolio

        # Current total portfolio value.
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price is not None else pos.avg_cost)

        needs_rebalance = (
            self._last_valuation_date is None
            or (context.date - self._last_valuation_date).days >= self.rebalance_days
        )

        if needs_rebalance:
            # ── full rebalance ────────────────────────────────────────
            self._compute_all_valuations(context.date)
            candidates = self._rank_candidates(context)

            self._target_shares = {}
            if candidates:
                top = candidates[: self.max_positions]
                weights = self._optimize_weights(top)
                for i, cand in enumerate(top):
                    alloc = weights[i] * total_value
                    shares = int(alloc / cand["price"])
                    if shares > 0:
                        self._target_shares[cand["ticker"]] = shares
        else:
            # ── between rebalances: exit overvalued positions ─────────
            for ticker in list(self._target_shares):
                val = self._valuations.get(ticker)
                if val is None:
                    del self._target_shares[ticker]
                    continue
                price = context.get_price(ticker)
                if price is None:
                    continue
                if price > val["intrinsic_per_share"]:
                    del self._target_shares[ticker]

        # ── build desired portfolio ───────────────────────────────────
        positions: dict[str, Position] = {}
        invested = 0.0
        for ticker, shares in self._target_shares.items():
            price = context.get_price(ticker)
            if price is not None and shares > 0:
                positions[ticker] = Position(
                    ticker=ticker,
                    shares=float(shares),
                    avg_cost=price,
                )
                invested += shares * price

        return Portfolio(cash=total_value - invested, positions=positions)
