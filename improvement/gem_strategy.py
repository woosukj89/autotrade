"""
Global Equities Momentum (GEM) - Gary Antonacci's published, real-world
long-only dual-momentum strategy (from "Dual Momentum Investing", 2014).
Genuinely different in kind from everything else tried this session:
instead of picking individual high-beta/momentum STOCKS (concentrated,
high idiosyncratic AND systematic risk), it rotates among a small set of
BROAD equity asset classes (each one already a diversified basket, much
lower idiosyncratic risk per holding) - and holds only ONE at a time.

Classic GEM logic, monthly:
  1. Absolute momentum: is the domestic equity asset's trailing 12-month
     return positive? (Antonacci compares to T-bill return; simplified
     here to compare against 0%, since holding cash - not bonds - is the
     defensive fallback, per the user's long-only stocks/stock-equivalents
     preference - no fixed income.)
  2. If yes: relative momentum - hold whichever RISKY asset (among the
     configured list, e.g. domestic vs international equities) had the
     higher trailing 12-month return.
  3. If no (domestic momentum negative): go to cash entirely.

This is NOT a switching/timing overlay bolted onto a stock-picker (every
version of that failed this session, Iterations 25-27) - it's a complete,
standalone, minimal-turnover strategy in its own right: at most one
position at a time, rebalanced monthly, no per-stock trend filters, no
market filters, no stop-losses. Real-world GEM's published long-run
CAGR is lower than concentrated stock momentum (documented ~17-18%
annualized 1974-2020) but with dramatically lower MaxDD (~-17.8% is the
commonly cited figure for GEM's worst historical drawdown) - testing
whether that holds in THIS backtest, on THIS 20yr window, with real costs.
"""
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
from strategy import Strategy, Portfolio, Position, ExecutionContext


class GemStrategy(Strategy):
    def __init__(
        self,
        risky_assets: Optional[List[str]] = None,   # candidates to rotate among
        domestic_asset: str = 'SPY',                 # used for the absolute-momentum gate
        lookback_days: int = 252,
        rebalance_days: int = 30,
        top_n: int = 1,                               # classic GEM holds exactly 1
    ):
        self.risky_assets = risky_assets or ['SPY', 'EFA', 'EEM', 'VNQ']
        self.domestic_asset = domestic_asset
        self.lookback_days = lookback_days
        self.rebalance_days = rebalance_days
        self.top_n = top_n

        self._holdings: Dict[str, dict] = {}
        self._last_rebalance = None

    def _trailing_return(self, ticker: str, context: ExecutionContext) -> Optional[float]:
        hist = context.get_historical_prices(ticker, self.lookback_days + 5)
        if hist is None or len(hist) < self.lookback_days * 0.9:
            return None
        closes = hist['Close'].dropna()
        if len(closes) < 20:
            return None
        window = min(self.lookback_days, len(closes) - 1)
        p_end = float(closes.iloc[-1])
        p_start = float(closes.iloc[-1 - window])
        if p_start <= 0:
            return None
        return (p_end / p_start) - 1.0

    def execute(self, context: ExecutionContext) -> Portfolio:
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
            print(f"[GEM] Rebalancing on {context.date.strftime('%Y-%m-%d')}...")
            domestic_mom = self._trailing_return(self.domestic_asset, context)

            self._holdings = {}
            if domestic_mom is not None and domestic_mom > 0:
                scored = []
                for ticker in self.risky_assets:
                    mom = self._trailing_return(ticker, context)
                    if mom is not None:
                        scored.append((ticker, mom))
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen = scored[:self.top_n]
                print(f"[GEM] Absolute momentum positive ({domestic_mom*100:.1f}%), "
                      f"ranked: {[(t, round(m*100,1)) for t,m in scored]}")
                if chosen:
                    weight = 1.0 / len(chosen)
                    for ticker, mom in chosen:
                        price = context.get_price(ticker)
                        if price and price > 0:
                            shares = int((weight * total_value) / price)
                            if shares > 0:
                                self._holdings[ticker] = {'shares': shares, 'entry_price': price, 'momentum': mom}
            else:
                print(f"[GEM] Absolute momentum negative ({domestic_mom*100 if domestic_mom is not None else float('nan'):.1f}%) -> CASH")

            self._last_rebalance = context.date
            if self._holdings:
                print(f"[GEM] Holding: {list(self._holdings.keys())}")

        positions = {}
        invested = 0.0
        for ticker, holding in self._holdings.items():
            price = context.get_price(ticker)
            if price and holding['shares'] > 0:
                positions[ticker] = Position(ticker=ticker, shares=float(holding['shares']), avg_cost=holding['entry_price'])
                invested += holding['shares'] * price

        return Portfolio(cash=total_value - invested, positions=positions)
