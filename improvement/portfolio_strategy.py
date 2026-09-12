"""
Step 3 of the reframed process (SPEC.md §0): wire a binary classifier
(classifier.py) into an actual Strategy for backtest/backtest.py's engine.
100% aggressive (HighBetaGrowthStrategy) or 100% defensive (SH-heavy
basket) - no partial allocation, per the original direction.

The classifier itself (`classify()`) needs its full macro/breadth history
upfront (same reason BondRateAdaptiveStrategy in archive/bond_rate_v2
prefetched FRED data in __init__ - ExecutionContext only serves yfinance
ticker history, not FRED/breadth series). So the defensive/aggressive
call for the ENTIRE backtest window is precomputed once in __init__, then
looked up by date during execute() - not recomputed incrementally. This
does mean the live classifier signal isn't point-in-time-restricted to
"data available as of that day" in the same strict sense the trend/credit
signals elsewhere in this repo are (200-day SMAs, drawdowns, and credit
z-scores are all backward-looking only, so this is a minor practical
simplification, not a look-ahead cheat on the underlying signals - but
stated plainly since it's a real methodological choice).
"""
import os
import sys
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
from strategy import Strategy, Portfolio, Position, ExecutionContext
from high_beta_strategy import HighBetaGrowthStrategy

sys.path.insert(0, os.path.dirname(__file__))
from data_feed import load_all
from classifier import classify, ClassifierParams


DEFAULT_DEFENSIVE_BASKET = {'SH': 1.0}  # binary + simplest, best single historical performer


class BinaryClassifierStrategy(Strategy):
    def __init__(self, classifier_params: ClassifierParams, db_path: str = "data/fundamentals.sqlite",
                 defensive_basket: Optional[Dict[str, float]] = None, data: Optional[dict] = None,
                 max_positions: int = 15):
        self.high_beta_strategy = HighBetaGrowthStrategy(db_path=db_path, max_positions=max_positions)
        self.defensive_basket = defensive_basket or DEFAULT_DEFENSIVE_BASKET

        data = data if data is not None else load_all()
        self._defensive_series, self._transitions = classify(data, classifier_params)
        self._defensive_index = self._defensive_series.index

    def _is_defensive(self, date) -> bool:
        ts = pd.Timestamp(date)
        pos = self._defensive_index.searchsorted(ts, side='right') - 1
        if pos < 0:
            return False
        return bool(self._defensive_series.iloc[pos])

    def _build_defensive_portfolio(self, capital: float, context: ExecutionContext) -> Portfolio:
        positions: Dict[str, Position] = {}
        remaining_cash = capital
        for ticker, weight in self.defensive_basket.items():
            price = context.get_price(ticker)
            if not price or price <= 0:
                continue
            shares = int((capital * weight) // price)
            if shares > 0:
                positions[ticker] = Position(ticker=ticker, shares=float(shares), avg_cost=price)
                remaining_cash -= shares * price
        return Portfolio(cash=max(0.0, remaining_cash), positions=positions)

    def execute(self, context: ExecutionContext) -> Portfolio:
        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        if self._is_defensive(context.date):
            result = self._build_defensive_portfolio(total_value, context)
            print(f"[BinaryClassifier] {context.date.strftime('%Y-%m-%d')} DEFENSIVE  ${total_value:,.0f}")
            return result
        else:
            hb_context = ExecutionContext(
                date=context.date,
                portfolio=Portfolio(cash=total_value, positions={}),
                get_price_fn=context._get_price,
                get_historical_fn=context._get_historical,
                get_fundamentals_fn=context._get_fundamentals,
            )
            result = self.high_beta_strategy.execute(hb_context)
            print(f"[BinaryClassifier] {context.date.strftime('%Y-%m-%d')} AGGRESSIVE  ${total_value:,.0f}")
            return result
