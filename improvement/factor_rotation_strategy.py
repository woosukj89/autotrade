"""
Idea E: factor rotation. Always 100% invested in equities - never holds
cash or an inverse/SH position (that's idea A/binary, already tested) -
but switches WHICH stock sleeve it holds based on the same calibrated
regime classifier used all session: HighBetaGrowthStrategy (aggressive,
high-beta tech/semis tilt) when not-defensive, QualityDefensiveStrategy
(low-beta, quality, staples/healthcare/utilities tilt) when defensive.

Directly tests whether reducing FACTOR exposure (which stocks) controls
drawdown better than reducing MARKET exposure (how much is invested,
idea A) or buying insurance (idea B) - it never gives up market
participation, unlike both of those.
"""
import os
import sys
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
from strategy import Strategy, Portfolio, ExecutionContext
from high_beta_strategy import HighBetaGrowthStrategy

sys.path.insert(0, os.path.dirname(__file__))
from quality_strategy import QualityDefensiveStrategy
from data_feed import load_all
from classifier import classify, ClassifierParams


class FactorRotationStrategy(Strategy):
    def __init__(self, classifier_params: ClassifierParams, db_path: str = "data/fundamentals.sqlite",
                 data: Optional[dict] = None, max_positions: int = 15):
        self.aggressive = HighBetaGrowthStrategy(db_path=db_path, max_positions=max_positions)
        self.defensive = QualityDefensiveStrategy(db_path=db_path, max_positions=max_positions)

        data = data if data is not None else load_all()
        self._defensive_series, self._transitions = classify(data, classifier_params)
        self._defensive_index = self._defensive_series.index

    def _is_defensive(self, date) -> bool:
        ts = pd.Timestamp(date)
        pos = self._defensive_index.searchsorted(ts, side='right') - 1
        if pos < 0:
            return False
        return bool(self._defensive_series.iloc[pos])

    def execute(self, context: ExecutionContext) -> Portfolio:
        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        sleeve = self.defensive if self._is_defensive(context.date) else self.aggressive
        label = 'DEFENSIVE(quality)' if sleeve is self.defensive else 'AGGRESSIVE(highbeta)'

        sleeve_context = ExecutionContext(
            date=context.date, portfolio=Portfolio(cash=total_value, positions={}),
            get_price_fn=context._get_price, get_historical_fn=context._get_historical,
            get_fundamentals_fn=context._get_fundamentals,
        )
        result = sleeve.execute(sleeve_context)
        print(f"[FactorRotation] {context.date.strftime('%Y-%m-%d')} {label}  ${total_value:,.0f}")
        return result
