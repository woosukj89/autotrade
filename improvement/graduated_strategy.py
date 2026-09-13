"""
Idea A: graduated (not binary) allocation using this session's calibrated
4-signal vote (fast_panic, trend_break, credit_stress, breadth_stress).

Why: Step 3's finding was that binary 100%/0% gives up ALL upside on every
defensive day, including false positives and bear-market rallies, and lost
to just holding the stock-picker with no timing at all (see
test_diversification.py: 26.8% CAGR / 64.7% MaxDD doing NOTHING but
rebalancing the picker). A graduated exposure curve keeps most of that
upside during ambiguous (low-vote) periods while still meaningfully
de-risking when multiple independent signals agree (high-vote periods) -
directly targeting the flaw without giving up the strong baseline.

Exposure curve (tunable): more votes agreeing = more de-risked. An
`extreme_panic` day (rare, deep N-day drawdown) overrides straight to the
floor regardless of vote count - the safety valve for a 2020-style shock
a graduated signal might not otherwise react to fast enough.
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
from classifier import compute_votes, ClassifierParams

DEFAULT_EXPOSURE_MAP = {0: 1.00, 1: 0.80, 2: 0.55, 3: 0.30, 4: 0.15}
DEFAULT_EXTREME_EXPOSURE = 0.10
DEFAULT_DEFENSIVE_BASKET = {'SH': 1.0}


class GraduatedClassifierStrategy(Strategy):
    def __init__(self, classifier_params: ClassifierParams, db_path: str = "data/fundamentals.sqlite",
                 defensive_basket: Optional[Dict[str, float]] = None, data: Optional[dict] = None,
                 max_positions: int = 15, max_sector_weight: float = 0.50, max_position_weight: float = 0.15,
                 exposure_map: Optional[Dict[int, float]] = None, extreme_exposure: float = DEFAULT_EXTREME_EXPOSURE):
        self.high_beta_strategy = HighBetaGrowthStrategy(
            db_path=db_path, max_positions=max_positions,
            max_sector_weight=max_sector_weight, max_position_weight=max_position_weight)
        self.defensive_basket = defensive_basket or DEFAULT_DEFENSIVE_BASKET
        self.exposure_map = exposure_map or DEFAULT_EXPOSURE_MAP
        self.extreme_exposure = extreme_exposure

        data = data if data is not None else load_all()
        self._votes, self._extreme = compute_votes(data, classifier_params)
        self._index = self._votes.index

    def _exposure_for(self, date) -> float:
        ts = pd.Timestamp(date)
        pos = self._index.searchsorted(ts, side='right') - 1
        if pos < 0:
            return 1.0
        if bool(self._extreme.iloc[pos]):
            return self.extreme_exposure
        votes = int(self._votes.iloc[pos])
        return self.exposure_map.get(votes, min(self.exposure_map.values()))

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

        exposure = self._exposure_for(context.date)
        hb_capital = total_value * exposure
        def_capital = total_value * (1 - exposure)

        print(f"[Graduated] {context.date.strftime('%Y-%m-%d')} exposure={exposure*100:.0f}%  ${total_value:,.0f}")

        all_positions: Dict[str, Position] = {}
        remaining_cash = total_value

        if exposure > 0.02:
            hb_context = ExecutionContext(
                date=context.date, portfolio=Portfolio(cash=hb_capital, positions={}),
                get_price_fn=context._get_price, get_historical_fn=context._get_historical,
                get_fundamentals_fn=context._get_fundamentals,
            )
            hb_result = self.high_beta_strategy.execute(hb_context)
            for ticker, pos in hb_result.positions.items():
                all_positions[ticker] = pos
            remaining_cash -= (hb_capital - hb_result.cash)

        if (1 - exposure) > 0.02:
            def_result = self._build_defensive_portfolio(def_capital, context)
            for ticker, pos in def_result.positions.items():
                if ticker in all_positions:
                    existing = all_positions[ticker]
                    total_shares = existing.shares + pos.shares
                    avg_cost = (existing.shares * existing.avg_cost + pos.shares * pos.avg_cost) / total_shares
                    all_positions[ticker] = Position(ticker=ticker, shares=total_shares, avg_cost=avg_cost)
                else:
                    all_positions[ticker] = pos
            remaining_cash -= (def_capital - def_result.cash)

        remaining_cash = max(0.0, remaining_cash)
        return Portfolio(cash=remaining_cash, positions=all_positions)
