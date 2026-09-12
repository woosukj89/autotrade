"""
Bond & Rate Adaptive Strategy (v2)
===================================

Rebuilt from the design notes in Claude Code project memory for this repo
(see data/bond_rate_score.py's module docstring for the full context on why
this was rebuilt from spec rather than recovered). This subclasses
RegimeAdaptiveStrategy and overrides two things:

1. `_update_regime()` — uses the 7-component Bond & Rate bear score
   (data/bond_rate_score.py) instead of the MacroMom score (VIX/yield
   curve/breadth/price momentum), and additionally classifies WHICH kind of
   bear market conditions look like (inflation / recession / mixed).

2. `execute()` — when capital is allocated to the "defensive" side, this
   buys a bear-type-specific ETF basket (GLD/TLT/SH/SHY) instead of running
   the BearBetaStrategy stock-picker. Everything the memory notes' bear
   positioning research found: SH (inverse S&P 500) was the only instrument
   positive across all 4 historical bear types tested; TLT lost -30.8% in
   the 2022 inflation bear specifically; Bitcoin is confirmed NOT defensive
   (amplifies losses in every bear type) — so it's deliberately absent from
   every basket here.

Everything else (the aggressive high-beta sleeve, the allocation-weight
table mapping bear score -> HB/BB split) is inherited unchanged from
RegimeAdaptiveStrategy — the spec only called for changing the *signal* and
the *defensive instrument*, not the aggressive side or the weight table.

One deliberate deviation from the parent class, flagged here rather than
buried in a comment: RegimeAdaptiveStrategy's `_update_regime()` requires
"2 consecutive monthly readings >= 60" plus a price-momentum confirmation
before it's allowed to go defensive (see its _consecutive_high_scores /
_momentum_confirmed logic). That gate exists to stop the noisier MacroMom
score from whipsawing. Bond Rate v2's whole purpose is to react FASTER to
short bears that gate would specifically suppress, so this override applies
its threshold directly, in both directions, with no persistence
requirement. This is a testable hypothesis, not a certainty — if the
backtest shows this whipsaws too much (too many small reallocations
eating into returns via transaction costs/taxes), reintroducing a lighter
persistence check is the first thing to try.

NOTE ON FILE STABILITY: see data/bond_rate_score.py's module docstring —
this file has already disappeared from disk once during development with
no clear cause found in the repo's own automation. Commit this to a branch
once validated rather than leaving it as long-lived uncommitted WIP.
"""

import sys
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import pandas as pd

try:
    from strategies.strategy import Strategy, Portfolio, Position, ExecutionContext
    from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
except ImportError:
    from strategy import Strategy, Portfolio, Position, ExecutionContext
    from regime_adaptive_strategy import RegimeAdaptiveStrategy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
from bond_rate_score import (
    prefetch_bond_rate_fred_data,
    compute_bond_rate_score,
    classify_bear_type,
    get_bond_rate_risk_level,
    DEFENSIVE_BASKETS,
)


class BondRateAdaptiveStrategy(RegimeAdaptiveStrategy):
    """
    Regime-adaptive strategy driven by the Bond & Rate bear score instead of
    MacroMom, with a bear-type-specific ETF basket standing in for the
    bear-beta stock sleeve on the defensive side.
    """

    # TUNING ITERATION 2 (9/2): backtested walk-through of 2022 showed the
    # score was correctly DEFENSIVE (65-83) for the entire year, but the
    # inherited RegimeAdaptiveStrategy.ALLOCATION_THRESHOLDS table only ever
    # allocated 60/40 or 30/70 (HB/Def) even at max conviction - it never
    # committed below 30% high-beta. Checked the defensive basket's own 2022
    # return in isolation: +2.7% (GLD +0.8%, SHY -3.8%, SH +19.7%, blended
    # 40/40/20) - it was FINE. The portfolio still lost ~25% peak-to-trough
    # in 2022 almost entirely because 30-60% stayed in high-beta tech/semis
    # names that individually fell 40-50%+ that year. The old table's
    # "stay aggressive longer, minimal hedge" philosophy was calibrated for
    # MacroMom, a noisier signal not meant to be fully trusted. This score
    # has been backtest-validated as a real early-warning signal, so it
    # should be allowed to actually commit capital once triggered, rather
    # than treating every defensive signal as tentative.
    ALLOCATION_THRESHOLDS_BOND_RATE = [
        # (max_score, high_beta_weight, defensive_weight)
        (50, 1.00, 0.00),   # LOW: fully aggressive
        (60, 0.75, 0.25),   # WATCH: light hedge
        (70, 0.40, 0.60),   # DEFENSIVE (lower band): meaningful commitment
        (80, 0.15, 0.85),   # DEFENSIVE (higher band): strong commitment
        (100, 0.00, 1.00),  # DEFENSIVE (extreme): fully committed
    ]

    def _get_allocation_weights(self, bear_score: float) -> Tuple[float, float]:
        for max_score, hb_weight, def_weight in self.ALLOCATION_THRESHOLDS_BOND_RATE:
            if bear_score <= max_score:
                return (hb_weight, def_weight)
        return (0.00, 1.00)

    def __init__(self, *args, fred_api_key: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Prefetched ONCE for the whole backtest range — see
        # bond_rate_score.prefetch_bond_rate_fred_data's docstring for why
        # this can't go through ExecutionContext like the parent's SPY/VIX
        # data does (ExecutionContext only serves yfinance ticker history,
        # not FRED series).
        print("[BondRateAdaptive] Prefetching FRED data for full backtest range...")
        self._fred_history: Dict[str, pd.Series] = prefetch_bond_rate_fred_data(
            fred_api_key=fred_api_key
        )
        self._current_bear_type: str = "mixed"

    def _slice_as_of(self, context: ExecutionContext) -> Dict[str, pd.Series]:
        """Point-in-time slice of the prefetched FRED history — no data
        after context.date is visible to the score, to avoid look-ahead
        bias in the backtest."""
        as_of = pd.Timestamp(context.date)
        return {
            key: series[series.index <= as_of]
            for key, series in self._fred_history.items()
            if len(series[series.index <= as_of]) > 0
        }

    def _update_regime(self, context: ExecutionContext) -> None:
        """Bond & Rate v2 regime update — see module docstring for how this
        differs from the parent's MacroMom-based version."""
        inputs = self._slice_as_of(context)

        if not inputs:
            print(f"[BondRateAdaptive] {context.date.strftime('%Y-%m-%d')}: no FRED data yet, using default allocation")
            return

        bear_score, component_scores = compute_bond_rate_score(inputs)
        bear_type = classify_bear_type(component_scores)
        risk_level = get_bond_rate_risk_level(bear_score)

        self._current_bear_score = bear_score
        self._current_bear_type = bear_type
        self._bear_score_history.append(bear_score)
        if len(self._bear_score_history) > 52:
            self._bear_score_history = self._bear_score_history[-52:]
        self._score_history_log.append({'date': context.date, 'score': bear_score})

        # Direct threshold application — no persistence/momentum gate.
        # See module docstring for why this intentionally differs from the parent.
        new_allocation = self._get_allocation_weights(bear_score)
        old_allocation = self._current_allocation
        allocation_change = abs(new_allocation[0] - old_allocation[0])

        date_str = context.date.strftime('%Y-%m-%d')
        n_avail = len(component_scores)
        print(f"[BondRate {date_str}] Score={bear_score:.1f} ({n_avail}/7 components) Level={risk_level:10s} "
              f"Type={bear_type:10s} Alloc: {new_allocation[0]*100:.0f}%/{new_allocation[1]*100:.0f}% (HB/Def)")

        if allocation_change >= self.min_realloc_change:
            direction = '-> DEFENSIVE' if new_allocation[0] < old_allocation[0] else '-> AGGRESSIVE'
            print(f"  -> REALLOCATION {direction}: {old_allocation[0]*100:.0f}% -> {new_allocation[0]*100:.0f}% High Beta "
                  f"(defensive basket: {DEFENSIVE_BASKETS[bear_type]})")
            self._current_allocation = new_allocation
            self._reposition_log.append({
                'date':     context.date,
                'trigger':  f'bond-rate-{bear_type}' if bear_score > 55 else 'bond-rate-release',
                'from_hb':  old_allocation[0],
                'from_bb':  old_allocation[1],
                'to_hb':    new_allocation[0],
                'to_bb':    new_allocation[1],
                'score':    bear_score,
            })

        self._last_regime_check = context.date

    def _build_etf_basket_portfolio(self, capital: float, context: ExecutionContext) -> Portfolio:
        """Buy the current bear-type's defensive ETF basket with `capital`."""
        basket = DEFENSIVE_BASKETS[self._current_bear_type]
        positions: Dict[str, Position] = {}
        remaining_cash = capital

        for ticker, weight in basket.items():
            price = context.get_price(ticker)
            if not price or price <= 0:
                continue
            target_value = capital * weight
            shares = int(target_value // price)
            if shares > 0:
                positions[ticker] = Position(ticker=ticker, shares=float(shares), avg_cost=price)
                remaining_cash -= shares * price

        return Portfolio(cash=max(0.0, remaining_cash), positions=positions)

    def execute(self, context: ExecutionContext) -> Portfolio:
        """Same capital-split structure as RegimeAdaptiveStrategy.execute(),
        but the defensive slice buys the bear-type ETF basket instead of
        running BearBetaStrategy."""
        if self._should_check_regime(context.date):
            self._update_regime(context)

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        hb_weight, bb_weight = self._current_allocation
        hb_capital = total_value * hb_weight
        bb_capital = total_value * bb_weight

        print(f"[BondRateAdaptive] Total: ${total_value:,.0f}")
        print(f"  High Beta: ${hb_capital:,.0f} ({hb_weight*100:.0f}%)")
        print(f"  Defensive ({self._current_bear_type}): ${bb_capital:,.0f} ({bb_weight*100:.0f}%)")

        all_positions: Dict[str, Position] = {}
        remaining_cash = total_value

        if hb_weight > 0.05:
            hb_portfolio = Portfolio(cash=hb_capital, positions={})
            hb_context = ExecutionContext(
                date=context.date,
                portfolio=hb_portfolio,
                get_price_fn=context._get_price,
                get_historical_fn=context._get_historical,
                get_fundamentals_fn=context._get_fundamentals,
            )
            hb_result = self.high_beta_strategy.execute(hb_context)
            for ticker, pos in hb_result.positions.items():
                all_positions[ticker] = pos
            remaining_cash -= (hb_capital - hb_result.cash)

        if bb_weight > 0.05:
            bb_result = self._build_etf_basket_portfolio(bb_capital, context)
            for ticker, pos in bb_result.positions.items():
                if ticker in all_positions:
                    existing = all_positions[ticker]
                    total_shares = existing.shares + pos.shares
                    avg_cost = (existing.shares * existing.avg_cost + pos.shares * pos.avg_cost) / total_shares
                    all_positions[ticker] = Position(ticker=ticker, shares=total_shares, avg_cost=avg_cost)
                else:
                    all_positions[ticker] = pos
            remaining_cash -= (bb_capital - bb_result.cash)

        remaining_cash = max(0, remaining_cash)
        print(f"[BondRateAdaptive] Combined portfolio: {len(all_positions)} positions")

        return Portfolio(cash=remaining_cash, positions=all_positions)


class RegimeAdaptiveImprovedExecution(RegimeAdaptiveStrategy):
    """
    Control variant for isolating signal quality from execution quality.

    Every comparison so far ran BondRateAdaptiveStrategy (new signal + new
    execution: steeper allocation table, SH-weighted ETF baskets) against
    plain RegimeAdaptiveStrategy (old MacroMom signal + old execution: the
    original table that never commits past 70% defensive, BearBetaStrategy
    stocks instead of ETFs). That's confounded — some of Bond Rate's edge
    could just be the execution change, which would help ANY signal, not
    something specific to the new score.

    This strategy uses the ORIGINAL MacroMom signal (`_update_regime` fully
    inherited, unchanged — including its persistence/momentum gate) but
    swaps in the exact same execution BondRateAdaptiveStrategy uses: the
    same steep allocation table, and the same SH-weighted "Full Defense"
    basket for the defensive sleeve (MacroMom has no bear-type
    classification, so there's no inflation/recession split to route on —
    every defensive period uses the all-weather basket).
    """

    def _get_allocation_weights(self, bear_score: float) -> Tuple[float, float]:
        for max_score, hb_weight, def_weight in BondRateAdaptiveStrategy.ALLOCATION_THRESHOLDS_BOND_RATE:
            if bear_score <= max_score:
                return (hb_weight, def_weight)
        return (0.00, 1.00)

    def _build_etf_basket_portfolio(self, capital: float, context: ExecutionContext) -> Portfolio:
        basket = DEFENSIVE_BASKETS["mixed"]  # no bear-type signal available from MacroMom
        positions: Dict[str, Position] = {}
        remaining_cash = capital
        for ticker, weight in basket.items():
            price = context.get_price(ticker)
            if not price or price <= 0:
                continue
            target_value = capital * weight
            shares = int(target_value // price)
            if shares > 0:
                positions[ticker] = Position(ticker=ticker, shares=float(shares), avg_cost=price)
                remaining_cash -= shares * price
        return Portfolio(cash=max(0.0, remaining_cash), positions=positions)

    def execute(self, context: ExecutionContext) -> Portfolio:
        """Identical structure to RegimeAdaptiveStrategy.execute(), except
        the defensive slice buys the SH-weighted ETF basket instead of
        running BearBetaStrategy. _update_regime (the signal) is untouched."""
        if self._should_check_regime(context.date):
            self._update_regime(context)

        portfolio = context.portfolio
        total_value = portfolio.cash
        for ticker, pos in portfolio.positions.items():
            price = context.get_price(ticker)
            total_value += pos.shares * (price if price else pos.avg_cost)

        hb_weight, bb_weight = self._current_allocation
        hb_capital = total_value * hb_weight
        bb_capital = total_value * bb_weight

        print(f"[RegimeImprovedExec] Total: ${total_value:,.0f}")
        print(f"  High Beta: ${hb_capital:,.0f} ({hb_weight*100:.0f}%)")
        print(f"  Defensive: ${bb_capital:,.0f} ({bb_weight*100:.0f}%)")

        all_positions: Dict[str, Position] = {}
        remaining_cash = total_value

        if hb_weight > 0.05:
            hb_portfolio = Portfolio(cash=hb_capital, positions={})
            hb_context = ExecutionContext(
                date=context.date,
                portfolio=hb_portfolio,
                get_price_fn=context._get_price,
                get_historical_fn=context._get_historical,
                get_fundamentals_fn=context._get_fundamentals,
            )
            hb_result = self.high_beta_strategy.execute(hb_context)
            for ticker, pos in hb_result.positions.items():
                all_positions[ticker] = pos
            remaining_cash -= (hb_capital - hb_result.cash)

        if bb_weight > 0.05:
            bb_result = self._build_etf_basket_portfolio(bb_capital, context)
            for ticker, pos in bb_result.positions.items():
                if ticker in all_positions:
                    existing = all_positions[ticker]
                    total_shares = existing.shares + pos.shares
                    avg_cost = (existing.shares * existing.avg_cost + pos.shares * pos.avg_cost) / total_shares
                    all_positions[ticker] = Position(ticker=ticker, shares=total_shares, avg_cost=avg_cost)
                else:
                    all_positions[ticker] = pos
            remaining_cash -= (bb_capital - bb_result.cash)

        remaining_cash = max(0, remaining_cash)
        print(f"[RegimeImprovedExec] Combined portfolio: {len(all_positions)} positions")

        return Portfolio(cash=remaining_cash, positions=all_positions)


if __name__ == "__main__":
    print("BondRateAdaptiveStrategy rebuilt — run via backtest/run_bond_rate_backtest.py "
          "for a real comparison against RegimeAdaptiveStrategy.")
