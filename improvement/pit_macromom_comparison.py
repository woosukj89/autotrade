"""
Directly answers: does the LIVE MacroMom strategy (RegimeAdaptiveStrategy)
already meet CAGR>25%/MaxDD<30% once tested honestly - point-in-time SEC
EDGAR fundamentals, daily cadence matching the live cron, real slippage +
regulatory fees, same methodology as Iteration 22's redo? If so, per
explicit direction, no further strategy search is needed - the existing
live strategy already clears the bar and the goal is met.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from backtest import Backtest
from pit_strategies import PitRegimeAdaptiveStrategy, PitFundamentalsStore

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
SLIPPAGE_BPS = 5.0


def main():
    print('Loading point-in-time fundamentals store...')
    pit_store = PitFundamentalsStore()
    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=0.25)

    print(f"\n{'='*60}\nRunning: PitRegimeAdaptive (MacroMom, point-in-time) 20yr daily\n{'='*60}")
    result = bt.backtest(
        lookback_years=20, end_year=2025, starting_fund=100000,
        strategy=PitRegimeAdaptiveStrategy(pit_store=pit_store, db_path=db_path),
        time_period='d', slippage_bps=SLIPPAGE_BPS, regulatory_fees=True)

    print(f"\n{'='*80}\nPIT MACROMOM (LIVE STRATEGY), 20yr daily, point-in-time + real costs\n{'='*80}")
    print(f"CAGR: {result.cagr*100:.1f}%   MaxDD: {result.max_drawdown*100:.1f}%   Sharpe: {result.sharpe_ratio:.2f}")
    print(f"Slippage: ${result.slippage_cost:,.0f}   RegFees: ${result.regulatory_fees:,.0f}   Tax: ${result.tax_paid:,.0f}")

    if result.cagr > 0.251 and result.max_drawdown < 0.30:
        print("\nMEETS TARGET (CAGR>25.1% AND MaxDD<30%)")
    else:
        print("\nDoes NOT meet the CAGR>25.1% AND MaxDD<30% target.")

    out_path = os.path.join(RESULTS_DIR, 'pit_macromom_full_20yr.json')
    with open(out_path, 'w') as f:
        json.dump({'cagr': result.cagr, 'max_drawdown': result.max_drawdown,
                   'sharpe_ratio': result.sharpe_ratio, 'slippage_cost': result.slippage_cost,
                   'regulatory_fees': result.regulatory_fees, 'tax_paid': result.tax_paid}, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
