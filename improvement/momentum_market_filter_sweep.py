"""
Adds the asset-class-level half of "dual momentum" (Antonacci) on top of
Iteration 24's best base config (9mo lookback, monthly rebal, 15
positions, 200d per-stock trend filter: 21.7% CAGR / 55.1% MaxDD
unfiltered) - a MARKET-level trend filter (SPY vs its own SMA) that goes
defensive across the WHOLE portfolio during a broad downturn, instead of
only filtering individual stocks (which still buys "best of a bad bunch"
during a real bear market).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from backtest import Backtest
from momentum_strategy import MomentumStrategy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

BASE = dict(max_positions=15, lookback_days=189, rebalance_days=30, trend_sma_days=200)

VARIANTS = [
    ('no market filter (baseline)', dict(market_filter=False)),
    ('market filter, SMA200, full cash', dict(market_filter=True, market_filter_sma_days=200, market_filter_defensive_weight=0.0)),
    ('market filter, SMA150, full cash', dict(market_filter=True, market_filter_sma_days=150, market_filter_defensive_weight=0.0)),
    ('market filter, SMA200, 30% kept invested', dict(market_filter=True, market_filter_sma_days=200, market_filter_defensive_weight=0.3)),
    ('market filter, SMA200, 50% kept invested', dict(market_filter=True, market_filter_sma_days=200, market_filter_defensive_weight=0.5)),
    ('market filter, SMA100, full cash', dict(market_filter=True, market_filter_sma_days=100, market_filter_defensive_weight=0.0)),
]


def main():
    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=0.25)
    rows = []
    for name, kwargs in VARIANTS:
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        strategy = MomentumStrategy(**{**BASE, **kwargs})
        result = bt.backtest(lookback_years=20, end_year=2025, starting_fund=100000,
                              strategy=strategy, time_period='M',
                              slippage_bps=5.0, regulatory_fees=True)
        if result and result.snapshots:
            rows.append((name, result.cagr, result.max_drawdown, result.sharpe_ratio))
            print(f"  -> CAGR={result.cagr*100:.1f}% MaxDD={result.max_drawdown*100:.1f}% Sharpe={result.sharpe_ratio:.2f}")

    print(f"\n{'='*90}\nMOMENTUM + MARKET FILTER SWEEP (20yr monthly, after 25% tax + costs)\n{'='*90}")
    print(f"{'Variant':<48}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}")
    print('-' * 72)
    for name, cagr, dd, sharpe in rows:
        print(f"{name:<48}{cagr*100:>7.1f}%{dd*100:>7.1f}%{sharpe:>8.2f}")

    target_hit = [r for r in rows if r[1] > 0.251 and r[2] < 0.30]
    print()
    if target_hit:
        print(f"MEETS TARGET: {[r[0] for r in target_hit]}")
    else:
        print("No variant meets the CAGR>25.1% AND MaxDD<30% target.")

    out = {name: {'cagr': cagr, 'max_drawdown': dd, 'sharpe_ratio': sharpe} for name, cagr, dd, sharpe in rows}
    with open(os.path.join(RESULTS_DIR, 'momentum_market_filter_sweep_20yr.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/momentum_market_filter_sweep_20yr.json")


if __name__ == '__main__':
    main()
