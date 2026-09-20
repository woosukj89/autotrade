"""
Fine-tuning around the best market-filter result found (SMA200, 30% kept
invested: 22.1% CAGR / 27.6% MaxDD / 5.39 Sharpe - the closest anything
has come to CAGR>25.1%/MaxDD<30% all session, and mechanistically sound
rather than noise-driven). Sweeps defensive_weight more finely and tries
combining with the other strong base configs found in Iteration 24
(6mo lookback, 10-position concentration).
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

VARIANTS = [
    ('lb189_n15, defw=0.15', dict(max_positions=15, lookback_days=189, market_filter=True, market_filter_defensive_weight=0.15)),
    ('lb189_n15, defw=0.20', dict(max_positions=15, lookback_days=189, market_filter=True, market_filter_defensive_weight=0.20)),
    ('lb189_n15, defw=0.25', dict(max_positions=15, lookback_days=189, market_filter=True, market_filter_defensive_weight=0.25)),
    ('lb189_n15, defw=0.35', dict(max_positions=15, lookback_days=189, market_filter=True, market_filter_defensive_weight=0.35)),
    ('lb189_n15, defw=0.40', dict(max_positions=15, lookback_days=189, market_filter=True, market_filter_defensive_weight=0.40)),
    ('lb126_n15, defw=0.30', dict(max_positions=15, lookback_days=126, market_filter=True, market_filter_defensive_weight=0.30)),
    ('lb189_n10, defw=0.30', dict(max_positions=10, lookback_days=189, market_filter=True, market_filter_defensive_weight=0.30)),
    ('lb189_n20, defw=0.30', dict(max_positions=20, lookback_days=189, market_filter=True, market_filter_defensive_weight=0.30)),
]


def main():
    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=0.25)
    rows = []
    for name, kwargs in VARIANTS:
        kwargs.setdefault('rebalance_days', 30)
        kwargs.setdefault('trend_sma_days', 200)
        kwargs.setdefault('market_filter_sma_days', 200)
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        strategy = MomentumStrategy(**kwargs)
        result = bt.backtest(lookback_years=20, end_year=2025, starting_fund=100000,
                              strategy=strategy, time_period='M',
                              slippage_bps=5.0, regulatory_fees=True)
        if result and result.snapshots:
            rows.append((name, result.cagr, result.max_drawdown, result.sharpe_ratio))
            print(f"  -> CAGR={result.cagr*100:.1f}% MaxDD={result.max_drawdown*100:.1f}% Sharpe={result.sharpe_ratio:.2f}")

    print(f"\n{'='*90}\nFINE-TUNE SWEEP (20yr monthly, after 25% tax + costs)\n{'='*90}")
    print(f"{'Variant':<32}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}")
    print('-' * 56)
    for name, cagr, dd, sharpe in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<32}{cagr*100:>7.1f}%{dd*100:>7.1f}%{sharpe:>8.2f}")

    target_hit = [r for r in rows if r[1] > 0.251 and r[2] < 0.30]
    print()
    if target_hit:
        print(f"MEETS TARGET: {[r[0] for r in target_hit]}")
    else:
        print("No variant meets the CAGR>25.1% AND MaxDD<30% target.")

    out = {name: {'cagr': cagr, 'max_drawdown': dd, 'sharpe_ratio': sharpe} for name, cagr, dd, sharpe in rows}
    with open(os.path.join(RESULTS_DIR, 'momentum_finetune_sweep_20yr.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/momentum_finetune_sweep_20yr.json")


if __name__ == '__main__':
    main()
