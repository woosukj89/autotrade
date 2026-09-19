"""
Parameter sweep for the new momentum-based stock-picker (momentum_strategy.py),
redesigning the RETURN source itself (per explicit direction) rather than
timing/hedging around the existing fundamentals-based picker. Full 20yr
history, monthly cadence for sweep speed (matching how earlier sweeps in
this session worked - monthly is fine for comparing configs; the winner
gets redone at daily cadence with full costs for the final honest number),
real slippage + regulatory fees, no options, pure long-only.
"""
import itertools
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
    # (lookback_days, rebalance_days, max_positions, trend_sma_days, vol_scale_weighting)
    (252, 30, 15, 200, False),   # baseline: classic 12-1 momentum, monthly rebal
    (252, 30, 15, 200, True),    # + inverse-vol weighting
    (126, 30, 15, 200, False),   # shorter (6mo) lookback
    (189, 30, 15, 200, False),   # 9mo lookback
    (252, 60, 15, 200, False),   # quarterly rebalance (less turnover/tax)
    (252, 90, 15, 200, False),   # rebalance every ~quarter+
    (252, 30, 10, 200, False),   # more concentrated
    (252, 30, 25, 200, False),   # more diversified
    (252, 30, 15, 150, False),   # looser trend filter
    (252, 30, 15, 100, False),   # much looser trend filter
    (252, 30, 15, 200, True),    # duplicate check placeholder replaced below
]
# de-dup and replace the placeholder with a genuinely new combo
VARIANTS[-1] = (126, 30, 15, 200, True)  # short lookback + vol-scaled


def main():
    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=0.25)

    rows = []
    seen = set()
    for lookback, rebal, maxpos, trend, volscale in VARIANTS:
        key = (lookback, rebal, maxpos, trend, volscale)
        if key in seen:
            continue
        seen.add(key)
        name = f"lb{lookback}_rb{rebal}_n{maxpos}_tr{trend}_vs{int(volscale)}"
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        strategy = MomentumStrategy(
            max_positions=maxpos, lookback_days=lookback, rebalance_days=rebal,
            trend_sma_days=trend, vol_scale_weighting=volscale,
        )
        result = bt.backtest(lookback_years=20, end_year=2025, starting_fund=100000,
                              strategy=strategy, time_period='M',
                              slippage_bps=5.0, regulatory_fees=True)
        if result and result.snapshots:
            rows.append((name, result.cagr, result.max_drawdown, result.sharpe_ratio, result.total_return))
            print(f"  -> CAGR={result.cagr*100:.1f}% MaxDD={result.max_drawdown*100:.1f}% Sharpe={result.sharpe_ratio:.2f}")

    print(f"\n{'='*90}\nMOMENTUM SWEEP (20yr monthly, after 25% tax + costs)\n{'='*90}")
    print(f"{'Variant':<40}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}")
    print('-' * 65)
    for name, cagr, dd, sharpe, ret in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<40}{cagr*100:>7.1f}%{dd*100:>7.1f}%{sharpe:>8.2f}")

    target_hit = [r for r in rows if r[1] > 0.251 and r[2] < 0.30]
    print()
    if target_hit:
        print(f"MEETS TARGET: {[r[0] for r in target_hit]}")
    else:
        print("No variant meets the CAGR>25.1% AND MaxDD<30% target.")

    out = {name: {'cagr': cagr, 'max_drawdown': dd, 'sharpe_ratio': sharpe, 'total_return': ret}
           for name, cagr, dd, sharpe, ret in rows}
    with open(os.path.join(RESULTS_DIR, 'momentum_sweep_20yr_monthly.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/momentum_sweep_20yr_monthly.json")


if __name__ == '__main__':
    main()
