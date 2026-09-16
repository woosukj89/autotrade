"""
Idea E refinement: the calibrated classifiers (multi_def_summary.json) were
tuned to minimize false-positive rate for a BINARY strategy, where a false
positive costs 100% of that period's upside (full exit to cash/SH). For
factor rotation, a false positive only costs the gap between the aggressive
and defensive SLEEVE's returns (both are still equities, both still
compound) - much cheaper. So a more trigger-happy classifier (lower
vote_threshold, faster confirmation) that would have been a bad idea for
the binary strategy may be a net win here. Hand-tests a few variants
against alltime_15's calibrated base (the best classifier found so far)
through the real 20yr backtest.
"""
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from backtest import Backtest
from classifier import ClassifierParams
from factor_rotation_strategy import FactorRotationStrategy
from data_feed import load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    with open(os.path.join(RESULTS_DIR, 'multi_def_summary.json')) as f:
        summary = json.load(f)
    base = ClassifierParams(**summary['alltime_15']['best_params'])

    variants = {
        'base (alltime_15, reused)': base,
        'vote_threshold=1': replace(base, vote_threshold=1),
        'combo_confirm_days=1,panic_cooldown=1': replace(base, combo_confirm_days=1, panic_cooldown_days=1),
        'vote_threshold=1,combo_confirm=1': replace(base, vote_threshold=1, combo_confirm_days=1, panic_cooldown_days=1),
        'trend_confirm_days=10': replace(base, trend_confirm_days=10),
        'vote_threshold=1,trend_confirm=10,panic_cooldown=1': replace(
            base, vote_threshold=1, trend_confirm_days=10, panic_cooldown_days=1, combo_confirm_days=1),
        'require_credit_calm_to_exit=False': replace(base, require_credit_calm_to_exit=False),
    }

    print('Loading shared classifier data once...')
    clf_data = load_all()
    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=0.25)

    rows = []
    for name, params in variants.items():
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        strategy = FactorRotationStrategy(params, db_path=db_path, data=clf_data, max_positions=15)
        result = bt.backtest(lookback_years=20, end_year=2025, starting_fund=100000,
                              strategy=strategy, time_period='M')
        if result and result.snapshots:
            rows.append((name, result.cagr, result.max_drawdown, result.sharpe_ratio, result.total_return))

    print(f"\n{'='*90}\nFACTOR ROTATION TUNING (20yr, after 25% tax)\n{'='*90}")
    print(f"{'Variant':<48}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}")
    print('-' * 80)
    for name, cagr, dd, sharpe, ret in rows:
        print(f"{name:<48}{cagr*100:>7.1f}%{dd*100:>7.1f}%{sharpe:>8.2f}")

    target_hit = [r for r in rows if r[1] > 0.251 and r[2] < 0.30]
    print()
    if target_hit:
        print(f"MEETS TARGET (CAGR>25.1% AND MaxDD<30%): {[r[0] for r in target_hit]}")
    else:
        print("No variant meets the CAGR>25.1% AND MaxDD<30% target yet.")


if __name__ == '__main__':
    main()
