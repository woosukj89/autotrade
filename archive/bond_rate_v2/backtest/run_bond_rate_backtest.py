"""
Bond & Rate Adaptive vs. Regime Adaptive (MacroMom) Backtest Runner
=====================================================================

Rebuilt from spec — see strategies/bond_rate_strategy.py and
data/bond_rate_score.py for the "why this had to be rebuilt from notes,
not recovered" context.

Runs three strategies plus a SPY buy-and-hold benchmark over a lookback
window, using the tax-aware Backtest engine (backtest/backtest.py's
tax_rate param) at 25% to match the methodology documented in the project
memory notes ("after 25% tax, $100k start"):

1. Bond Rate Adaptive — new signal + new execution (steep allocation
   table, SH-weighted ETF baskets).
2. Regime Adaptive (MacroMom) — the original, unmodified live strategy:
   old signal + old execution.
3. Regime Adaptive (Improved Execution) — a control: the ORIGINAL MacroMom
   signal, but running through the SAME execution changes Bond Rate
   Adaptive got. Isolates how much of Bond Rate's edge is the new signal
   vs. just the execution change (which would help any signal).

Also saves each strategy's score history and reposition (reallocation)
log to CSV so the comparison chart can show the actual mechanism — when
each strategy crossed its threshold and what allocation it moved to —
not just the resulting equity curve.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest import Backtest
from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from strategies.bond_rate_strategy import BondRateAdaptiveStrategy, RegimeAdaptiveImprovedExecution


def run_backtest(name, strategy, bt, lookback_years, end_year):
    print(f"\n{'='*60}")
    print(f"Running: {name}  ({lookback_years}yr, ending {end_year})")
    print('=' * 60)
    result = bt.backtest(
        lookback_years=lookback_years,
        end_year=end_year,
        starting_fund=100000,
        strategy=strategy,
        time_period='M',  # Monthly rebalancing, matches original methodology
    )
    return result, strategy


def save_signal_logs(name, strategy, years, slug):
    """Save score history + reposition log so the chart can show the
    actual signal/allocation mechanism, not just the resulting equity."""
    score_log = getattr(strategy, '_score_history_log', [])
    reposition_log = getattr(strategy, '_reposition_log', [])

    if score_log:
        df = pd.DataFrame(score_log)
        fn = f"bond_rate_{slug}_{years}yr_scores.csv"
        df.to_csv(fn, index=False)
        print(f"Saved: {fn} ({len(df)} rows)")

    if reposition_log:
        df = pd.DataFrame(reposition_log)
        fn = f"bond_rate_{slug}_{years}yr_repositions.csv"
        df.to_csv(fn, index=False)
        print(f"Saved: {fn} ({len(df)} rows)")


def _benchmark_max_drawdown(result) -> float:
    values = result.benchmark_values
    if not values:
        return 0.0
    peak = np.maximum.accumulate(np.array(values))
    dd = (peak - np.array(values)) / np.where(peak > 0, peak, 1)
    return float(dd.max())


def print_comparison(results, label):
    print(f"\n{'='*76}")
    print(label)
    print('=' * 76)
    print(f"{'Strategy':<32} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Return':>10}")
    print('-' * 76)
    for name, (result, _strategy) in results.items():
        if result and result.snapshots:
            print(f"{name:<32} {result.cagr*100:>7.1f}% {result.max_drawdown*100:>7.1f}% "
                  f"{result.sharpe_ratio:>8.2f} {result.total_return*100:>9.1f}%")

    first = next((r for r, _s in results.values() if r), None)
    if first and first.benchmark_values:
        bench_dd = _benchmark_max_drawdown(first)
        print(f"{'SPY (no tax, benchmark)':<32} {first.benchmark_cagr*100:>7.1f}% "
              f"{bench_dd*100:>7.1f}% {'':>8} {first.benchmark_total_return*100:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Bond Rate Adaptive vs Regime Adaptive backtest')
    parser.add_argument('--years', type=int, default=10, help='Lookback years (default: 10)')
    parser.add_argument('--end-year', type=int, default=2025, help='End year (default: 2025)')
    parser.add_argument('--tax-rate', type=float, default=0.25, help='Realized-gains tax rate (default: 0.25)')
    parser.add_argument('--save-csv', action='store_true', help='Save per-strategy CSVs + signal logs')
    parser.add_argument('--skip-regime', action='store_true', help='Skip the RegimeAdaptive comparisons (faster smoke test)')
    args = parser.parse_args()

    db_path = 'data/fundamentals.sqlite'
    bt = Backtest(db_path=db_path, tax_rate=args.tax_rate)

    results = {}
    slugs = {}

    results['Bond Rate Adaptive'] = run_backtest(
        'Bond Rate Adaptive',
        BondRateAdaptiveStrategy(db_path=db_path),
        bt, args.years, args.end_year,
    )
    slugs['Bond Rate Adaptive'] = 'bond_rate_adaptive'

    if not args.skip_regime:
        results['Regime Adaptive (MacroMom)'] = run_backtest(
            'Regime Adaptive (MacroMom)',
            RegimeAdaptiveStrategy(db_path=db_path),
            bt, args.years, args.end_year,
        )
        slugs['Regime Adaptive (MacroMom)'] = 'regime_adaptive_macromom'

        results['Regime Adaptive (Improved Exec)'] = run_backtest(
            'Regime Adaptive (Improved Exec)',
            RegimeAdaptiveImprovedExecution(db_path=db_path),
            bt, args.years, args.end_year,
        )
        slugs['Regime Adaptive (Improved Exec)'] = 'regime_adaptive_improved_exec'

    print_comparison(
        results,
        f"{args.years}-YEAR COMPARISON (after {args.tax_rate*100:.0f}% tax, ending {args.end_year})"
    )

    if args.save_csv:
        print()
        for name, (result, strategy) in results.items():
            slug = slugs[name]
            if result and result.snapshots:
                df = result.portfolio_summary_over_time()
                fn = f"bond_rate_{slug}_{args.years}yr_results.csv"
                df.to_csv(fn, index=False)
                print(f"Saved: {fn}")
            save_signal_logs(name, strategy, args.years, slug)


if __name__ == '__main__':
    main()
