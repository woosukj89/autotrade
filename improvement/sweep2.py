"""
Second-stage guided sweep: narrows to the region the first sweep + manual
probing pointed at (see PROGRESS.md). Key findings baked into these ranges:
  - entry_buffer_pct stays loose (0.01-0.02) - tight buffers cost coverage
    across the board, first sweep never preferred anything tighter.
  - credit_z_threshold raised into 1.0-2.5 (was 0.25-2.0) - z-score DOES
    discriminate real bears from false alarms here (2022 max z=3.45 vs
    2011's 2.13, 2010's ~1.0-1.2), unlike VIX. First sweep's preference for
    0.25 was gaming coverage at FPR's expense, not a real optimum.
  - trend_confirm_days pushed lower (3-15) so the trend-only path can carry
    2022 coverage on days credit z is below threshold, without waiting the
    original 20-25 days.
  - fast_panic_confirm_days kept 1-3 (persistence gate from v4, real but
    modest effect).
"""
import random

from sweep import run_sweep_with_ranges, print_top, all_targets_met, PARAM_RANGES

GUIDED_RANGES = dict(PARAM_RANGES)
GUIDED_RANGES.update({
    'entry_buffer_pct': [0.01, 0.015, 0.02],
    'exit_buffer_pct': [0.0, 0.005, 0.01],
    'credit_z_threshold': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    'credit_z_lookback': [189, 252],
    'trend_confirm_days': [3, 5, 7, 10, 12, 15],
    'sma_window': [100, 150, 200],
    'fast_panic_confirm_days': [1, 2, 3],
    'combo_confirm_days': [1, 2],
    'drawdown_threshold': [-0.08, -0.10, -0.12, -0.14],
    'min_defensive_days': [0, 3, 5, 10, 15, 20],
    'panic_cooldown_days': [2, 3, 5, 7],
    'require_credit_calm_to_exit': [False, True],
})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=3)
    args = parser.parse_args()

    print(f'Guided sweep: {args.n} combos (seed={args.seed})...')
    results = run_sweep_with_ranges(args.n, args.seed, GUIDED_RANGES)
    print_top(results, k=20)
    passing = [r for r in results if all_targets_met(r[2])]
    print(f"\n{len(passing)} / {len(results)} combos met ALL 3 targets simultaneously")
