"""
Third-stage sweep, for the v6 voting architecture (classify() now requires
>=vote_threshold of {fast_panic, trend_break, credit_stress} to agree,
replacing the "any 1 of 3 paths" OR logic that plateaued at ~10-13% FPR
across 45,000+ combos in sweep.py/sweep2.py - see PROGRESS.md).

Ranges carry forward what sweep2 established (loose entry_buffer_pct,
credit_z_threshold 1.0-2.5, require_credit_calm_to_exit) and add the new
voting-specific knobs (vote_threshold, extreme_drawdown_threshold,
combo_confirm_days as the vote's own persistence gate).
"""
from sweep import run_sweep_with_ranges, print_top, all_targets_met, PARAM_RANGES

GUIDED_RANGES = dict(PARAM_RANGES)
GUIDED_RANGES.update({
    'entry_buffer_pct': [0.01, 0.015, 0.02, 0.025],
    'exit_buffer_pct': [0.0, 0.005, 0.01],
    'credit_z_threshold': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    'credit_z_lookback': [189, 252],
    'trend_confirm_days': [10, 15, 20, 25, 30, 40],  # pure-fallback path, keep it slow/rare
    'sma_window': [100, 150, 200],
    'combo_confirm_days': [1, 2, 3, 4, 5],
    'drawdown_threshold': [-0.08, -0.10, -0.12, -0.14],
    'min_defensive_days': [0, 3, 5, 10, 15, 20],
    'panic_cooldown_days': [2, 3, 5, 7],
    'require_credit_calm_to_exit': [False, True],
    'extreme_drawdown_threshold': [-0.16, -0.18, -0.20, -0.22],
    'vote_threshold': [2, 3],
})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    print(f'Voting-architecture sweep: {args.n} combos (seed={args.seed})...')
    results = run_sweep_with_ranges(args.n, args.seed, GUIDED_RANGES)
    print_top(results, k=20)
    passing = [r for r in results if all_targets_met(r[2])]
    print(f"\n{len(passing)} / {len(results)} combos met ALL 3 targets simultaneously")
