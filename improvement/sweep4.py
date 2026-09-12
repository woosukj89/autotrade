"""
Fourth-stage sweep: adds breadth_stress as a 4th vote input (see breadth.py,
signals.breadth_stress_signal). vote_threshold now spans {2, 3} out of 4
possible signals (was {2, 3} out of 3 - 3-of-4 is a meaningfully different,
stricter bar than 2-of-3, worth exploring properly rather than just
reusing the 3-signal sweep's threshold choices).
"""
from sweep import run_sweep_with_ranges, print_top, all_targets_met, PARAM_RANGES
from sweep3 import GUIDED_RANGES as V6_RANGES

GUIDED_RANGES = dict(V6_RANGES)
GUIDED_RANGES.update({
    'vote_threshold': [2, 3],
    'breadth_threshold': [25, 30, 35, 40, 45, 50, 55],
})

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()

    print(f'4-signal voting sweep: {args.n} combos (seed={args.seed})...')
    results = run_sweep_with_ranges(args.n, args.seed, GUIDED_RANGES)
    print_top(results, k=20)
    passing = [r for r in results if all_targets_met(r[2])]
    print(f"\n{len(passing)} / {len(results)} combos met ALL 3 targets simultaneously")
