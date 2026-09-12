"""
Randomized parameter sweep over ClassifierParams, scored against ground
truth. Random sampling (not full grid - the space is way too large) from
reasonable ranges, ranked by a combined loss:

    loss = max(0, 0.85 - coverage) + max(0, fpr - 0.05) + max(0, 0.05 - defensive_return)

0 = all three SPEC.md targets met simultaneously. All three terms are on
comparable 0-1-ish scales (coverage/fpr are fractions; defensive_return is
also expressed as a fraction, uncapped above but only its DEFICIT below 5%
counts against the loss, so a very profitable defensive sleeve doesn't
inflate the score - the metric target is "more than 5%", not "as much as
possible").

Usage:
    python sweep.py --n 3000 --seed 0
"""
import argparse
import random
import time
from dataclasses import replace

import pandas as pd

from data_feed import load_all
from ground_truth import load_ground_truth
from classifier import classify, ClassifierParams
from backtest_classifier import score, defensive_sleeve_returns


PARAM_RANGES = {
    'drawdown_lookback': [5, 7, 10, 15, 20],
    'drawdown_threshold': [-0.08, -0.10, -0.12, -0.14, -0.16, -0.18],
    'sma_window': [50, 75, 100, 150, 200],
    'entry_buffer_pct': [0.01, 0.02, 0.03, 0.04, 0.05],
    'exit_buffer_pct': [0.0, 0.005, 0.01, 0.02],
    'trend_confirm_days': [3, 5, 7, 10, 15, 20, 25],
    'credit_z_lookback': [126, 189, 252],
    'credit_z_threshold': [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    'credit_momentum_lookback': [5, 10, 15, 20],
    'panic_cooldown_days': [2, 3, 5, 7, 10, 15],
    'min_defensive_days': [0, 5, 10, 15, 20, 30, 40, 50],
    'combo_confirm_days': [1, 2, 3, 5, 8],
    'require_credit_calm_to_exit': [False, True],
    'extreme_drawdown_threshold': [-0.16, -0.18, -0.20, -0.22, -0.25],
    'vote_threshold': [2, 3],
}


def random_params(rng: random.Random, ranges: dict = None) -> ClassifierParams:
    ranges = ranges or PARAM_RANGES
    kwargs = {k: rng.choice(v) for k, v in ranges.items()}
    return ClassifierParams(**kwargs)


def combined_loss(result: dict) -> float:
    cov_deficit = max(0.0, 0.85 - result['coverage'])
    fpr_excess = max(0.0, result['false_positive_rate'] - 0.05)
    ret_deficit = max(0.0, 0.05 - result['defensive_cum_return'])
    return cov_deficit + fpr_excess + ret_deficit


def all_targets_met(result: dict) -> bool:
    return (result['coverage'] >= 0.85 and
            result['false_positive_rate'] < 0.05 and
            result['defensive_cum_return'] > 0.05)


def run_sweep(n: int, seed: int = 0):
    return run_sweep_with_ranges(n, seed, PARAM_RANGES)


def run_sweep_with_ranges(n: int, seed: int, ranges: dict):
    data = load_all()
    gt = load_ground_truth()
    def_ret_cache = defensive_sleeve_returns(data['spy'].index)

    rng = random.Random(seed)
    results = []
    t0 = time.time()
    for i in range(n):
        params = random_params(rng, ranges)
        try:
            defensive, _transitions = classify(data, params)
            result = score(defensive, gt, _def_ret_cache=def_ret_cache)
        except Exception as e:
            continue
        loss = combined_loss(result)
        results.append((loss, params, result))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{n} ({elapsed:.1f}s, {elapsed/(i+1)*1000:.1f}ms/combo)')

    results.sort(key=lambda x: x[0])
    return results


def print_top(results, k=15):
    print(f"\n{'='*100}\nTOP {k} CANDIDATES (loss=0 means all 3 targets met)\n{'='*100}")
    for rank, (loss, params, result) in enumerate(results[:k], 1):
        cov = result['coverage'] * 100
        fpr = result['false_positive_rate'] * 100
        dret = result['defensive_cum_return'] * 100
        passed = all_targets_met(result)
        marker = ' <<< ALL TARGETS MET' if passed else ''
        print(f"#{rank:2d} loss={loss:.4f}  cov={cov:5.1f}%  fpr={fpr:5.1f}%  defret={dret:6.1f}%{marker}")
        print(f"     {params}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(f'Running {args.n} random combos (seed={args.seed})...')
    results = run_sweep(args.n, args.seed)
    print_top(results, k=20)

    passing = [r for r in results if all_targets_met(r[2])]
    print(f"\n{len(passing)} / {len(results)} combos met ALL 3 targets simultaneously")
