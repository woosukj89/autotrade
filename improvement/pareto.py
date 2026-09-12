"""
Runs the voting-architecture sweep and reports the actual Pareto frontier
(best FPR at each coverage tier, best coverage at each FPR tier) instead of
collapsing everything to one arbitrary weighted-loss champion. Also saves
full results to results/ for reference.
"""
import json
import argparse
from dataclasses import asdict

from sweep import run_sweep_with_ranges, all_targets_met
from sweep3 import GUIDED_RANGES


def report_frontier(results):
    # Best FPR for coverage >= each tier
    print(f"\n{'='*90}\nBEST FALSE-POSITIVE RATE AT EACH COVERAGE TIER\n{'='*90}")
    for tier in [0.85, 0.80, 0.75, 0.70, 0.65]:
        candidates = [r for r in results if r[2]['coverage'] >= tier]
        if not candidates:
            print(f"coverage >= {tier*100:.0f}%: no candidates found")
            continue
        best = min(candidates, key=lambda r: r[2]['false_positive_rate'])
        res = best[2]
        print(f"coverage >= {tier*100:.0f}%: best FPR={res['false_positive_rate']*100:.1f}%  "
              f"(actual cov={res['coverage']*100:.1f}%, defret={res['defensive_cum_return']*100:.1f}%)")

    print(f"\n{'='*90}\nBEST COVERAGE AT EACH FALSE-POSITIVE-RATE TIER\n{'='*90}")
    for tier in [0.05, 0.07, 0.10, 0.15]:
        candidates = [r for r in results if r[2]['false_positive_rate'] <= tier]
        if not candidates:
            print(f"FPR <= {tier*100:.0f}%: no candidates found")
            continue
        best = max(candidates, key=lambda r: r[2]['coverage'])
        res = best[2]
        print(f"FPR <= {tier*100:.0f}%: best coverage={res['coverage']*100:.1f}%  "
              f"(actual FPR={res['false_positive_rate']*100:.1f}%, defret={res['defensive_cum_return']*100:.1f}%)")
        print(f"    {best[1]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()

    print(f'Running {args.n} combos for frontier analysis...')
    results = run_sweep_with_ranges(args.n, args.seed, GUIDED_RANGES)
    report_frontier(results)

    passing = [r for r in results if all_targets_met(r[2])]
    print(f"\n{len(passing)} / {len(results)} combos met ALL 3 targets simultaneously")

    # Save top 50 by combined loss for reference
    results.sort(key=lambda x: x[0])
    out = []
    for loss, params, res in results[:50]:
        out.append({'loss': loss, 'params': asdict(params),
                     'coverage': res['coverage'], 'false_positive_rate': res['false_positive_rate'],
                     'defensive_cum_return': res['defensive_cum_return']})
    with open('results/pareto_top50.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved results/pareto_top50.json")
