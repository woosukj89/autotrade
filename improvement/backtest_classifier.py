"""
Score a classifier's day-by-day DEFENSIVE/AGGRESSIVE output against ground
truth (ground_truth_periods.json) on the 3 target metrics from SPEC.md §2:

  - coverage: pooled (defensive AND in-bear days) / (total in-bear days), >= 85% target
  - false_positive_rate: pooled (defensive AND NOT in-bear days) / (total NOT-in-bear days), < 5% target
  - defensive_return: return of a short-SPY-proxy (SH where available, else -1x SPY)
    held only on days classified defensive, > 5% target (and must not be negative)

Classifier-first: this does NOT run a portfolio backtest. It only checks
whether the classifier's day-by-day calls match reality. See SPEC.md §6.
"""
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

from data_feed import load_all, fetch_price_series
from ground_truth import load_ground_truth, WINDOW_START, WINDOW_END
from classifier import classify, ClassifierParams


def build_bear_mask(index: pd.DatetimeIndex, gt: dict, key: str = 'bears') -> pd.Series:
    mask = pd.Series(False, index=index)
    for period in gt[key]:
        start = pd.Timestamp(period['peak_date'])
        end = pd.Timestamp(period['trough_date'])
        mask.loc[(index >= start) & (index <= end)] = True
    return mask


def defensive_sleeve_returns(index: pd.DatetimeIndex) -> pd.Series:
    """Daily returns of the defensive sleeve proxy: real SH returns where
    available (inception 2006-06-19), synthetic -1x SPY before that."""
    spy = fetch_price_series('SPY').reindex(index).ffill()
    spy_ret = spy.pct_change().fillna(0.0)
    try:
        sh = fetch_price_series('SH').reindex(index)
        sh_ret = sh.pct_change()
        combined = sh_ret.where(sh_ret.notna(), -spy_ret)
        return combined.fillna(0.0)
    except Exception:
        return -spy_ret


def score(defensive: pd.Series, gt: dict, window_start=WINDOW_START, window_end=WINDOW_END,
          _def_ret_cache: pd.Series = None) -> dict:
    idx = defensive.index
    in_window = (idx >= pd.Timestamp(window_start)) & (idx <= pd.Timestamp(window_end))
    defensive_w = defensive[in_window]
    idx_w = defensive_w.index

    bear_mask = build_bear_mask(idx_w, gt, 'bears')
    corr_mask = build_bear_mask(idx_w, gt, 'corrections')
    not_bear_mask = ~bear_mask

    total_bear_days = int(bear_mask.sum())
    covered_bear_days = int((defensive_w & bear_mask).sum())
    coverage = covered_bear_days / total_bear_days if total_bear_days else float('nan')

    total_not_bear_days = int(not_bear_mask.sum())
    false_positive_days = int((defensive_w & not_bear_mask).sum())
    false_positive_rate = false_positive_days / total_not_bear_days if total_not_bear_days else float('nan')

    total_corr_days = int(corr_mask.sum())
    covered_corr_days = int((defensive_w & corr_mask).sum())
    corr_coverage = covered_corr_days / total_corr_days if total_corr_days else float('nan')

    # def_ret doesn't depend on classifier params - callers doing a sweep
    # should precompute once (on the FULL index) and pass via
    # _def_ret_cache to avoid re-reading price CSVs on every one of
    # thousands of calls. Reindex to idx_w since the cache may span a wider
    # range (e.g. includes pre-window lookback data) than the scoring window.
    def_ret = (_def_ret_cache.reindex(idx_w) if _def_ret_cache is not None
               else defensive_sleeve_returns(idx_w))
    defensive_daily_rets = def_ret[defensive_w.to_numpy()]
    if len(defensive_daily_rets) > 0:
        defensive_cum_return = float((1 + defensive_daily_rets).prod() - 1)
    else:
        defensive_cum_return = 0.0

    per_bear = []
    for period in gt['bears']:
        start = pd.Timestamp(period['peak_date'])
        end = pd.Timestamp(period['trough_date'])
        window_mask = (idx_w >= start) & (idx_w <= end)
        bear_days = int(window_mask.sum())
        covered = int((defensive_w & window_mask).sum())
        bear_def_rets = def_ret[window_mask & defensive_w]
        bear_ret = float((1 + bear_def_rets).prod() - 1) if len(bear_def_rets) else 0.0
        per_bear.append({
            'peak_date': period['peak_date'], 'trough_date': period['trough_date'],
            'depth_pct': period['depth_pct'], 'days': bear_days,
            'covered_days': covered, 'coverage_pct': covered / bear_days * 100 if bear_days else 0.0,
            'defensive_return_pct': bear_ret * 100,
        })

    return {
        'coverage': coverage,
        'false_positive_rate': false_positive_rate,
        'defensive_cum_return': defensive_cum_return,
        'correction_coverage_informational': corr_coverage,
        'total_bear_days': total_bear_days,
        'covered_bear_days': covered_bear_days,
        'total_not_bear_days': total_not_bear_days,
        'false_positive_days': false_positive_days,
        'total_defensive_days': int(defensive_w.sum()),
        'per_bear': per_bear,
    }


def print_report(result: dict):
    print(f"\n{'='*70}\nCLASSIFIER SCORE vs GROUND TRUTH\n{'='*70}")
    cov = result['coverage'] * 100
    fpr = result['false_positive_rate'] * 100
    dret = result['defensive_cum_return'] * 100
    print(f"Coverage:              {cov:6.1f}%   (target >= 85%)   {'PASS' if cov >= 85 else 'FAIL'}")
    print(f"False-positive rate:   {fpr:6.1f}%   (target <  5%)    {'PASS' if fpr < 5 else 'FAIL'}")
    print(f"Defensive-period ret:  {dret:6.1f}%   (target >  5%)    {'PASS' if dret > 5 else 'FAIL'}")
    print(f"\n(Informational) Correction coverage: {result['correction_coverage_informational']*100:.1f}%")
    print(f"\nBear days: {result['total_bear_days']}, covered: {result['covered_bear_days']}")
    print(f"Not-bear days: {result['total_not_bear_days']}, false-positive days: {result['false_positive_days']}")
    print(f"Total days classified DEFENSIVE: {result['total_defensive_days']}")

    print(f"\n{'Bear':<24}{'Depth':>8}{'Days':>7}{'Covered':>9}{'Coverage':>10}{'DefRet':>10}")
    for b in result['per_bear']:
        print(f"{b['peak_date']}->{b['trough_date']:<10}{b['depth_pct']:>7.1f}%{b['days']:>7}"
              f"{b['covered_days']:>9}{b['coverage_pct']:>9.1f}%{b['defensive_return_pct']:>9.1f}%")


if __name__ == '__main__':
    data = load_all()
    gt = load_ground_truth()
    defensive, transitions = classify(data)
    result = score(defensive, gt)
    print_report(result)
    print(f"\n{len(transitions)} total transitions ({len(transitions)//2} entries)")
