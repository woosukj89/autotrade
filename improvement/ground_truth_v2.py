"""
Multi-definition ground truth generator (SPEC.md §0 reframe): the fixed
20%-rule ground truth only produced 3 true bears in 20 years - too small a
sample to calibrate a <5% false-positive classifier against (confirmed
empirically, see PROGRESS.md). Instead of one fixed definition, generate
several and let Step 3 (actual portfolio growth) pick the winner.

Two STYLES, each mechanical/reproducible (no hand-picked dates):

  - "alltime": the original rule (ground_truth.py) - peak is the running
    all-time high, bear ends at trough (recovery back above the peak is
    NOT required for the scored window - see ground_truth.py's note on
    why peak-to-trough, not peak-to-recovery, is what's scored). Matches
    the user's own description: ">10% drawdown from previous peak...until
    a new high is achieved compared to previous peak" ends the SEARCH for
    a new peak, but the drawdown/defensive-relevant window is peak->trough.

  - "rolling": peak is the max over a trailing N-day window, not all-time.
    Resets faster than an all-time high, so it produces more frequent,
    shorter, more cyclically-local episodes - useful because "no new
    all-time high in years" (e.g. a slow multi-year grind back to even)
    would otherwise keep an all-time-peak definition's episode open long
    past when a tactical strategy should have gone back to aggressive.
    This is the "define your own" addition, not something the user
    specified directly.

Usage:
    python ground_truth_v2.py           # generate all definitions, save + summarize
"""
import json
import os
from dataclasses import asdict
from typing import List

import pandas as pd

from ground_truth import (
    fetch_spy_closes, label_periods, BearPeriod,
    WINDOW_START, WINDOW_END,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), 'ground_truth_defs')


def label_periods_rolling(closes: pd.Series, threshold: float, window: int) -> List[BearPeriod]:
    """Rolling-peak drawdown episodes: in-bear whenever price is >= threshold
    below its trailing `window`-day high; episode ends when that drawdown
    recovers back under threshold (not necessarily a full round-trip to the
    peak price itself - the rolling peak itself may have since drifted).
    """
    rolling_peak = closes.rolling(window, min_periods=1).max()
    drawdown = (rolling_peak - closes) / rolling_peak
    in_bear_flag = drawdown >= threshold

    periods: List[BearPeriod] = []
    in_episode = False
    ep_start_idx = None
    trough_price = None
    trough_date = None
    peak_at_start = None

    dates = closes.index
    for i, date in enumerate(dates):
        if in_bear_flag.iloc[i]:
            if not in_episode:
                in_episode = True
                ep_start_idx = i
                peak_at_start = rolling_peak.iloc[i]
                trough_price = closes.iloc[i]
                trough_date = date
            else:
                if closes.iloc[i] < trough_price:
                    trough_price = closes.iloc[i]
                    trough_date = date
        else:
            if in_episode:
                periods.append(BearPeriod(
                    kind=f'rolling{window}_{threshold}',
                    peak_date=dates[ep_start_idx].strftime('%Y-%m-%d'),
                    peak_price=float(peak_at_start),
                    trough_date=trough_date.strftime('%Y-%m-%d'),
                    trough_price=float(trough_price),
                    recovery_date=date.strftime('%Y-%m-%d'),
                    depth_pct=float((peak_at_start - trough_price) / peak_at_start * 100),
                    decline_days=(trough_date - dates[ep_start_idx]).days,
                ))
                in_episode = False

    if in_episode:
        periods.append(BearPeriod(
            kind=f'rolling{window}_{threshold}',
            peak_date=dates[ep_start_idx].strftime('%Y-%m-%d'),
            peak_price=float(peak_at_start),
            trough_date=trough_date.strftime('%Y-%m-%d'),
            trough_price=float(trough_price),
            recovery_date=None,
            depth_pct=float((peak_at_start - trough_price) / peak_at_start * 100),
            decline_days=(trough_date - dates[ep_start_idx]).days,
        ))

    return periods


def _within_window(p: BearPeriod, start=WINDOW_START, end=WINDOW_END) -> bool:
    return start <= p.peak_date <= end


def merge_adjacent(periods: List[BearPeriod], gap_days: int = 5) -> List[BearPeriod]:
    """Rolling-window episodes can chatter (in/out/in again within days as
    price oscillates near the threshold). Merge episodes separated by a gap
    shorter than `gap_days` into one - this is a labeling-noise cleanup,
    not a classifier concern."""
    if not periods:
        return periods
    periods = sorted(periods, key=lambda p: p.peak_date)
    merged = [periods[0]]
    for p in periods[1:]:
        prev = merged[-1]
        gap = (pd.Timestamp(p.peak_date) - pd.Timestamp(prev.trough_date)).days
        if gap <= gap_days:
            # extend prev through this episode
            if p.trough_price < prev.trough_price:
                prev.trough_price = p.trough_price
                prev.trough_date = p.trough_date
            prev.recovery_date = p.recovery_date
            prev.decline_days = (pd.Timestamp(prev.trough_date) - pd.Timestamp(prev.peak_date)).days
        else:
            merged.append(p)
    return merged


DEFINITIONS = [
    # (name, style, threshold, window_or_None)
    ('alltime_05', 'alltime', 0.05, None),
    ('alltime_10', 'alltime', 0.10, None),
    ('alltime_15', 'alltime', 0.15, None),
    ('alltime_20', 'alltime', 0.20, None),
    ('rolling252_10', 'rolling', 0.10, 252),
    ('rolling252_15', 'rolling', 0.15, 252),
    ('rolling126_10', 'rolling', 0.10, 126),
]


def generate_all(save: bool = True) -> dict:
    closes = fetch_spy_closes()
    results = {}
    for name, style, threshold, window in DEFINITIONS:
        if style == 'alltime':
            all_periods = label_periods(closes, threshold=threshold, kind=name)
        else:
            all_periods = label_periods_rolling(closes, threshold=threshold, window=window)
            all_periods = merge_adjacent(all_periods)

        periods = [p for p in all_periods if _within_window(p)]
        total_days = sum(p.decline_days for p in periods)
        results[name] = {
            'style': style, 'threshold': threshold, 'window': window,
            'window_range': [WINDOW_START, WINDOW_END],
            'bears': [asdict(p) for p in periods],
            'n_episodes': len(periods),
            'total_decline_days': total_days,
        }
        print(f'{name:16s} style={style:8s} thr={threshold:.0%}  window={str(window):5s}  '
              f'episodes={len(periods):3d}  total_days={total_days:5d}')

        if save:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, f'{name}.json'), 'w') as f:
                json.dump(results[name], f, indent=2)

    return results


if __name__ == '__main__':
    generate_all()
