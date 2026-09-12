"""
Mechanical, reproducible bear-market (and correction) labeler.

Implements the standard peak-to-trough drawdown rule (see SPEC.md §3):
a bear begins at the most recent closing high and is confirmed once price
closes >= threshold below that high; it ends (for SCORING purposes) at the
trough — the lowest close before price recovers back above the pre-bear
peak. This is deliberately NOT hand-curated: run this file and whatever it
finds against SPY closes IS the ground truth, so results are reproducible
and not subject to cherry-picking.

Two thresholds:
  - 0.20 (bear market, the primary ground truth used for the 85%/5%/5% targets)
  - 0.10 (correction, secondary/informational only — see SPEC.md §3.2)

Usage:
    python ground_truth.py                  # fetch, label, print, cache to disk
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import json
import os

import pandas as pd
import yfinance as yf


CACHE_PATH = os.path.join(os.path.dirname(__file__), 'ground_truth_periods.json')

# Extra lookback before the nominal 20yr window so a peak that occurred
# slightly before 2005 (with the decline continuing into it) isn't missed.
DATA_START = '2003-01-01'
WINDOW_START = '2005-01-01'
WINDOW_END = '2025-12-31'


@dataclass
class BearPeriod:
    kind: str            # "bear" (>=20%) or "correction" (10-20%)
    peak_date: str
    peak_price: float
    trough_date: str
    trough_price: float
    recovery_date: Optional[str]   # date price closed back above peak_price, if it has by data end
    depth_pct: float               # (peak - trough) / peak
    decline_days: int              # peak_date -> trough_date, calendar days


def fetch_spy_closes(start=DATA_START, end=WINDOW_END) -> pd.Series:
    df = yf.download('SPY', start=start, end=end, progress=False, auto_adjust=True)
    closes = df['Close']
    if hasattr(closes, 'squeeze'):
        closes = closes.squeeze()
    closes.index = pd.to_datetime(closes.index)
    return closes.sort_index()


def label_periods(closes: pd.Series, threshold: float, kind: str) -> List[BearPeriod]:
    """Mechanical peak-to-trough-to-recovery labeling at a fixed drawdown threshold.

    Note: episodes are found using the FULL data history (including
    pre-window lookback) so a peak from just before WINDOW_START isn't
    missed, but episodes are only kept if their peak_date falls within
    [WINDOW_START, WINDOW_END] later, by the caller.
    """
    periods: List[BearPeriod] = []
    peak_price = closes.iloc[0]
    peak_date = closes.index[0]
    in_episode = False
    trough_price = None
    trough_date = None
    episode_peak_price = None
    episode_peak_date = None

    for date, price in closes.items():
        if not in_episode:
            if price > peak_price:
                peak_price = price
                peak_date = date
            else:
                drawdown = (peak_price - price) / peak_price
                if drawdown >= threshold:
                    in_episode = True
                    episode_peak_price = peak_price
                    episode_peak_date = peak_date
                    trough_price = price
                    trough_date = date
        else:
            if price < trough_price:
                trough_price = price
                trough_date = date
            if price >= episode_peak_price:
                periods.append(BearPeriod(
                    kind=kind,
                    peak_date=episode_peak_date.strftime('%Y-%m-%d'),
                    peak_price=float(episode_peak_price),
                    trough_date=trough_date.strftime('%Y-%m-%d'),
                    trough_price=float(trough_price),
                    recovery_date=date.strftime('%Y-%m-%d'),
                    depth_pct=float((episode_peak_price - trough_price) / episode_peak_price * 100),
                    decline_days=(trough_date - episode_peak_date).days,
                ))
                in_episode = False
                peak_price = price
                peak_date = date

    # Unresolved episode at end of data (no recovery yet) — still valid ground
    # truth for peak->trough scoring, recovery_date left as None.
    if in_episode:
        periods.append(BearPeriod(
            kind=kind,
            peak_date=episode_peak_date.strftime('%Y-%m-%d'),
            peak_price=float(episode_peak_price),
            trough_date=trough_date.strftime('%Y-%m-%d'),
            trough_price=float(trough_price),
            recovery_date=None,
            depth_pct=float((episode_peak_price - trough_price) / episode_peak_price * 100),
            decline_days=(trough_date - episode_peak_date).days,
        ))

    return periods


def _within_window(p: BearPeriod, start=WINDOW_START, end=WINDOW_END) -> bool:
    return start <= p.peak_date <= end


def build_ground_truth(save: bool = True) -> dict:
    closes = fetch_spy_closes()

    bears_all = label_periods(closes, threshold=0.20, kind='bear')
    corrections_all = label_periods(closes, threshold=0.10, kind='correction')

    bears = [p for p in bears_all if _within_window(p)]
    # Corrections that overlap a bear's peak-trough window are dropped —
    # every bear is by definition also a >=10% correction; we only want the
    # ADDITIONAL shallower (10-20%) events as the secondary category.
    bear_windows = [(p.peak_date, p.trough_date) for p in bears_all]

    def overlaps_a_bear(p: BearPeriod) -> bool:
        return any(bw[0] <= p.peak_date <= bw[1] or bw[0] <= p.trough_date <= bw[1] for bw in bear_windows)

    corrections = [p for p in corrections_all if _within_window(p) and not overlaps_a_bear(p)]

    result = {
        'generated_at': datetime.now().isoformat(),
        'window': [WINDOW_START, WINDOW_END],
        'threshold_bear': 0.20,
        'threshold_correction': 0.10,
        'bears': [asdict(p) for p in bears],
        'corrections': [asdict(p) for p in corrections],
    }

    if save:
        with open(CACHE_PATH, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Saved {CACHE_PATH}')

    return result


def load_ground_truth() -> dict:
    if not os.path.exists(CACHE_PATH):
        return build_ground_truth(save=True)
    with open(CACHE_PATH) as f:
        return json.load(f)


if __name__ == '__main__':
    gt = build_ground_truth()

    print(f"\n{'='*78}\nBEARS (>=20% drawdown) — {gt['window'][0]} to {gt['window'][1]}\n{'='*78}")
    print(f"{'Peak':<12}{'Trough':<12}{'Recovery':<12}{'Depth':>8}{'Decline days':>15}")
    total_bear_days = 0
    for b in gt['bears']:
        print(f"{b['peak_date']:<12}{b['trough_date']:<12}{(b['recovery_date'] or 'ongoing'):<12}"
              f"{b['depth_pct']:>7.1f}%{b['decline_days']:>15}")
        total_bear_days += b['decline_days']
    print(f"\nTotal bear (decline-phase) days: {total_bear_days}")

    print(f"\n{'='*78}\nCORRECTIONS (10-20% drawdown, not already part of a bear)\n{'='*78}")
    print(f"{'Peak':<12}{'Trough':<12}{'Recovery':<12}{'Depth':>8}{'Decline days':>15}")
    for c in gt['corrections']:
        print(f"{c['peak_date']:<12}{c['trough_date']:<12}{(c['recovery_date'] or 'ongoing'):<12}"
              f"{c['depth_pct']:>7.1f}%{c['decline_days']:>15}")
