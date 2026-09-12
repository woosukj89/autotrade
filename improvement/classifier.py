"""
Binary bear/not-bear classifier: combines the signals in signals.py into a
single state machine. Output is a boolean pd.Series (True = DEFENSIVE), not
a continuous score - per SPEC.md, there is no partial allocation.

Entry (NOT_IN_BEAR -> IN_BEAR), any ONE of:
  (a) fast_panic fires on its own (no confirmation delay - it's built to be
      selective already: a real N-day drawdown or a real VIX spike/level).
  (b) trend_break sustained for `trend_confirm_days` consecutive days (the
      slow-grind path - no single-day panic, just a confirmed trend down).
  (c) trend_break AND credit_stress both true on the same day (a faster
      confirmed-combo path: trend turning down WITH independent credit
      confirmation, don't need to wait out the full sustain period).

Exit (IN_BEAR -> NOT_IN_BEAR), ALL of:
  - trend_release is true (price back within 1% of its SMA)
  - fast_panic has been false for `panic_cooldown_days` consecutive days
  (deliberately no credit-spread condition on exit - credit tends to lag
  on the way down AND the way up; gating the exit on it would cost coverage)

This asymmetry (harder to enter, easier to exit) is deliberate per SPEC.md
§5.3: protects the false-positive budget on entry, protects coverage
(doesn't overstay) on exit.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from signals import fast_panic_signal, trend_break_signal, trend_release_signal, credit_stress_signal


@dataclass
class ClassifierParams:
    drawdown_lookback: int = 10
    drawdown_threshold: float = -0.12

    sma_window: int = 150
    entry_buffer_pct: float = 0.03
    exit_buffer_pct: float = 0.01
    trend_confirm_days: int = 10

    credit_z_lookback: int = 252
    credit_z_threshold: float = 1.0
    credit_momentum_lookback: int = 10

    panic_cooldown_days: int = 5
    min_defensive_days: int = 20


def compute_raw_signals(data: Dict[str, pd.Series], p: ClassifierParams) -> pd.DataFrame:
    spy, vix, baa10y = data['spy'], data['vix'], data['baa10y']
    df = pd.DataFrame(index=spy.index)
    df['fast_panic'] = fast_panic_signal(spy, p.drawdown_lookback, p.drawdown_threshold)
    df['trend_break'] = trend_break_signal(spy, p.sma_window, p.entry_buffer_pct)
    df['trend_release'] = trend_release_signal(spy, p.sma_window, p.exit_buffer_pct)
    df['credit_stress'] = credit_stress_signal(
        baa10y, p.credit_z_lookback, p.credit_z_threshold, p.credit_momentum_lookback)
    return df


def classify(data: Dict[str, pd.Series], p: ClassifierParams = None) -> Tuple[pd.Series, List[dict]]:
    """Returns (defensive: bool Series, transitions: list of {date, to, reason})."""
    p = p or ClassifierParams()
    sig = compute_raw_signals(data, p)

    trend_break_run = (
        sig['trend_break']
        .groupby((~sig['trend_break']).cumsum())
        .cumcount() + 1
    ) * sig['trend_break']

    panic_quiet_run = (
        (~sig['fast_panic'])
        .groupby(sig['fast_panic'].cumsum())
        .cumcount() + 1
    ) * (~sig['fast_panic'])

    defensive = pd.Series(False, index=sig.index)
    transitions = []
    in_bear = False
    days_in_state = 0

    for date in sig.index:
        row = sig.loc[date]
        days_in_state += 1
        if not in_bear:
            reason = None
            if row['fast_panic']:
                reason = 'fast_panic'
            elif trend_break_run.loc[date] >= p.trend_confirm_days:
                reason = 'trend_sustained'
            elif row['trend_break'] and row['credit_stress']:
                reason = 'trend+credit_confirmed'
            if reason:
                in_bear = True
                days_in_state = 0
                transitions.append({'date': date.strftime('%Y-%m-%d'), 'to': 'DEFENSIVE', 'reason': reason})
        else:
            # min_defensive_days: bear-market rallies (sharp counter-trend
            # bounces within a real bear, e.g. 2022 Mar-Apr and Jul-Aug) can
            # satisfy trend_release within days of entry, causing whipsaw
            # exits mid-bear. Require a minimum dwell time before exit is
            # even considered - this directly targets that failure mode.
            if days_in_state >= p.min_defensive_days and row['trend_release'] and panic_quiet_run.loc[date] >= p.panic_cooldown_days:
                in_bear = False
                days_in_state = 0
                transitions.append({'date': date.strftime('%Y-%m-%d'), 'to': 'AGGRESSIVE', 'reason': 'trend_release+panic_quiet'})
        defensive.loc[date] = in_bear

    return defensive, transitions


if __name__ == '__main__':
    from data_feed import load_all
    data = load_all()
    defensive, transitions = classify(data)
    print(f'{len(transitions)} transitions')
    for t in transitions[-20:]:
        print(t)
    print(f'\nTotal days DEFENSIVE: {defensive.sum()} / {len(defensive)} ({defensive.mean()*100:.1f}%)')
