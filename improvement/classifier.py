"""
Binary bear/not-bear classifier: combines the signals in signals.py into a
single state machine. Output is a boolean pd.Series (True = DEFENSIVE), not
a continuous score - per SPEC.md, there is no partial allocation.

v6 ARCHITECTURE (voting, not "any 1 of 3 paths" OR logic): an earlier
version let fast_panic, a sustained trend break, or a trend+credit
same-day coincidence EACH independently trigger entry. That plateaued at
~10-13% false-positive rate across 45,000+ swept parameter combinations
(see PROGRESS.md) - any single one of 3 fairly noisy signals was too easy
to satisfy by chance. Replaced with:

Entry (NOT_IN_BEAR -> IN_BEAR), either of:
  (a) extreme_panic fires alone, no confirmation delay - a rare, very deep
      N-day drawdown (default -20%), reserved as a safety valve for a
      genuine fast crash so voting's extra confirmation delay can't cost
      real coverage on a 2020-tier event.
  (b) >= `vote_threshold` (default 2) of {fast_panic, trend_break,
      credit_stress} agree on the SAME day, persisted `combo_confirm_days`.
      Two independently-noisy signals coinciding is a much stronger filter
      than any one alone.
  (c) trend_break sustained for `trend_confirm_days` consecutive days with
      NO other corroboration at all - a pure fallback for a slow grind
      where neither panic nor credit ever confirms.

Exit (IN_BEAR -> NOT_IN_BEAR), ALL of:
  - trend_release is true (price back within 1% of its SMA)
  - fast_panic has been false for `panic_cooldown_days` consecutive days
  - if `require_credit_calm_to_exit`: credit_stress has also subsided
    (targets bear-market-rally whipsaws, e.g. 2022 Mar-Apr/Jul-Aug, where
    price bounced above trend but credit conditions were still stressed)

This asymmetry (harder to enter, easier to exit) is deliberate per SPEC.md
§5.3: protects the false-positive budget on entry, protects coverage
(doesn't overstay) on exit.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from signals import fast_panic_signal, trend_break_signal, trend_release_signal, credit_stress_signal, breadth_stress_signal


@dataclass
class ClassifierParams:
    """Defaults are the best point found on the empirical Pareto frontier
    after adding breadth as a 4th vote signal (see PROGRESS.md +
    results/pareto_top50_v7.json) - 110,000+ combos swept total across 4
    architectures: coverage=85.2%, FPR=11.6%, defensive_return=70.2%. Meets
    2 of 3 SPEC.md targets, and defensive_return is now far above its
    5% bar (was 27.2% pre-breadth). FPR remains the binding constraint -
    the frontier shows FPR<5% only achievable near 65% coverage even with
    4 independent signals voting (see PROGRESS.md for the full frontier
    table and what was ruled out along the way).
    """
    drawdown_lookback: int = 10
    drawdown_threshold: float = -0.10

    sma_window: int = 200
    entry_buffer_pct: float = 0.015
    exit_buffer_pct: float = 0.005
    trend_confirm_days: int = 20

    credit_z_lookback: int = 189
    credit_z_threshold: float = 1.25
    credit_momentum_lookback: int = 5

    panic_cooldown_days: int = 7
    min_defensive_days: int = 0
    fast_panic_confirm_days: int = 1
    combo_confirm_days: int = 1
    require_credit_calm_to_exit: bool = True

    # v6: voting architecture (see classify() docstring) - an extreme,
    # rare drawdown still fires alone as a safety valve; everything else
    # requires >= vote_threshold of {fast_panic, trend_break, credit_stress,
    # breadth_stress} to agree, instead of any single one being sufficient.
    extreme_drawdown_threshold: float = -0.16
    vote_threshold: int = 2

    # v7: breadth (% of S&P 500 above own 200d SMA) - independent
    # information from trend/drawdown/credit (see breadth.py). Not a clean
    # standalone discriminator (checked directly - 2011's false alarm had
    # LOWER breadth than 2022's real bear) but a real, measured improvement
    # once added as a 4th vote input (see PROGRESS.md v7 section).
    breadth_threshold: float = 55.0


def compute_raw_signals(data: Dict[str, pd.Series], p: ClassifierParams) -> pd.DataFrame:
    spy, vix, baa10y = data['spy'], data['vix'], data['baa10y']
    df = pd.DataFrame(index=spy.index)
    df['fast_panic'] = fast_panic_signal(spy, p.drawdown_lookback, p.drawdown_threshold)
    df['extreme_panic'] = fast_panic_signal(spy, p.drawdown_lookback, p.extreme_drawdown_threshold)
    df['trend_break'] = trend_break_signal(spy, p.sma_window, p.entry_buffer_pct)
    df['trend_release'] = trend_release_signal(spy, p.sma_window, p.exit_buffer_pct)
    df['credit_stress'] = credit_stress_signal(
        baa10y, p.credit_z_lookback, p.credit_z_threshold, p.credit_momentum_lookback)
    if 'breadth' in data:
        df['breadth_stress'] = breadth_stress_signal(data['breadth'], p.breadth_threshold)
    else:
        df['breadth_stress'] = False
    return df


def classify(data: Dict[str, pd.Series], p: ClassifierParams = None) -> Tuple[pd.Series, List[dict]]:
    """Returns (defensive: bool Series, transitions: list of {date, to, reason}).

    Performance note: iterates numpy arrays, not pandas .loc lookups - the
    original row-by-row .loc version took ~600ms/call (dominant cost, vs.
    ~20ms for scoring), which was the bottleneck for the sweep this was
    written to support. This version is 1-2 orders of magnitude faster for
    the same logic - state-machine loops don't vectorize away entirely, but
    all the per-row pandas overhead does.
    """
    p = p or ClassifierParams()
    sig = compute_raw_signals(data, p)

    def run_length(flag: pd.Series) -> pd.Series:
        """Consecutive-True run length ending at each row (0 where False)."""
        return (flag.groupby((~flag).cumsum()).cumcount() + 1) * flag

    trend_break_run = run_length(sig['trend_break'])
    panic_quiet_run = run_length(~sig['fast_panic'])
    # Persistence gates (TUNING v4): a single-day fast_panic or a single-day
    # trend+credit coincidence turned out to be the dominant false-positive
    # source (see PROGRESS.md - a 10000-combo sweep plateaued at ~10-14%
    # FPR regardless of thresholds, pointing at an architectural gap, not a
    # tuning one). Requiring these to persist a couple of days filters
    # one-off blips - real crashes accelerate over consecutive days, blips
    # don't - at the cost of a small amount of entry latency.
    fast_panic_run = run_length(sig['fast_panic'])
    # v7: 4-way vote (was 3-way) - breadth_stress added as independent
    # information (see breadth.py / signals.breadth_stress_signal).
    votes = (sig['fast_panic'].astype(int) + sig['trend_break'].astype(int)
             + sig['credit_stress'].astype(int) + sig['breadth_stress'].astype(int))
    vote_flag = votes >= p.vote_threshold
    vote_run = run_length(vote_flag)

    dates = sig.index
    fast_panic = sig['fast_panic'].to_numpy()
    extreme_panic = sig['extreme_panic'].to_numpy()
    trend_break = sig['trend_break'].to_numpy()
    trend_release = sig['trend_release'].to_numpy()
    credit_stress = sig['credit_stress'].to_numpy()
    trend_break_run_arr = trend_break_run.to_numpy()
    panic_quiet_run_arr = panic_quiet_run.to_numpy()
    fast_panic_run_arr = fast_panic_run.to_numpy()
    vote_run_arr = vote_run.to_numpy()

    n = len(dates)
    defensive_arr = [False] * n
    transitions = []
    in_bear = False
    days_in_state = 0
    trend_confirm_days = p.trend_confirm_days
    min_defensive_days = p.min_defensive_days
    panic_cooldown_days = p.panic_cooldown_days
    combo_confirm_days = p.combo_confirm_days
    require_credit_calm_to_exit = p.require_credit_calm_to_exit

    for i in range(n):
        days_in_state += 1
        if not in_bear:
            # v6 voting architecture: an extreme (rare) drawdown fires alone
            # as a safety valve (no confirmation delay - reserved for a
            # genuine crash, e.g. 1987/2020-tier moves). Everything else
            # requires >= vote_threshold of {fast_panic, trend_break,
            # credit_stress} to agree, persisted `combo_confirm_days` -
            # replaces the old "any 1 of 3 paths" OR logic that plateaued
            # at ~10-13% FPR across 45,000 swept combos (see PROGRESS.md).
            # trend_sustained kept as a pure-fallback for a slow grind with
            # no corroboration at all, gated by a much longer confirm window.
            reason = None
            if extreme_panic[i]:
                reason = 'extreme_panic'
            elif vote_run_arr[i] >= combo_confirm_days:
                reason = 'vote_2of3'
            elif trend_break_run_arr[i] >= trend_confirm_days:
                reason = 'trend_sustained'
            if reason:
                in_bear = True
                days_in_state = 0
                transitions.append({'date': dates[i].strftime('%Y-%m-%d'), 'to': 'DEFENSIVE', 'reason': reason})
        else:
            # min_defensive_days: bear-market rallies (sharp counter-trend
            # bounces within a real bear, e.g. 2022 Mar-Apr and Jul-Aug) can
            # satisfy trend_release within days of entry, causing whipsaw
            # exits mid-bear. Require a minimum dwell time before exit is
            # even considered - this directly targets that failure mode.
            # require_credit_calm_to_exit (v5): min_defensive_days was
            # found to inflate EVERY episode's length by the same fixed
            # floor, including short-lived pure-noise false alarms that
            # didn't need it - only real bear-rally whipsaws did. Gating
            # exit on credit stress having also subsided is more targeted:
            # it only holds the position open when credit conditions are
            # still genuinely stressed (which is what distinguishes a real
            # bear-market rally from a false alarm clearing out).
            credit_calm_ok = (not require_credit_calm_to_exit) or (not credit_stress[i])
            if days_in_state >= min_defensive_days and trend_release[i] and credit_calm_ok and panic_quiet_run_arr[i] >= panic_cooldown_days:
                in_bear = False
                days_in_state = 0
                transitions.append({'date': dates[i].strftime('%Y-%m-%d'), 'to': 'AGGRESSIVE', 'reason': 'trend_release+panic_quiet'})
        defensive_arr[i] = in_bear

    defensive = pd.Series(defensive_arr, index=dates)
    return defensive, transitions


if __name__ == '__main__':
    from data_feed import load_all
    data = load_all()
    defensive, transitions = classify(data)
    print(f'{len(transitions)} transitions')
    for t in transitions[-20:]:
        print(t)
    print(f'\nTotal days DEFENSIVE: {defensive.sum()} / {len(defensive)} ({defensive.mean()*100:.1f}%)')
