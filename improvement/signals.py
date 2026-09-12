"""
Candidate signals for the binary bear classifier. Each returns a boolean
pd.Series (True = this signal is currently flagging stress), aligned to the
same daily index as the input data. Signals are deliberately kept simple and
independently interpretable — the classifier (classifier.py) decides how to
combine them; a signal firing does NOT by itself mean "go defensive."

Three distinct roles, per SPEC.md §5:
  - fast_panic_signal: reacts within days, for shock events (2020-style).
    Pure price/vol based - no macro data lag.
  - trend_break_signal: the slow, reliable backbone for grinding bears
    (2022-style) that never produce a single panic day.
  - credit_stress_signal: independent macro confirmation (validated this
    session as informative for 2008/2022) to raise confidence and cut
    false positives on the trend signal.
"""
import numpy as np
import pandas as pd


def fast_panic_signal(spy: pd.Series, drawdown_lookback: int = 10,
                       drawdown_threshold: float = -0.12) -> pd.Series:
    """True on a sharp N-day price drawdown.

    TUNING (v1->v2, empirical): VIX level/spike as standalone OR conditions
    were checked against actual VIX behavior in real bears vs. false alarms
    (see improvement/results/ notes) and turned out to be a poor
    discriminator for THIS purpose - 2022's own bear-period max VIX (36.5)
    was lower than several false-alarm episodes' max VIX (2011: 48.0, a
    2024 flash spike: 38.6). Acute volatility spikes predict near-term
    panic, not sustained 20%+ declines. The N-day price drawdown itself
    (37 flagged days total, cleanly concentrated in 2008/2011/2020/2025)
    was far more selective, so it's now the sole trigger. VIX level is kept
    as an optional AND-confirmation (both default off / very high bar) for
    future tuning, not a standalone trigger.
    """
    rolling_max = spy.rolling(drawdown_lookback).max()
    n_day_drawdown = (spy - rolling_max) / rolling_max
    drawdown_flag = n_day_drawdown <= drawdown_threshold
    return drawdown_flag.fillna(False)


def trend_break_signal(spy: pd.Series, sma_window: int = 150, buffer_pct: float = 0.03) -> pd.Series:
    """True when price closes more than `buffer_pct` below its `sma_window`-day
    SMA. The buffer (vs. a bare crossing) is the whipsaw-reduction step
    researched in SPEC.md §4 - price has to clearly break trend, not just
    tick under the line.
    """
    sma = spy.rolling(sma_window).mean()
    return (spy < sma * (1 - buffer_pct)).fillna(False)


def trend_release_signal(spy: pd.Series, sma_window: int = 150, buffer_pct: float = 0.01) -> pd.Series:
    """Companion to trend_break_signal for EXITING defensive: a much smaller
    buffer (price back above SMA * 0.99) so the classifier doesn't need a
    full trend re-confirmation to release - per SPEC.md's asymmetric design,
    exiting should be easier than entering (protects coverage, and a
    short-biased defensive sleeve shouldn't fight a recovering market).
    """
    sma = spy.rolling(sma_window).mean()
    return (spy >= sma * (1 - buffer_pct)).fillna(False)


def breadth_stress_signal(breadth_pct: pd.Series, threshold: float = 40.0) -> pd.Series:
    """True when S&P 500 breadth (% of constituents above their own 200-day
    SMA) is below `threshold` - i.e. weakness is broad-based, not confined
    to a few names. See breadth.py's module docstring for what this is and
    its survivorship-bias caveat. Checked directly against real bears vs.
    false alarms: doesn't cleanly separate them on its own (2011's false
    alarm had LOWER breadth than 2022's real bear - 2011 was a broader but
    shorter panic), but it's genuinely independent information from
    trend/drawdown/credit, so it's included as a 4th vote input rather than
    discarded on that univariate read alone.
    """
    return (breadth_pct < threshold).fillna(False)


def credit_stress_signal(baa10y: pd.Series, z_lookback: int = 252, z_threshold: float = 1.0,
                          momentum_lookback: int = 10) -> pd.Series:
    """True when the Baa-10Y credit spread is both elevated relative to its
    own trailing-year norm (z-score) AND still widening over the last
    2 weeks. Requiring both avoids flagging a spread that's high but already
    stabilizing/improving.
    """
    baa10y = baa10y.ffill()
    rolling_mean = baa10y.rolling(z_lookback).mean()
    rolling_std = baa10y.rolling(z_lookback).std()
    z = (baa10y - rolling_mean) / rolling_std.replace(0, np.nan)
    elevated = z >= z_threshold
    widening = baa10y.diff(momentum_lookback) > 0
    return (elevated & widening).fillna(False)
