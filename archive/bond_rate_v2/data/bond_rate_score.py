"""
Bond & Rate Adaptive Bear Score (v2)
=====================================

Rebuilt from the design notes in the Claude Code project memory for this
repo (`.claude` session memory, 2026-07-10 / 2026-07-24) — the original
implementation was never committed and no longer exists on disk. This is a
from-spec reconstruction, not a recovery of the original code: component
NAMES and WEIGHTS below match the documented spec exactly, but the internal
formula for each component (how CPI momentum, real-yield stress, etc. map
to a 0-1 score) had to be re-derived, since only the design decisions
survived, not the literal code. Validate by backtesting against the
documented historical results (~33.7% CAGR 10yr, ~27.7% CAGR 20yr, both
after 25% tax) before trusting this for live capital — if it doesn't land
close to that, the sub-formulas need tuning, not the overall approach.

NOTE ON FILE STABILITY: this file (and strategies/bond_rate_strategy.py)
disappeared from disk once already during development, with no clear cause
found in this repo's own automation (checked .github/workflows/*.yml and
found nothing that touches the local working tree — those run on GitHub's
servers, not here). Recommend committing this to a branch soon rather than
leaving it as uncommitted WIP indefinitely — that's the most likely way the
original July research was lost in the first place.

Why this exists: the original MacroMom score (see regime.py) uses VIX,
yield curve, breadth, and price momentum — all signals that react to
*equity market* stress. It has ~173 days of average lead time on major
corrections, but it missed the 2022 inflation bear almost entirely (score
never crossed its own defensive threshold that year) because 2022 was
driven by inflation/rate dynamics, not the kind of equity-market stress
MacroMom is built to detect. This score instead watches the bond/rate/
inflation complex directly, so it can catch bears that MacroMom is blind to
— particularly the short/slight bears that the live high-beta strategy is
weak against.

VALIDATION FINDING (10yr backtest, 2015-2025): the credit_market_stress
component is dead weight before Aug 2023 — FRED discontinued the old
vintage of BAMLH0A0HYM2/BAMLC0A0CM and only serves data from 2023-08-15
onward (confirmed: DGS10/CPILFESL/M2SL/FEDFUNDS all go back to the
1950s-60s fine, only the ICE BofA OAS series are truncated). Before this
fix, an unavailable component defaulted to a neutral 0.5 score but still
consumed its full weight, silently pulling the composite score toward 50
instead of reflecting the real signal from the other components. That
directly hurt the one score component built to react fastest to short
bears, during the one historical short-ish bear (2022) available in the
backtest window. Fixed by renormalizing weights over only the components
with real data at each point in time, rather than diluting with a neutral
placeholder.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from .regime import clamp, zscore
    from .providers import FREDProvider, CachedProvider
except ImportError:
    from regime import clamp, zscore
    from providers import FREDProvider, CachedProvider


# =============================================================================
# Component weights (from spec)
# =============================================================================

BOND_RATE_WEIGHTS = {
    "inflation_pressure": 0.22,      # CPI YoY momentum
    "real_yield_stress": 0.18,       # TIPS 10Y/5Y momentum + level bonus above 2%
    "yield_curve_shape": 0.18,       # 2Y-10Y spread momentum + butterfly curvature
    "policy_gap": 0.15,              # Fed funds vs. neutral rate gap
    "credit_market_stress": 0.15,    # HY OAS + IG OAS 4-week widening velocity
    "monetary_policy_stress": 0.07,  # M2 contraction + hiking pace
    "ppi_pressure": 0.05,            # PPI YoY momentum
}

DEFENSIVE_THRESHOLD = 60.0
WATCH_THRESHOLD = 55.0

# Monthly-native FRED series that need the two-step ffill resample below.
_MONTHLY_SERIES = {"cpi_core", "ppi", "m2", "fed_funds"}

# All FRED friendly-names this score needs (see FREDProvider.SERIES_MAP).
# baa10y is a long-history (1986+) fallback for credit_market_stress, since
# the ICE BofA OAS series (hy_oas/ig_oas) only go back to 2023-08-15.
_REQUIRED_SERIES = [
    "cpi_core", "ppi", "real_y10", "real_y5", "t10y2y",
    "y5", "y10", "y30", "hy_oas", "ig_oas", "m2", "fed_funds", "baa10y",
]


# =============================================================================
# Data fetching
# =============================================================================

def prefetch_bond_rate_fred_data(
    fred_api_key: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    use_cache: bool = True,
) -> Dict[str, pd.Series]:
    """
    Fetch every FRED series this score needs, once, for the full date range.

    Bond Rate v2's inputs are all macro (FRED) series, which the backtest
    engine's ExecutionContext cannot serve incrementally (it only wraps
    yfinance ticker history — see strategies/regime_adaptive_strategy.py's
    _fetch_regime_inputs_from_context for how the *old* MacroMom score works
    around this using Yahoo tickers instead). So this is meant to be called
    ONCE up front (in BondRateAdaptiveStrategy.__init__), with the strategy
    then slicing the result up to `context.date` at each step to avoid
    look-ahead bias — the same principle, different plumbing.

    THE CRITICAL BUG (documented in memory, must not be reintroduced):
    monthly-published series (CPI, PPI, M2, Fed Funds) must be resampled to
    weekly with `.ffill()`, not just `.resample('W-FRI').last().dropna()`.
    Without the ffill, weeks with no new monthly print get dropped entirely,
    so a series that looks weekly is actually only ~12 points/year — and
    `.pct_change(52)` silently computes a 52-*month* lookback instead of a
    52-*week* (1-year) one. This exact bug pattern already exists correctly
    handled elsewhere in this codebase (see RegimeDataFetcher.fetch_all's
    m2_yoy / real_rate handling in data/providers.py) — this function
    follows the same two-step pattern for every monthly series.
    """
    if start_date is None:
        start_date = datetime(2003, 1, 1)  # extra lookback for 20yr backtests + rolling windows
    if end_date is None:
        end_date = datetime.now()

    fred = FREDProvider(api_key=fred_api_key)
    if use_cache:
        fred = CachedProvider(fred, cache_dir=".data_cache")

    result: Dict[str, pd.Series] = {}
    for key in _REQUIRED_SERIES:
        try:
            if key in _MONTHLY_SERIES:
                # Step 1: fetch at native monthly frequency (no gaps).
                raw = fred.get_series(key, start_date, end_date, frequency="M")
                # Step 2: forward-fill onto a weekly index — THE fix.
                result[key] = raw.resample("W-FRI").ffill().dropna()
            else:
                # Daily-native series (yields, spreads) — weekly is fine as-is.
                result[key] = fred.get_series(key, start_date, end_date, frequency="W")
        except Exception as e:
            print(f"[BondRateScore] Warning: could not fetch '{key}': {e}")

    return result


# =============================================================================
# Component scores
# =============================================================================

def _yoy(series: pd.Series) -> pd.Series:
    """Year-over-year % change on a weekly series (52 weeks = 1 year, post-fix)."""
    return series.pct_change(52) * 100


def inflation_pressure_score(cpi_core: pd.Series) -> Optional[float]:
    """CPI YoY level (dominant) vs. ~2% target, plus 13-week acceleration.

    TUNING ITERATION 1 (8/29): originally a 50/50 level/momentum blend with
    the level saturating at 6% YoY. Backtest showed this under-reacted to
    2022 — CPI YoY was already ~7% in Jan 2022, but a 50/50 blend still let
    a flattening (non-accelerating) momentum term hold the composite score
    down for months. A *sustained* high print is itself the signal, not
    just an accelerating one — reweighted to 65/35 level/momentum and
    tightened the saturation band (5% instead of 6%) so a persistently hot
    CPI print pushes the score up faster even without further acceleration.
    """
    yoy = _yoy(cpi_core).dropna()
    if len(yoy) < 14:
        return None
    level_score = clamp((yoy.iloc[-1] - 2.0) / 3.0)
    accel = yoy.iloc[-1] - yoy.iloc[-13]
    momentum_score = clamp(0.5 + accel * 0.5)
    return 0.65 * level_score + 0.35 * momentum_score


def ppi_pressure_score(ppi: pd.Series) -> Optional[float]:
    """PPI YoY momentum — same shape as inflation_pressure, PPI leads CPI."""
    yoy = _yoy(ppi).dropna()
    if len(yoy) < 14:
        return None
    level_score = clamp((yoy.iloc[-1] - 1.5) / 5.0)
    accel = yoy.iloc[-1] - yoy.iloc[-13]
    momentum_score = clamp(0.5 + accel * 0.4)
    return 0.5 * level_score + 0.5 * momentum_score


def real_yield_stress_score(real_y10: pd.Series, real_y5: pd.Series) -> Optional[float]:
    """TIPS 10Y/5Y momentum + a level bonus once real yields clear 2%.

    Real yields rising fast = tightening financial conditions (bad for
    growth/duration assets). A sustained real yield above 2% has historically
    coincided with risk-off regimes (e.g. 2022-2023).
    """
    both = pd.concat([real_y10, real_y5], axis=1).mean(axis=1).dropna()
    if len(both) < 14:
        return None
    change_13w = both.iloc[-1] - both.iloc[-13]
    momentum_score = clamp(0.5 + change_13w * 0.4)
    if both.iloc[-1] > 2.0:
        momentum_score = min(1.0, momentum_score + 0.15)
    return momentum_score


def yield_curve_shape_score(t10y2y: pd.Series, y5: pd.Series, y10: pd.Series, y30: pd.Series) -> Optional[float]:
    """2Y-10Y spread momentum (fast flattening/inversion = bearish) plus
    butterfly curvature (2*y10 - y5 - y30) — an unusual hump/dip in the
    belly of the curve relative to its own recent range signals stress.
    """
    spread = -t10y2y.dropna()  # FRED's T10Y2Y is 10Y-2Y; flip sign to match "2Y-10Y" framing
    scores = []
    if len(spread) >= 14:
        change_13w = spread.iloc[-1] - spread.iloc[-13]
        curve_score = clamp(0.5 - change_13w * 0.25)
        if spread.iloc[-1] > 0:  # actual inversion (2Y > 10Y)
            curve_score = min(1.0, curve_score + 0.15)
        scores.append((curve_score, 0.65))

    curvature = (2 * y10 - y5 - y30).dropna()
    if len(curvature) >= 104:
        z = zscore(curvature.tail(104).values)[-1]
        curvature_score = clamp(0.5 + abs(z) * 0.15)
        scores.append((curvature_score, 0.35))

    if not scores:
        return None
    total_w = sum(w for _, w in scores)
    return sum(s * w for s, w in scores) / total_w


def policy_gap_score(fed_funds: pd.Series, cpi_core: pd.Series) -> Optional[float]:
    """Fed funds vs. an estimated neutral rate — asymmetric by design.

    NOTE: there's no real-time FRED series for r-star (the neutral rate).
    Approximated as: 0.5% assumed real neutral (roughly the Fed's own
    longer-run SEP dot) + a DAMPENED long-run inflation-expectations proxy.

    TUNING ITERATION 1 (8/29): fixed the inflation-expectations proxy (was a
    3-year trailing average of *realized* CPI, which rose in lockstep with
    2022's own inflation and muted the signal) — widened to 5 years and
    capped the swing to +/-1.5pp from a 2% anchor. Also switched the gap
    from signed to absolute, so being far *behind* the curve (funds near 0%
    while inflation ran hot in Jan-Feb 2022) would flag as bearish, not just
    being restrictive.

    TUNING ITERATION 2 (9/2): the absolute-value version overcorrected.
    Backtesting the full 20yr window showed the score running persistently
    elevated (0.7-0.8) throughout 2011-2014 — appropriate ZIRP stimulus
    during the post-GFC recovery, not a bear signal — because ANY large gap
    from "neutral" got flagged regardless of direction or context. Loose
    policy is only actually dangerous when inflation is ALSO elevated (the
    Fed is behind the curve); loose policy during low/normal inflation is
    just appropriate accommodation. Restrictive policy (funds above
    neutral) is treated as bearish regardless of inflation level, since
    tightening financial conditions is a more universal headwind for risk
    assets on its own. Net effect: still catches 2022 (loose + hot
    inflation), no longer false-flags 2011-2014 (loose + calm inflation).
    """
    if len(fed_funds) < 5:
        return None
    cpi_yoy = _yoy(cpi_core).dropna()
    if len(cpi_yoy) == 0:
        return None
    trailing_infl = cpi_yoy.tail(260).mean() if len(cpi_yoy) >= 52 else cpi_yoy.mean()
    if pd.isna(trailing_infl):
        trailing_infl = 2.0
    current_infl = cpi_yoy.iloc[-1]
    # TUNING ITERATION 3 (9/4): pure 5yr trailing average stayed artificially
    # elevated through 2023-2025 because 2022's spike was still inside the
    # window, making the Fed's deliberately-restrictive-but-working policy
    # (inflation actually cooling) look like a persistent warning sign.
    # Blending in the current reading lets "neutral" fall faster once
    # inflation genuinely normalizes, instead of staying stuck near its
    # clamped ceiling for years after a spike rolls out of the average.
    blended_infl = 0.5 * trailing_infl + 0.5 * current_infl
    anchored_infl = clamp(blended_infl, 0.5, 3.5)  # dampen how far "expectations" can drift from target
    neutral_rate = 0.5 + anchored_infl
    gap = fed_funds.iloc[-1] - neutral_rate

    if gap < 0:
        # Loose policy: only bearish if inflation is ALSO elevated (Fed
        # behind the curve). infl_elevation is 0 at/below 2.5% CPI YoY,
        # maxed by 5.5%+.
        infl_elevation = clamp((current_infl - 2.5) / 3.0)
        return clamp(0.5 + abs(gap) * 0.35 * infl_elevation)
    else:
        # Restrictive policy: bearish on its own (tightening financial
        # conditions is a real valuation headwind), but scaled down from
        # iteration 2's 0.30 - that, combined with the slow-to-adapt neutral
        # estimate, was flagging 2023-2025's soft-landing disinflation
        # (restrictive-but-working policy) almost as strongly as a genuine
        # warning. 0.20 still lets a truly extreme/sustained restrictive
        # stance reach DEFENSIVE, just not from a moderate, expected gap.
        return clamp(0.5 + gap * 0.20)


def _credit_market_stress_score_baa10y(baa10y: pd.Series) -> Optional[float]:
    """Fallback credit-stress score using the Baa-10Y corporate spread when
    the ICE BofA OAS series isn't available (pre-Aug-2023 — see
    credit_market_stress_score). Goes back to 1986 on FRED, so it covers
    2008 and every other backtest window this strategy is tested against.

    Level-dominant rather than momentum-dominant, unlike the OAS version:
    BAA10Y sits ~1.5-2.5pp in calm markets and spiked to 6.16 in Dec 2008,
    so the absolute level alone is highly informative (confirmed: it only
    hit 2.37 in Oct 2022 — barely above the 2019 calm-market average of
    2.23 — correctly reflecting that 2022 wasn't a credit-driven bear).
    """
    if len(baa10y) < 5:
        return None
    level = baa10y.iloc[-1]
    level_score = clamp((level - 2.0) / 2.5)  # ~2.0 = calm anchor, ~4.5+ = crisis-level, maxed
    widening = level - baa10y.iloc[-4] if len(baa10y) >= 5 else 0.0
    momentum_score = clamp(0.5 + widening * 0.5)
    return 0.7 * level_score + 0.3 * momentum_score


def credit_market_stress_score(
    hy_oas: Optional[pd.Series],
    ig_oas: Optional[pd.Series],
    baa10y: Optional[pd.Series] = None,
) -> Optional[float]:
    """HY + IG OAS combined, 4-week widening velocity. Widening spreads
    (investors demanding more compensation for credit risk) is one of the
    fastest-reacting stress signals available — this is the main component
    aimed at catching *short* bears the old score misses.

    Returns None (not 0.5) when there's insufficient history and no
    fallback, so the composite score renormalizes weight onto the other
    components instead of diluting toward a neutral value with this
    component's full weight still attached — see module docstring's
    "VALIDATION FINDING". FRED's BAMLH0A0HYM2/BAMLC0A0CM only go back to
    2023-08-15 in this environment; before that date, falls back to
    `baa10y` (Moody's Baa-10Y spread, 1986+) if provided, rather than going
    fully blind — see _credit_market_stress_score_baa10y.
    """
    if hy_oas is not None and ig_oas is not None:
        combined = (hy_oas + ig_oas).dropna()
        if len(combined) >= 5:
            widening = combined.iloc[-1] - combined.iloc[-4]
            return clamp(0.5 + widening * 0.15)
    if baa10y is not None:
        return _credit_market_stress_score_baa10y(baa10y)
    return None


def monetary_policy_stress_score(m2: pd.Series, fed_funds: pd.Series) -> Optional[float]:
    """M2 contraction (tightening liquidity) + hiking pace (13-week change
    in Fed funds). Low weight (7%) — slow-moving, mostly confirmatory.
    """
    scores = []
    m2_yoy = _yoy(m2).dropna()
    if len(m2_yoy) >= 1:
        scores.append((clamp(0.5 - m2_yoy.iloc[-1] * 0.05), 0.5))
    if len(fed_funds) >= 14:
        hike_pace = fed_funds.iloc[-1] - fed_funds.iloc[-13]
        scores.append((clamp(0.5 + hike_pace * 0.3), 0.5))
    if not scores:
        return None
    total_w = sum(w for _, w in scores)
    return sum(s * w for s, w in scores) / total_w


# =============================================================================
# Composite score
# =============================================================================

# Economic domain clusters — the fix for "diluted by an unrelated calm
# component" found while testing the BAA10Y credit fallback (8/31): a
# straight weighted average across all 7 components lets an accurately-calm
# component in one domain (e.g. credit_market_stress correctly showing 2022
# wasn't a credit event) drag the composite down even while a DIFFERENT
# domain (inflation) is screaming. Grouping into domains and taking the
# worst domain's own weighted average (see compute_bond_rate_score) means a
# severe bear in any one domain isn't hidden by calm in another.
COMPONENT_CLUSTERS = {
    "inflation": ["inflation_pressure", "ppi_pressure", "real_yield_stress"],
    "credit_growth": ["credit_market_stress", "monetary_policy_stress"],
    "rates_curve": ["yield_curve_shape", "policy_gap"],
}

# How much the composite score leans on the worst domain vs. the overall
# average. Pure max() (1.0) would react fastest but risks whipsawing on a
# single noisy component; blending in some overall-average weight keeps a
# bit of the original smoothing/robustness. Not swept/optimized — a
# reasonable starting point, backtest-validate before changing further.
CLUSTER_MAX_WEIGHT = 0.7


def compute_bond_rate_score(inputs: Dict[str, pd.Series]) -> Tuple[float, Dict[str, float]]:
    """
    Compute the 7-component Bond & Rate bear score.

    Args:
        inputs: dict of pd.Series, already sliced to "as of" a given date
                (see prefetch_bond_rate_fred_data + BondRateAdaptiveStrategy
                for how the point-in-time slicing is done in a backtest).
                Expected keys: cpi_core, ppi, real_y10, real_y5, t10y2y,
                y5, y10, y30, hy_oas, ig_oas, m2, fed_funds.

    Returns:
        (bear_score 0-100, component_scores dict of each 0-1 sub-score —
        only components with real data are included). The composite blends
        two things, both computed with BOND_RATE_WEIGHTS renormalized over
        whichever components are actually available (never diluted by a
        missing component defaulting to neutral):
        - the worst-scoring COMPONENT_CLUSTERS domain's own weighted average
          (catches a severe bear concentrated in one domain)
        - the overall weighted average across all available components
          (keeps some cross-domain smoothing/robustness)
    """
    raw: Dict[str, Optional[float]] = {}

    raw["inflation_pressure"] = (
        inflation_pressure_score(inputs["cpi_core"]) if "cpi_core" in inputs else None
    )
    raw["ppi_pressure"] = (
        ppi_pressure_score(inputs["ppi"]) if "ppi" in inputs else None
    )
    raw["real_yield_stress"] = (
        real_yield_stress_score(inputs["real_y10"], inputs["real_y5"])
        if "real_y10" in inputs and "real_y5" in inputs else None
    )
    raw["yield_curve_shape"] = (
        yield_curve_shape_score(inputs["t10y2y"], inputs["y5"], inputs["y10"], inputs["y30"])
        if all(k in inputs for k in ("t10y2y", "y5", "y10", "y30")) else None
    )
    raw["policy_gap"] = (
        policy_gap_score(inputs["fed_funds"], inputs["cpi_core"])
        if "fed_funds" in inputs and "cpi_core" in inputs else None
    )
    raw["credit_market_stress"] = credit_market_stress_score(
        inputs.get("hy_oas"), inputs.get("ig_oas"), inputs.get("baa10y")
    )
    raw["monetary_policy_stress"] = (
        monetary_policy_stress_score(inputs["m2"], inputs["fed_funds"])
        if "m2" in inputs and "fed_funds" in inputs else None
    )

    available = {k: v for k, v in raw.items() if v is not None}
    if not available:
        return 50.0, {}

    weight_sum = sum(BOND_RATE_WEIGHTS[k] for k in available)
    overall_avg = sum(available[k] * BOND_RATE_WEIGHTS[k] for k in available) / weight_sum

    cluster_scores = []
    for members in COMPONENT_CLUSTERS.values():
        cluster_available = {k: available[k] for k in members if k in available}
        if not cluster_available:
            continue
        cluster_weight_sum = sum(BOND_RATE_WEIGHTS[k] for k in cluster_available)
        cluster_scores.append(
            sum(cluster_available[k] * BOND_RATE_WEIGHTS[k] for k in cluster_available) / cluster_weight_sum
        )
    worst_cluster = max(cluster_scores) if cluster_scores else overall_avg

    bear_score = (CLUSTER_MAX_WEIGHT * worst_cluster + (1 - CLUSTER_MAX_WEIGHT) * overall_avg) * 100

    return bear_score, available


def classify_bear_type(scores: Dict[str, float]) -> str:
    """
    Classify current stress as 'inflation', 'recession', or 'mixed'.

    Thresholds are from the documented spec exactly (0.65 / 0.62 / +0.08 / +0.05).
    The inflation_signal / recession_signal groupings below are reconstructed —
    the spec names the threshold logic but not which components feed each
    signal, so this groups components by which regime they conceptually
    indicate. Validate against the 2022 backtest window (should classify as
    'inflation') as a sanity check.

    Uses only whichever components are actually present in `scores` (see
    compute_bond_rate_score's renormalization) — a component missing from a
    signal group just drops out of that group's average rather than forcing
    in a neutral 0.5.
    """
    inflation_components = [scores[k] for k in ("inflation_pressure", "ppi_pressure", "real_yield_stress") if k in scores]
    recession_components = [scores[k] for k in ("credit_market_stress", "monetary_policy_stress", "yield_curve_shape") if k in scores]

    inflation_signal = sum(inflation_components) / len(inflation_components) if inflation_components else 0.5
    recession_signal = sum(recession_components) / len(recession_components) if recession_components else 0.5

    if inflation_signal > 0.65 and inflation_signal > recession_signal + 0.08:
        return "inflation"
    elif recession_signal > 0.62 and recession_signal > inflation_signal + 0.05:
        return "recession"
    else:
        return "mixed"


def get_bond_rate_risk_level(bear_score: float) -> str:
    """Simple two-threshold classification, per spec: >60 defensive, >55 watch."""
    if bear_score > DEFENSIVE_THRESHOLD:
        return "DEFENSIVE"
    elif bear_score > WATCH_THRESHOLD:
        return "WATCH"
    else:
        return "LOW"


# Defensive ETF baskets by bear type — originally recovered verbatim from
# scripts/run_bear_profit_research.py's BASKETS dict ('Inflation Bear',
# 'Recession Bear', 'Full Defense'), cross-checked exact match on 8/25.
#
# TUNING ITERATION 2 (9/2): reweighted toward SH after finding the original
# weights only got the portfolio to "lose less," not "hold value," during a
# correctly-called bear. The research this basket came from (research_
# bear_profit.md) already showed SH was the ONE instrument positive across
# every historical bear type tested — including being the single best
# performer in the 2008 GFC (+89.4%) despite that being TLT's supposed
# specialty. The original weights (SH only 20-40%) under-used that finding
# in favor of "less bad" hedges (GLD, SHY) that mostly just avoid losses
# rather than generate them. Verified against actual 2022 ETF returns
# (GLD +0.8%, SHY -3.8%, SH +19.7%): the original inflation basket returned
# only +2.7% blended for the year despite SH alone returning +19.7% —
# confirms the original weighting was leaving real upside on the table.
# TLT is kept meaningful in the recession basket specifically (its own
# proven strength: +25.6% GFC, +10.5% COVID) rather than abandoned for SH.
DEFENSIVE_BASKETS = {
    "inflation": {"SH": 0.40, "GLD": 0.30, "SHY": 0.30},
    "recession": {"SH": 0.30, "TLT": 0.45, "GLD": 0.25},
    "mixed": {"SH": 0.55, "TLT": 0.25, "GLD": 0.20},  # "Full Defense" — safest all-weather choice
}


if __name__ == "__main__":
    print("=" * 70)
    print("BOND & RATE ADAPTIVE BEAR SCORE (v2)")
    print("=" * 70)

    print("\nFetching FRED data (this can take a minute on first run)...")
    data = prefetch_bond_rate_fred_data()

    score, components = compute_bond_rate_score(data)
    bear_type = classify_bear_type(components)
    risk_level = get_bond_rate_risk_level(score)

    print(f"\nBear Score: {score:.1f}/100")
    print(f"Risk Level: {risk_level}")
    print(f"Bear Type:  {bear_type}")
    print("\nComponent Scores (available only):")
    for factor, s in components.items():
        bar = "#" * int(s * 20) + "-" * (20 - int(s * 20))
        print(f"  {factor:25s} [{bar}] {s:.2f}  (weight {BOND_RATE_WEIGHTS[factor]*100:.0f}%)")

    print(f"\nSuggested defensive basket if triggered: {DEFENSIVE_BASKETS[bear_type]}")
