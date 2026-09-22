"""
Strategy-agnostic macro-context snapshot for the live report email.

Verified during the live-deployment planning pass that the LIVE trading
path does not actually fetch credit spread, true market breadth, or
FRED-sourced yield data today - RegimeAdaptiveStrategy's own live regime
inputs come only through ExecutionContext.get_historical_prices()
(yfinance SPY/^VIX/^TNX/^IRX), since the live ExecutionContext classes
expose no FRED access. This module is a new, independent fetch using the
already-fetched-in-research RegimeDataFetcher (data/providers.py), used
purely for REPORT DISPLAY - never as a trading signal for either live
driver. FRED_API_KEY is already a wired GitHub Actions secret and
fredapi is already a dependency, so no new credentials are needed.

Must never block or crash a trading run: every function here is safe to
call and returns None on any failure rather than raising.
"""
from typing import Optional, Dict

from data.providers import RegimeDataFetcher


def fetch_raw_macro_metrics(fred_api_key: Optional[str] = None) -> Optional[Dict[str, float]]:
    """Latest VIX / VIX3M / credit spread / breadth% / 10Y-3M yield spread,
    for informational display only. Returns None on any failure (network,
    missing FRED key, etc.) - callers should treat that as "omit the
    macro context section this run", not an error.
    """
    try:
        fetcher = RegimeDataFetcher(fred_api_key=fred_api_key, use_cache=True, cache_hours=24)
        series = fetcher.fetch_all()

        def _last(key):
            s = series.get(key)
            if s is None or len(s) == 0:
                return None
            try:
                return float(s.dropna().iloc[-1])
            except Exception:
                return None

        vix = _last('vix')
        vix_3m = _last('vix_3m')
        credit_spread = _last('credit_spread')
        pct_above_200dma = _last('pct_above_200dma')
        y10 = _last('y10')
        y3m = _last('y3m')
        yield_curve_10y3m = (y10 - y3m) if (y10 is not None and y3m is not None) else None

        metrics = {
            'vix': vix,
            'vix_3m': vix_3m,
            'credit_spread': credit_spread,
            'pct_above_200dma': pct_above_200dma,
            'yield_curve_10y3m': yield_curve_10y3m,
        }
        # If EVERY metric failed to fetch, treat the whole snapshot as
        # unavailable rather than showing an all-N/A panel.
        if all(v is None for v in metrics.values()):
            return None
        return metrics
    except Exception as e:
        print(f"[macro_snapshot] Could not fetch macro metrics (non-fatal, report will omit this section): {e}")
        return None


def format_macro_text_block(macro_metrics: Optional[Dict[str, float]],
                             bear_magnitude_pct: Optional[float] = None,
                             time_to_correction: Optional[str] = None) -> list:
    """Plain-text rendering of the macro snapshot, shared by both live
    drivers' save_report_to_file() so the two render identically instead
    of drifting. Returns a list of lines (no trailing newlines)."""
    if not macro_metrics:
        return []

    def _fmt(key, suffix="", decimals=1):
        v = macro_metrics.get(key)
        return f"{v:.{decimals}f}{suffix}" if v is not None else "N/A"

    lines = []
    lines.append("-" * 70)
    lines.append("MACRO CONTEXT")
    lines.append("-" * 70)
    lines.append(f"VIX / VIX3M:              {_fmt('vix')} / {_fmt('vix_3m')}")
    lines.append(f"Credit Spread (HY-IG):    {_fmt('credit_spread', '%', 2)}")
    lines.append(f"Breadth (% above 200DMA): {_fmt('pct_above_200dma', '%')}")
    lines.append(f"10Y-3M Yield Spread:      {_fmt('yield_curve_10y3m', '%', 2)}")
    if bear_magnitude_pct is not None or time_to_correction is not None:
        mag_str = f"{bear_magnitude_pct:.0f}%" if bear_magnitude_pct is not None else "N/A"
        ttc_str = time_to_correction or "N/A"
        lines.append(f"Est. Bear Magnitude:      {mag_str}")
        lines.append(f"Est. Time to Correction:  {ttc_str}")
    lines.append("(Informational context only - not necessarily a trading signal.)")
    lines.append("")
    return lines
