# Archive: Bond Rate Adaptive v2 (Sept 2026)

Archived when the approach was superseded by the work in `../../improvement/`.

## What this was

A regime-detection strategy meant to catch bears MacroMom (the live
`RegimeAdaptiveStrategy`) misses, using a 7-component bond/rate/inflation
score instead of VIX/yield-curve/breadth/momentum. Backtested well against
MacroMom on CAGR/MaxDD/Sharpe (see `charts/`), and confirmed via a control
variant (`RegimeAdaptiveImprovedExecution`, in `strategies/`) that the edge
came from the signal, not just from committing more capital to defense.

## Why it was superseded, not just refined

The approach used a **graduated allocation** (score maps to a HB/defensive
weight split, e.g. 60/40, 30/70) rather than a **binary bear/not-bear
classification** with full commitment. The actual goal is closer to: detect
bear regimes as early and precisely as possible, then commit fully — no
partial hedge. That reframes this from "tune the allocation table more" to
"design a proper regime classifier," which is what `improvement/` does.

The backtests here are still useful ground truth for comparison — this
approach's 2022 drawdown (~5-6%) and 2008 drawdown (~43%) are a real bar to
beat, not just a discarded attempt.

## Contents

- `data/bond_rate_score.py` — the 7-component score + FRED prefetch
- `strategies/bond_rate_strategy.py` — `BondRateAdaptiveStrategy` +
  `RegimeAdaptiveImprovedExecution` (the execution-only control variant)
- `backtest/run_bond_rate_backtest.py` — 3-way backtest runner
- `scripts/plot_bond_rate_comparison.py`, `plot_bond_rate_mechanism.py` —
  charting
- `scripts/spy_bear_analysis.py`, `spy_bear_combo_analysis.py`,
  `spy_bear_hunt.py`, `run_bear_hunt_backtest.py`,
  `run_bear_profit_research.py` — earlier signal/instrument research
  (recovered from a prior session's transcript after the originals were
  lost; see git log for that story) that fed into this strategy's
  defensive-basket choices
- `results/` — backtest output CSVs (2/10/20yr, all 3 strategies:
  portfolio values, score history, reposition log)
- `charts/` — final comparison and mechanism charts
