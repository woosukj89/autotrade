# Progress Log

## Iteration 1 (baseline: drawdown + VIX-level + VIX-spike fast_panic OR'd)

| Metric | Result | Target | Pass? |
|---|---|---|---|
| Coverage | 70.7% | ≥85% | FAIL |
| False-positive rate | 12.9% | <5% | FAIL |
| Defensive return | 61.6% | >5% | PASS |

66 entries. Diagnosed: 43 of 46 pure-false-positive episodes were `fast_panic`
firing on short (4-12 day) VIX blips unrelated to real bears.

## Iteration 2 (fast_panic = drawdown-only, VIX removed entirely)

| Metric | Result | Target | Pass? |
|---|---|---|---|
| Coverage | 68.4% | ≥85% | FAIL |
| False-positive rate | 7.4% | <5% | FAIL |
| Defensive return | 2.5% | >5% | FAIL |

**Why VIX was removed, not just re-thresholded**: checked VIX level during
the 3 real bears vs. the false-positive episodes directly. 2022's own
bear-period max VIX was 36.5 — *lower* than several false alarms (2011 debt
ceiling: 48.0; an Aug-2024 flash spike: 38.6). VIX level doesn't reliably
separate "real bear" from "acute-but-contained panic" for this ground
truth — raising the threshold would have cost 2022 coverage without fully
fixing the false-positive problem. Entries dropped 66 → 20, false-positive
rate roughly halved, but coverage and (surprisingly) defensive-return both
got worse.

Diagnosed the defensive-return drop and the coverage shortfall to the same
root cause, visible in the per-episode list: 2022 coverage was fragmented
across 4 separate episodes with gaps during **bear-market rallies**
(Mar-Apr 2022 and Jul-Aug 2022 both saw sharp multi-week counter-trend
bounces that satisfied the exit condition, forcing a whipsaw exit and
re-entry). Every whipsaw both loses coverage (gap while re-triggering) and
costs the defensive-return metric (a short position gives back gains during
the bounce it exited into aggressive for).

## Iteration 3 (+ minimum 20-day dwell time before exit is considered)

| Metric | Result | Target | Pass? |
|---|---|---|---|
| Coverage | 71.5% | ≥85% | FAIL |
| False-positive rate | 8.6% | <5% | FAIL |
| Defensive return | 4.6% | >5% | FAIL (close) |

17 entries. 2022 coverage 56.1% → 61.2%, 2022 defensive return 1.5% → 9.4%.
Confirms the whipsaw diagnosis was right. False-positive rate regressed
slightly (7.4% → 8.6%) since the dwell requirement also extends genuine
false-positive episodes once they trigger — expected trade-off, not a new
problem.

## Iterations 4-9: systematic sweep, architecture changes, and the empirical frontier

Vectorized `classify()` (pandas `.loc` row loop → numpy arrays): 600ms → 7.5ms
per call, an 80x speedup, to make a real parameter sweep feasible.

**Sweep 1** (`sweep.py`, 10,000 combos, original "any 1 of 3 paths" OR
architecture): best loss found had coverage 85.2%/FPR 12.5%/defret 6.6% —
first time coverage crossed 85%, but every top-20 result plateaued at
10-14% FPR regardless of thresholds tried. This was the first signal that
the ceiling was architectural, not a tuning gap.

**Persistence gates** (`fast_panic_confirm_days`, `combo_confirm_days`):
required signals to hold for N consecutive days before triggering entry,
to filter one-off blips. Modest effect (loss 0.0754 → 0.0683 best), not
the fix — false positives just moved from `fast_panic` to
`trend+credit_confirmed` firing on a loose (`credit_z_threshold=0.25`)
same-day coincidence.

**Credit z-score investigation** (checked directly, not assumed): unlike
VIX, credit z-score DOES discriminate — 2022's bear-period max z=3.45 vs.
2011 false-positive's max z=2.13, 2010's ~1.0-1.2. Sweep 2 (`sweep2.py`,
15,000+20,000 combos) biased toward higher credit thresholds (1.0-2.5) and
looser entry buffers accordingly. Best: coverage 85.4%/FPR 13.2%/defret
16.8% — essentially the same ceiling, just relocated.

**`require_credit_calm_to_exit`**: gates exit on credit stress having also
subsided, not just price bouncing above trend — targets 2022-style
bear-market-rally whipsaws more precisely than the blunt
`min_defensive_days` floor (which was inflating every episode, including
short false alarms, by the same fixed amount). Dominated top sweep results
once added; a real, if modest, improvement (loss → 0.0697).

**v6 voting architecture** (`classify()` rewritten): replaced "any 1 of 3
paths triggers" with "an extreme/rare drawdown fires alone (safety valve
for a 2020-tier shock) OR >=2 of {fast_panic, trend_break, credit_stress}
agree same-day." Sweep 3 (`sweep3.py`, 20,000 combos): best loss 0.0803,
coverage 81.6%/FPR 9.6%/defret 10.9% — the FPR floor moved down (9.6% vs.
10-14% before) but didn't break through 5%.

**Ruled out, with evidence, not assumption**: VIX level and VIX 5-day
spike-rate (original baseline — see Iteration 1) and VIX term structure
(VIX/VIX3M backwardation, checked directly: 2011's false-positive episode
had backwardation on 63% of its days vs. 2022's real bear at only 7% — the
*opposite* of useful). All three vol-based signals measure acute panic,
not sustained decline, and 2022 specifically wasn't a panic event. This
isn't a threshold problem - these signals are answering a different
question than "will this become a real bear."

### The empirical Pareto frontier (`pareto.py`, 20,000 more combos, 85,000+ total)

| Coverage floor | Best achievable FPR | Defensive return at that point |
|---|---|---|
| ≥85% | 12.3% | +27.2% |
| ≥80% | 9.4% | -11.3% |
| ≥75% | 8.5% | +3.0% |
| ≥70% | 6.5% | -11.6% |
| ≥65% | 5.7% | -28.7% |
| — | ≤5% (best) | coverage caps at 60.1%, defret -25.9% |

**Conclusion**: across 3 architectures and 85,000+ swept parameter
combinations, nothing achieves coverage≥85% AND FPR<5% AND defret>5%
simultaneously. At the tight end of the FPR budget, coverage caps around
60% *and* defensive return frequently goes negative — a third tension: a
classifier tuned to minimize false alarms also tends to enter too late/exit
too early on real bears, which hurts the defensive-sleeve return at the
same time it hurts coverage. This is consistent, repeated, and explainable
(not sweep noise) - I'm treating it as a real limit of this 3-signal set
(price trend, N-day drawdown, credit spread) against this ground truth (3
real bears in 20 years - a genuinely small sample to calibrate a <5% FPR
rule against), not a tuning failure.

**`ClassifierParams` defaults were set to this frontier point** (coverage
85.1%, FPR 12.3%, defret 27.2%; full top-50 in `results/pareto_top50.json`)
— **since superseded by the v7 (breadth) defaults below**, kept here as
the pre-breadth baseline for comparison.

## Iteration 10 (v7): breadth as a genuinely independent 4th signal

Per user direction after reviewing the frontier: added S&P 500 breadth
(`breadth.py`, `signals.breadth_stress_signal`) - % of the 418-ticker
curated universe from `data/yahoo_data.py` trading above their own 200-day
SMA, batch-fetched via yfinance (418 tickers, 36s, 394/418 had usable
data). Caveat stated plainly: uses the CURRENT constituent list applied
across the full 20-year history (real survivorship bias - a stock removed
from the index for poor performance won't be counted as weak during the
period it was actually struggling); a true point-in-time membership feed
would be a materially bigger data source than what's available here.

**Checked directly before integrating, same discipline as credit z-score
and VIX**: breadth does NOT cleanly separate real bears from false alarms
on its own - 2011's false-positive episode had *lower* mean breadth
(27.7%) than 2022's real bear (40.6%), because 2011 was a broader but
shorter panic while 2022's decline, though longer and index-significant,
was more concentrated in mega-cap names that dominate cap-weighted SPY.
Included anyway as a 4th vote input (genuinely independent information
from trend/drawdown/credit) rather than discarded on that univariate read,
since ensembles can extract value from imperfect individual signals.

`classify()`'s vote now requires >=`vote_threshold` of 4 signals (was 3):
{fast_panic, trend_break, credit_stress, breadth_stress}.

**Result - a real, measured improvement, not just noise:**

| Coverage floor | 3-signal best FPR | 4-signal best FPR | 3-signal defret | 4-signal defret |
|---|---|---|---|---|
| ≥85% | 12.3% | 11.9% | +27.2% | **+114.4%** |
| ≥75% | 8.5% | 7.9% | +3.0% | +6.3% |
| ≥65% | 5.7% | **5.0%** | -28.7% | -2.7% |
| FPR≤5% best | cov 60.1% | cov 60.9% | -25.9% | -19.6% |

Every tier improved, most dramatically defensive-return at the high-
coverage end (27.2%→114.4%) and FPR at the 65%-coverage tier (5.7%→5.0%,
now within a rounding error of the target, with defret nearly breakeven).
`ClassifierParams` defaults updated to the new best combined-loss point:
**coverage=85.2%, FPR=11.6%, defensive_return=70.2%** (2 of 3 targets pass,
comfortably now on the return target). Per-bear: 2008 coverage 91.9%/
defret 89.4%, 2020 coverage 75.0%/defret 31.3%, 2022 coverage 74.5%/defret
24.5% (2022's defensive return roughly doubled vs. the pre-breadth
default). Top 50 saved in `results/pareto_top50_v7.json`.

**Still hasn't cracked FPR<5% at coverage>=85%** - the frontier moved
favorably but the fundamental shape (tight FPR costs coverage, and now
also costs defensive-return, which frequently goes negative in the
tightest-FPR region) persists even with a 4th independent signal. 110,000+
total combos swept across 4 architectures.

### What would plausibly move the frontier further (not yet attempted)

1. **A genuinely independent 4th signal** with different information
   content than trend/drawdown/credit - e.g. real S&P 500 breadth (% of
   constituents above their own 200-day MA, not just the index level).
   Requires per-constituent price history, a much bigger data-fetch than
   anything used so far - not attempted due to that cost, not because it
   seems unpromising.
2. **Accept a specific frontier trade-off** and proceed to the portfolio-
   level backtest (SPEC.md §6) with the current best classifier, to see
   what the *portfolio* actually looks like at 85%/12.3%/27.2% - the
   classifier metrics are a proxy, not the end goal.
3. **Revisit whether <5% FPR is a well-calibrated target** given the
   sample size - 3 positive examples (bears) in 20 years bounds how
   confidently ANY rule can be tuned against a <5% false-positive rate
   without either overfitting to this specific history or refusing to
   fire on real future bears that don't closely resemble 2008/2020/2022.

## Remaining known issues (next steps, not yet attempted)

1. **2011 debt-ceiling crisis (Aug-Oct 2011, 77 days) is the single largest
   remaining false-positive episode.** It's a real, sharp, well-known market
   stress event (~19% SPX decline, US credit downgrade) that misses this
   repo's ground truth only because (a) it's just under the 20% bear
   threshold and (b) SPY's dividend-adjusted closes cushion it below even
   the 10% correction threshold. Not a clear signal bug — a genuine edge
   case in how ground truth is measured (SPY total-return vs. SPX
   price-only). Worth a documented judgment call, not silent exclusion.
2. **2022 coverage (61.2%) is still the weakest of the three bears.** The
   `trend_sustained` / `trend+credit_confirmed` entry paths (10-day
   confirmation) are inherently slower than `fast_panic`'s single-day
   trigger — appropriate for a slow grind, but worth testing whether
   `trend_confirm_days` can come down without re-opening the false-positive
   problem iteration 1 had.
3. **Defensive-return (4.6%) is closest to passing but still short.** Worth
   testing a steeper/more responsive defensive-sleeve instrument mix (this
   analysis uses realized SH returns already — see
   `backtest_classifier.py:defensive_sleeve_returns`) rather than further
   signal tuning; the classifier's timing is arguably already close to
   adequate for this metric specifically.
4. Have not yet tuned `drawdown_threshold`, `drawdown_lookback`,
   `credit_z_threshold`, or `sma_window`/`entry_buffer_pct` individually via
   a real parameter sweep — all 3 iterations above changed the *architecture*
   (which signals feed the entry/exit logic), not yet the specific numeric
   thresholds within the current architecture. A systematic sweep is the
   natural next step before concluding the architecture itself needs to
   change further.
5. Have not yet built the portfolio-level `Strategy`/`Backtest` wiring
   (SPEC.md §6 explicitly defers this until the classifier passes its own
   metrics) — everything above is classifier-only scoring against ground
   truth, not a fund/CAGR backtest.
