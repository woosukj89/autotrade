# Progress Log

## Iteration 22 (full redo: point-in-time data, daily cadence, real costs, no options) - MAJOR CORRECTION

Per explicit direction to fix 4 things before trusting any further result:
(1) find more reliable data and eliminate lookahead bias, (2) run at
daily cadence to match the live cron, (3) model real transaction costs
(Robinhood fees + slippage), (4) drop options entirely - plus build a
Robinhood-outage resilience test. This iteration is the full redo.

**(1) Point-in-time fundamentals via SEC EDGAR** (`edgar_pipeline.py`).
Found the underlying issue was worse than "yfinance is flaky": every
backtest this session (`HighBetaGrowthStrategy`/`QualityDefensiveStrategy`)
fetched fundamentals via `self._yahoo_provider.get_fundamentals_batch()`
directly, bypassing the point-in-time path (`_get_fundamentals`, already
present in `backtest.py`, filters `date <= simulated_date`) entirely - so
every historical backtest scored 2010-era stock picks using 2026
hindsight ROE/margins/growth. `data/edgar.py` already had an unused SEC
EDGAR ingestion pipeline built for exactly this; found and fixed THREE
real bugs in it before trusting the data:
  - `facts.tag` is already the resolved friendly concept name
    ('Revenue'), not the raw XBRL tag list - the original
    `isin(raw_tag_list)` match against it silently matched nothing.
  - `extract_tag_series()` returned only the FIRST XBRL tag with any
    data for the whole company and never merged across tags - since
    companies change tags over time (e.g. many switched revenue tags at
    the 2018 ASC 606 standard), this silently dropped most of a
    company's history whenever it had changed tags even once. Fixed by
    merging observations across all priority tags.
  - SEC's `fy` field labels which FILING contains an observation, not
    which period the value describes (a single 10-K includes prior-year
    comparatives, all stamped with the filing's own `fy`) - grouping by
    `fy` mixed annual/quarterly/restated values under one label
    (verified directly: produced Apple revenue numbers off by exactly 2
    fiscal years before the fix). Fixed by filtering to genuine ~365-day
    annual spans and keying by the actual period `end` date, taking the
    EARLIEST filing that disclosed each period (that's when it actually
    became knowable - not the latest, which is what a naive "most
    recent restatement" read would grab).
  Verified post-fix against known real Apple Revenue/NetIncome/Equity/
  TotalDebt figures for every year 2010-2025 - all correct. Ingested the
  424-ticker research universe (400 resolved via SEC's ticker list, 24
  not found - mostly delisted/acquired names like ATVI, BRK.B formatting),
  83,733 point-in-time fact-rows across 393 tickers.
  `PitFundamentalsStore`/`PitHighBetaGrowthStrategy`/
  `PitQualityDefensiveStrategy` (`pit_strategies.py`) consume this via an
  in-memory point-in-time lookup instead of the live yahoo shortcut -
  beta still comes from real price history (already point-in-time
  correct via `context.get_historical_prices`), only the fundamentals
  half changed.

**(2) Daily cadence** (`final_validated_comparison.py --time-period d`).
Matches the live cron's actual daily check, instead of the monthly
sampling every prior backtest this session used.

**(3) Real transaction costs** (`backtest.py`: new `slippage_bps` and
`regulatory_fees` params on `Backtest.backtest()`). Robinhood charges $0
commission on stocks, but real SEC Section 31 fees ($27.80/$1M proceeds)
and FINRA TAF ($0.000166/share, capped $8.30/trade) apply to every sell
regardless of broker. The dominant real cost is bid-ask spread, which the
existing flat-dollar `buffer_pricing` modeled poorly (a fixed dollar
amount is a wildly different fraction of a $20 stock vs a $900 stock) -
added a percentage-of-price `slippage_bps` model instead (5bps default,
a moderate assumption for liquid S&P-ish names). Both now tracked and
reported on `BacktestResult` (`slippage_cost`, `regulatory_fees`).

**(4) Options removed entirely.** `PitFactorRotationStrategy` has no
collar/hedge component - pure long-only equities, matching direction.

**(5) Outage-resilience simulator built** (`outage_strategy.py`,
`OutageSimulatingStrategy`) - wraps any strategy and goes "offline" for
scheduled windows (portfolio just sits and marks to market, no new
trades), modeling the observed Robinhood session-expiry issue. Not yet
run against the full 20yr window - next step.

**Full 20yr daily-cadence result, all 4 fixes combined**:

| Strategy | CAGR | MaxDD | Sharpe | Slippage | RegFees | Tax paid |
|---|---|---|---|---|---|---|
| PitHighBetaOnly (no timing) | 16.4% | 58.1% | 0.59 | $9,418 | $302 | $8,245,465 |
| PitQualityDefensiveOnly | 8.0% | 23.3% | 0.34 | $2,237 | $67 | $1,788,735 |
| PitFactorRotation[alltime_15] | 13.1% | 55.0% | 0.51 | $32,715 | $1,091 | $6,829,163 |
| PitFactorRotation[alltime_05] | 13.9% | 46.4% | 0.56 | $29,416 | $941 | $7,570,442 |
| PitFactorRotation[alltime_10] | 11.8% | 51.8% | 0.46 | $28,941 | $932 | $5,858,581 |

**This is a major correction to the entire session, not just a new data
point.** Every earlier headline number (HighBetaOnly 26.7-27.0% CAGR,
FactorRotation[alltime_15] 28.2% CAGR, the collar's 27.3%, etc.) was
generated with the lookahead bias described above still active. The
honest, point-in-time-correct number for the exact same unhedged
stock-picker is **16.4% CAGR**, roughly 10 points lower - hindsight
about which companies turned out to have great fundamentals was worth
far more to the backtested return than any of the market-timing/hedging/
rotation work this session actually tested. **Nothing from this session,
under honest assumptions, comes close to the CAGR>25.1% target anymore,**
let alone MaxDD<30%.

The rotation-vs-timing conclusion also flips again, consistent with (not
contradicting) two earlier findings this session (Step 3's original
"binary timing gives up upside" result, and the smoke-test tax finding
above): `PitFactorRotation` underperforms the plain `PitHighBetaOnly`
sleeve on CAGR in every one of the 3 definitions tested (11.8-13.9% vs
16.4%), with MaxDD only modestly better (46-55% vs 58%) - daily-cadence
regime switching now reacts to every real transition (not smoothed by
monthly sampling), incurring real slippage/regulatory/tax cost on a full
portfolio turnover each time, and this cost isn't fully recovered by the
defensive sleeve's lower volatility.

**Why MaxDD dropped less than CAGR** (58% vs the earlier 64-65%, a much
smaller change than CAGR's ~10-point drop): a plausible mechanism, not
fully isolated - today's-hindsight fundamentals likely over-selected
stocks that specifically turned out to be extreme winners (which are
also disproportionately the most volatile names), so removing that
hindsight reduces the return-inflation effect much more than it changes
the underlying risk profile of "high-beta tech/semis screen" as a
category.

**Not yet done**: the outage-resilience test (`--mode outage`) against
this corrected baseline - next step. Given the corrected numbers no
longer meet target by a wide margin, that test's finding will speak to
operational robustness, not to whether this specific approach is
deployable as-is - it currently doesn't clear the return bar honestly
measured, full stop.

Artifacts: `edgar_pipeline.py`, `pit_strategies.py`,
`pit_factor_rotation_strategy.py`, `outage_strategy.py`,
`final_validated_comparison.py`, `backtest.py` (slippage_bps/
regulatory_fees params), `edgar_universe.txt`,
`data/edgar_fundamentals.sqlite`,
`results/final_validated_{smoke_3yr,full_20yr}.json`.

---

## Iteration 21 (idea E+B combined, + root-caused stability investigation)

Per direction to "keep going and push further" after Iteration 20's
retraction. Combined idea E (factor rotation) with idea B (options
collar) - factor rotation's base MaxDD (~31-35%, see below) is a much
better starting point for a hedge overlay than idea B alone had
(HighBetaOnly's 64%), so it should need a far smaller, more realistic
hedge to close the gap.

**Root cause of the run-to-run noise found.** The repeated
`[YahooData] Fetching fundamentals for N tickers... Got fundamentals for 0
tickers` lines seen throughout this session's logs are yfinance's `.info`
endpoint silently failing an entire fetch batch - a known reliability
issue with that unofficial API under concurrent load
(`data/yahoo_data.py::get_fundamentals`, 15 parallel workers). This
changes which stocks clear `HighBetaGrowthStrategy`'s score>=50 filter
between runs. Confirmed the mechanism precisely: results are perfectly
deterministic WITHIN one script session (3 back-to-back reruns in
`stability_check.py` gave bit-for-bit identical 28.0%/31.0%/5.41, because
the sqlite fundamentals/beta caches stay populated for the whole session),
but differ BETWEEN separate sessions, because whatever partial data
yfinance happens to serve at the moment the cache is first populated gets
locked in until the cache expires. Not patched (`data/yahoo_data.py` is
shared with the live trader - out of scope for this research branch) but
now understood and worked around methodologically: don't trust a single
session's result this close to a boundary, sample across sessions
instead.

**FactorRotation[alltime_15] base (no hedge), 3 independent cache-state
draws**: CAGR is rock-solid (28.0%, 28.1%, 28.2% - a 0.2pp range,
comfortably above the 25.1% target every time). MaxDD is genuinely noisier
(31.0%, 33.5%, 35.1% - a 4.1pp range), consistent with MaxDD depending
more on which specific volatile stock was or wasn't in the 15-position
book during a specific crash week, while CAGR depends more on broad
tech/semis exposure that's robust to substitution among close-quality
candidates.

**Collar put15%/call8% (1x hedge ratio, skew=0.003 realistic) applied to
all 3 independent draws**:

| Base draw | Base MaxDD | Collared CAGR | Collared MaxDD | Meets target? |
|---|---|---|---|---|
| Original (Iter. 18) | 33.5% | 28.4% | 28.4% | YES |
| Rerun | 35.1% | 28.2% | 30.1% | NO (by 0.1pp) |
| Stability run 0-2 | 31.0% | 28.2% | 27.3% | YES |

**Honest verdict: this is the best-supported result of the entire
session, but it is a real edge case, not a clean pass.** CAGR clears the
target with no ambiguity in every draw. MaxDD, post-collar, lands right on
the 30% line - 2 of 3 independent draws clear it, the third misses by a
rounding error (30.1%). Given the demonstrated ~4pp natural MaxDD noise
in the base strategy alone (before any hedge), a result landing within
~1pp of the target on either side should be read as "at the boundary,"
not decisively resolved either way. This is nonetheless the most
practically credible candidate found all session: only a 1x hedge ratio
(not idea B's earlier 2-3x on the raw stock-picker), pure long-only
equities plus a modest, realistically-priced collar, CAGR far above
target with high confidence, and MaxDD that is very likely in the
low-to-high-20s to low-30s% range in practice - a large, real improvement
over every un-hedged approach tried (43-66% MaxDD), even if "definitely
under 30%" can't be claimed with full confidence given measurement noise
this session's data pipeline introduces.

Artifacts: `stability_check.py`,
`results/stability_run{0,1,2}_20yr.csv`, `results/factorrot_alltime15_rerun_20yr.csv`.

---

## Iteration 20 (validation) - Iteration 19's "meets target" claim RETRACTED

Per direction to validate the `vote_threshold=1` result before trusting it.
Two problems found, both real:

**1. The raw signal whipsaws daily, sampled on a fixed monthly grid.**
`classify()` with `vote_threshold=1` produces 616 transitions over the
20yr window with a MEDIAN EPISODE LENGTH OF 1 DAY (473 of 616 episodes
last <=3 days) - vs. 90 transitions for the original `vote_threshold=2`
base. Every backtest this session rebalances on a fixed monthly date (1st
business day of the month, `backtest.py`'s `FREQ_MAP['M']='MS'`), so the
strategy is sampling this chaotic daily-flipping signal at 240 fixed
points, not reacting to it continuously. The aggregate time-spent-
defensive is reasonably stable across different sampling-date offsets
(26-31% tested at +0/4/9/14/19 days), but which SPECIFIC months get
flagged almost certainly differs a lot between offsets given the 1-day
median episode length - meaning the specific reported number was likely
not reproducible.

**2. Confirmed directly: even the identical config re-run gives a
materially different result**, independent of point 1. Re-running the
EXACT same `vote_threshold=1` params (`validate_vote1.py`) gave **28.8%
CAGR / 33.6% MaxDD** - not the 26.2%/25.7% originally reported. This is
the same run-to-run non-determinism documented earlier this session
(yfinance fetch timing affects stock universe/selection each run), but
this is the first time it's been shown to swing MaxDD by ~8 points on an
identical config - large enough to flip a "meets target" claim into a
miss. Any single-run "meets target" claim near the boundary should be
treated with real skepticism from here on, not just this one.

**Stabilized version tested and still falls short**: fixed the whipsaw
directly (require the vote=1 trigger to persist `combo_confirm_days`=10,
`panic_cooldown_days`=15, `min_defensive_days`=10 - cuts transitions to
114, median episode length to 15 days, a genuinely stable signal, verified
via `classify()` before spending a full backtest run on it):

| Variant | CAGR | MaxDD | Sharpe | Trades |
|---|---|---|---|---|
| base (vote=2, reused alltime_15) | 28.3% | 34.5% | 5.42 | 1233 |
| vote=1, RAW (re-run of Iteration 19's headline) | 28.8% | 33.6% | 5.70 | 1189 |
| vote=1, combo_confirm=5 | 29.9% | 33.7% | 5.74 | 1194 |
| vote=1, combo_confirm=10 | 25.8% | 34.2% | 5.11 | 1141 |
| vote=1, stabilized (combo=10,cooldown=15,min_def=10) | 25.7% | 34.3% | 5.32 | 1236 |

**None of these meet CAGR>25.1% AND MaxDD<30%.** MaxDD clusters
consistently in the 33.5-34.5% band across every variant here regardless
of exactly how the vote threshold/persistence is tuned - a much tighter,
more consistent range than the single earlier outlier run suggested. This
looks like the real, repeatable floor for factor rotation on this
classifier family, not 25.7%.

**Corrected session state**: Iteration 19's "first strategy to meet the
target" claim is retracted - it did not survive validation. The honest
best results remain: idea B's skew-adjusted options collar (27.3% CAGR /
40.4% MaxDD, Iteration 17) for lowest validated MaxDD, and factor rotation
variants (26-30% CAGR / ~33-38% MaxDD across many configs and reruns,
Iterations 18-20) for best CAGR and Sharpe with no synthetic pricing
assumptions - genuinely the best approach found this session on a
risk-adjusted basis, just not confirmed to cross the strict <30% MaxDD
line. ~1200 trades over 20yr for the factor-rotation variants (~5/month)
is also a real, unmodeled cost this session's backtests don't price
(zero slippage/bid-ask spread by default - see Iteration 17's caveats,
same gap applies here).

Artifacts: `validate_vote1.py`.

---

## Iteration 19 (idea E tuning) - retracted, see Iteration 20 above

Insight: `multi_def_summary.json`'s classifiers were calibrated to minimize
false-positive rate for the BINARY strategy, where a false positive costs
100% of that period's upside (full exit to cash/SH). For factor rotation
a false positive only costs the gap between the two SLEEVES' returns - both
are still fully-invested equities (QualityDefensiveOnly alone still
compounds at 17.2% CAGR) - much cheaper. So a classifier tuned to avoid
false positives is very possibly UNDER-triggering for this strategy shape.
Hand-tested 7 variants of `alltime_15`'s calibrated params (`tune_factor_rotation.py`),
each run through the real 20yr backtest:

| Variant | CAGR | MaxDD | Sharpe |
|---|---|---|---|
| base (alltime_15, reused) | 28.0% | 33.7% | 5.39 |
| **vote_threshold=1** | **26.2%** | **25.7%** | 5.39 |
| combo_confirm_days=1,panic_cooldown=1 | 28.0% | 33.7% | 5.39 |
| vote_threshold=1,combo_confirm=1 | 28.1% | 31.9% | 5.60 |
| trend_confirm_days=10 | 28.0% | 33.7% | 5.39 |
| vote_threshold=1,trend_confirm=10,panic_cooldown=1 | 28.1% | 31.9% | 5.60 |
| require_credit_calm_to_exit=False | 27.4% | 35.0% | 5.25 |

**`vote_threshold=1` (any ONE of {fast_panic, trend_break, credit_stress,
breadth_stress} firing, persisted `combo_confirm_days`=2, otherwise
identical to `alltime_15`'s calibrated params) is the first and only
strategy this entire session to meet BOTH targets simultaneously: 26.2%
CAGR (>25.1% ✓) and 25.7% MaxDD (<30% ✓), 5.39 Sharpe.** Pure long-only
equities - no cash, no SH, no options/skew modeling required, unlike idea
B's closest attempt. The hypothesis was confirmed directly: lowering the
bar for "go defensive" from requiring 2-of-4 signals to just 1 (which
would have been a bad trade for the binary strategy - more false
whipsaws into 100% cash) is a net win here because whipsaws now cost the
much smaller aggressive-vs-defensive-sleeve return gap, not full market
exit.

`vote_threshold=1,combo_confirm=1` is an interesting near-miss worth
noting: 28.1% CAGR / 31.9% MaxDD / 5.60 Sharpe - higher CAGR and the best
Sharpe of the whole session, but MaxDD misses the <30% bar by 1.9pp.

**Caveat, stated plainly**: at `vote_threshold=1`, the classifier is no
longer really "detecting bears" in the strict sense it was calibrated
for - it spends much more total time in the defensive-sleeve state than
the actual historical bear-day fraction, since ANY single yellow flag
(even a brief VIX-adjacent panic blip or a short trend wobble) now
triggers a switch. That's fine and by design for this strategy shape
(the two sleeves are close enough in character that overreacting is
cheap), but it means this is better understood as "continuously tilt
toward quality/low-beta whenever any risk flag shows, revert when clear"
rather than "correctly identify bear markets" - a different, and for this
specific goal (CAGR + MaxDD), more effective framing than the one the
whole session started from.

Artifacts: `tune_factor_rotation.py`.

---

## Iteration 18 (idea E: factor rotation) - best result of the session

Per direction after the honest (skew-adjusted) idea B result fell short:
try factor rotation - always 100% invested in equities, but switch WHICH
stocks are held based on regime, instead of switching how much is invested
(idea A) or buying insurance (idea B).

**`quality_strategy.py` - `QualityDefensiveStrategy(HighBetaGrowthStrategy)`**:
reuses the exact same data pipeline (YahooDataProvider, beta calc, position
sizing) as the aggressive sleeve, but inverts the screen - low beta
(<=0.85, more points for <=0.65/<=0.45) instead of high, profitability/
balance-sheet/FCF quality still required (so this isn't just "buy whatever
has the lowest beta"), dividend-payer bonus, defensive sector scores
(staples/healthcare/utilities/gold miners) instead of tech/discretionary.
Standalone (always-on, no rotation) baseline: **17.2% CAGR / 32.8% MaxDD /
5.24 Sharpe** - confirms the quality/low-beta tilt genuinely has much less
drawdown even fully invested (32.8% vs. HighBetaOnly's 64.8%), not just a
noisy artifact - real information, just lower absolute compounding alone.

**`factor_rotation_strategy.py` - `FactorRotationStrategy`**: switches
between the aggressive (HighBetaGrowthStrategy) and defensive
(QualityDefensiveStrategy) sleeves using the exact same calibrated
`classify()` regime signal used for the binary approach - reused as-is,
not re-tuned, for the first pass. Full 20yr backtest, all 7 definitions:

| Strategy | CAGR | MaxDD | Sharpe | Return |
|---|---|---|---|---|
| HighBetaOnly (no timing) | 26.7% | 64.8% | 4.68 | 14073% |
| QualityDefensiveOnly (no timing) | 17.2% | 32.8% | 5.24 | 2659% |
| RegimeAdaptive (MacroMom, live) | 24.6% | 66.7% | 4.50 | 9827% |
| FactorRotation[alltime_05] | 27.0% | 32.3% | 5.49 | 14667% |
| FactorRotation[alltime_10] | 26.6% | 38.4% | 5.21 | 13717% |
| **FactorRotation[alltime_15]** | **28.2%** | **33.5%** | **5.44** | 17993% |
| FactorRotation[alltime_20] | 25.9% | 36.5% | 5.02 | 12266% |
| FactorRotation[rolling126_10] | 26.3% | 52.9% | 4.84 | 13040% |
| FactorRotation[rolling252_10] | 27.4% | 46.5% | 5.12 | 15795% |
| FactorRotation[rolling252_15] | 25.6% | 52.9% | 4.72 | 11705% |
| SPY (no tax) | 10.7% | 50.7% | — | 731% |

**Best result of the entire session, by a wide margin, with no synthetic
pricing assumptions required** (unlike idea B, which needed a Black-
Scholes/skew model to even evaluate): `FactorRotation[alltime_15]` hits
28.2% CAGR (clears the 25.1% target by 3.1pp, the highest CAGR found by
ANY approach all session) and 33.5% MaxDD (only 3.5pp above the <30%
target, vs. the collar's honest 40.4% or the graduated approach's 43.6%).
Sharpe (5.44) is also the best or near-best of anything tried.
`alltime_05` is comparably close (27.0%/32.3%). Both `rolling*` definitions
did notably worse on MaxDD (46.5-52.9%) - consistent with earlier findings
that the `rolling*` definitions' more frequent, shorter episodes don't
calibrate as cleanly as the `alltime_*` definitions.

**Still short of target, but for the first time by a small, plausible-to-
close margin rather than a structural one.**

Artifacts: `quality_strategy.py`, `factor_rotation_strategy.py`,
`run_factor_rotation_comparison.py`,
`results/factor_rotation_comparison_20yr.json` + per-strategy CSVs.

---

## Iteration 17 (idea B: options tail hedge)

`tail_hedge.py`: layers a rolling ~1-month SPY-based protective put / collar
on top of the already-backtested monthly NAV curves (no real historical
per-stock options data available, so SPY is used as the hedge underlying;
hedge notional is scaled by the portfolio's empirically estimated beta to
SPY - ~1.36 for HighBetaOnly, ~1.00-1.03 for the Graduated variants - to
partially correct for basis risk). Black-Scholes pricing, sigma=VIX/100 as
the 30-day IV proxy, flat r=3%. 25% tax on positive option P&L.

**First pass (flat IV, no skew) looked like a genuine win**: on the
HighBetaOnly base, a collar (long put + short call) financed largely by
the call leg, run at 3x the beta-implied hedge notional, put strike 10%
OTM / call strike 5% OTM, hit **28.3% CAGR / 28.4% MaxDD / 1.53 Sharpe** -
the first and only result this entire session to clear BOTH the CAGR>25.1%
and MaxDD<30% bars simultaneously.

**Caught before reporting it as real**: the base pricing used flat
sigma=VIX/100 for both the put and call legs. Real equity index options
trade with a volatility skew - OTM puts price richer than ATM, OTM calls
price cheaper (the "smirk") - which the flat-IV model ignores, making the
modeled put artificially cheap and the modeled call artificially rich
(overstating how much premium the short call actually finances). Added
`skew_per_pct_otm` to `simulate_overlay()` and reran as a stress test.

An unrealistically large skew value (0.015-0.03, i.e. 15-30 vol points of
skew on a 10% OTM strike) blew the whole simulation up (MaxDD into the
hundreds/thousands of percent, some runs NaN CAGR from negative NAV) -
initially alarming, but diagnosed as a bad stress-test parameter, not a
real flaw: that magnitude of skew is not physically realistic for SPX-
style options. Recalibrated to a defensible ballpark (~0.3-0.6 vol points
per 1% OTM, in line with typical SPX 25-delta risk-reversal levels) and
reran:

| Base | Skew stress | Best variant | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|---|
| HighBetaOnly | none (flat IV) | Collar put10/call5 x3.0 | 28.3% | 28.4% | 1.53 |
| HighBetaOnly | 0.003 (realistic) | Collar put10/call5 x3.0 | 27.3% | 40.4% | 1.28 |
| HighBetaOnly | 0.006 (elevated) | Collar put10/call5 x1.5 | 26.6% | 49.7% | 1.07 |
| Graduated[alltime_10] | 0.003 | Collar put10/call5 x3.0 | 24.7% | 42.5% | 1.24 |
| Graduated[alltime_15] | 0.003 | Collar put10/call5 x3.0 | 24.3% | 40.1% | 1.22 |

**Honest conclusion: the "meets target" result does not survive realistic
options pricing and should be discarded as a headline claim.** Under a
defensible skew assumption, the best finding is Collar[put10%/call5%,
3x hedge ratio] on the HighBetaOnly base: **27.3% CAGR / 40.4% MaxDD /
1.28 Sharpe** - this is nonetheless a real, load-bearing result: it is the
best CAGR found by ANY approach this entire session (beats the 27.0%
unhedged HighBetaOnly baseline slightly) while cutting MaxDD by 24 points
(64.0%→40.4%) at the same time - genuinely Pareto-better than doing
nothing, just not enough to clear the strict <30% bar. Idea A+B combined
(collar on top of an already-graduated/de-risked base) does NOT beat idea
B alone on the strongest base - Graduated[alltime_10]+collar tops out at
24.7%/42.5%, worse on both axes than HighBetaOnly+collar.

**Caveats not yet modeled** (flagged for the record, not swept under the
rug): no bid-ask spread / transaction cost, no margin/capital feasibility
check for a 3x-beta-notional options book (a real position of this size
relative to account equity may face real-world liquidity and margin
constraints not captured here), flat r=3% across a 20yr window that
actually ranged from ~0% to ~5%, and the skew stress test is a reasonable
ballpark, not a calibration against real historical SPX skew data.

Artifacts: `tail_hedge.py`.

---

## Iteration 14-16 (new target: CAGR>25.1% & MaxDD<30%): diversification, graduated allocation

Per direction after Step 3 concluded MacroMom wins by raw CAGR ("not good
enough... CAGR is too low"): new explicit numeric target, CAGR>25.1% AND
MaxDD<30% simultaneously, brainstorming mandate for approaches beyond more
bear-signal tuning. Options (puts/collars/hedges) explicitly approved as a
tool, alongside plain long-only stocks/ETFs. Priority ideas from user:
A (graduated allocation + better signals), B (tail hedge), D (redesign
stock-picker), E (factor rotation).

**Idea D probe — diversification alone** (`test_diversification.py`,
`HighBetaGrowthStrategy` alone, no timing, 3 concentration settings, 20yr):

| Variant | CAGR | MaxDD | Sharpe |
|---|---|---|---|
| default (15pos/50%sector/15%pos) | 26.8% | 64.7% | 4.72 |
| diversified (25pos/30%sector/8%pos) | 24.6% | 63.6% | 4.72 |
| very_diversified (35pos/20%sector/5%pos) | 23.0% | 58.9% | 4.67 |

Critical finding: the stock-picker ALONE (no market timing at all) beats
MacroMom on both CAGR and MaxDD. 20 years of MacroMom's timing overlay
added ~zero value on this window. Diversification alone trades CAGR for
MaxDD at a poor ratio (~9 points of CAGR given up per ~6 points of MaxDD
cut, worst case) and can't reach <30% MaxDD by itself even at 35 positions.

**Idea A — graduated (non-binary) allocation** (`graduated_strategy.py`
`GraduatedClassifierStrategy`, `compute_votes()` in `classifier.py`): scales
exposure by vote count (0-4 signals agreeing) instead of binary 100%/0%,
directly targeting Step 3's diagnosed flaw (binary gives up ALL upside on
every defensive day, including false positives and bear-rally chop).

First pass used the RAW daily vote count with `DEFAULT_EXPOSURE_MAP =
{0:1.00, 1:0.80, 2:0.55, 3:0.30, 4:0.15}`. Result was worse than expected -
lower CAGR AND lower MaxDD than the binary version of the *same* signal
(e.g. `rolling126_10`: graduated 21.7%/43.5% vs binary 23.7%/47.2%).
Diagnosis: the binary classifier's state machine has persistence gates and
hysteresis (`min_defensive_days`, run-length confirmation) layered on top
of these same votes; feeding raw unsmoothed daily votes into exposure
directly reacts to far more transient single-day noise than the binary
version ever fully exited on - more frequent partial de-risking for less
payoff, not genuinely less risk.

**Fix: smooth the vote signal** (`compute_votes(..., smooth_days=10)`,
10-day rolling mean on the vote count before mapping to exposure). Rerun,
same 20yr backtest, same methodology:

| Strategy | CAGR | MaxDD | Sharpe | Return |
|---|---|---|---|---|
| HighBetaOnly (no timing) | 27.0% | 64.0% | 4.72 | 14762% |
| RegimeAdaptive (MacroMom, live) | 22.1% | 66.0% | 4.16 | 6368% |
| Graduated[alltime_05] | 23.4% | 42.1% | 4.87 | 8082% |
| **Graduated[alltime_10]** | **24.4%** | **43.6%** | **5.02** | 9583% |
| Graduated[alltime_15] | 24.0% | 42.1% | 4.90 | 8955% |
| Graduated[alltime_20] | 22.7% | 50.3% | 4.66 | 7134% |
| Graduated[rolling126_10] | 23.0% | 56.8% | 4.53 | 7534% |
| Graduated[rolling252_10] | 22.6% | 44.4% | 4.63 | 6975% |
| Graduated[rolling252_15] | 23.7% | 63.8% | 4.50 | 8421% |
| SPY (no tax) | 10.7% | 50.7% | — | 731% |

(RegimeAdaptive/HighBetaOnly figures shift slightly run-to-run - confirmed
non-determinism from yfinance fetch timing affecting stock selection, not
a real difference in the strategies themselves.)

Smoothing worked as diagnosed: every `alltime_*` definition's CAGR jumped
2-3pp and Sharpe improved (Graduated[alltime_10] now has the best Sharpe of
any strategy tested, 5.02). But it's a genuine trade-off, not a free
lunch: 2 of 7 definitions (`rolling126_10`, `rolling252_15`) got *worse*
MaxDD after smoothing (43.5%→56.8%, 44.8%→63.8%) - delaying the exposure
cut to filter noise also delays reaction to a real fast selloff on those
definitions' more frequent, shorter rolling-peak episodes.

**Still short of target, and a consistent pattern is now visible**: across
every equity-only approach tried this session - binary classifier,
diversification alone, graduated/smoothed classifier - MaxDD keeps landing
in the low-40s to mid-60s%, never below ~42%. That's not a tuning gap
anymore, it looks like a structural floor for "long-only stocks +
market-timing" against this stock-picker's inherent volatility (individual
holdings run beta 1.2-2.2). No amount of exposure-curve or classifier
tuning has broken meaningfully below it. This is the case for idea B (tail
hedge via options) next - real convexity (not just reduced equity
exposure) is what a strict 30% MaxDD ceiling while retaining most upside
actually requires.

Artifacts: `classifier.py::compute_votes`, `graduated_strategy.py`,
`run_graduated_comparison.py`, `test_diversification.py`,
`results/graduated_comparison_20yr.json` + per-strategy CSVs.

---

## Iteration 11-13 (reframe, SPEC.md §0): multi-definition ground truth → real portfolio growth

Per explicit direction after the classifier-metric ceiling: stopped treating
one fixed bear definition + classifier metrics as the end goal, and ran the
actual 3-step process end to end.

**Step 1 — several bear definitions** (`ground_truth_v2.py`), all mechanical:
`alltime_{05,10,15,20}` (the original all-time-high peak-to-trough rule at
4 thresholds) and `rolling{126,252}_{10,15}` (a genuinely different style -
peak is a trailing-window high, not all-time, so episodes reset faster and
are more frequent/shorter - the "define your own" addition). Episode counts
ranged 3 (alltime_20, the original strict rule) to 42 (rolling126_10).

**Step 2 — per-definition classifier calibration** (`sweep_multi_def.py`,
relaxed to 80%/<5%/>5% per direction), then a further 25,000-combo targeted
sweep on the two closest (`rolling252_15`, `rolling126_10`). Best achieved:
coverage 80-85%, FPR 6.3-6.7% (still short of <5%, but far closer than the
original 20%-only ground truth ever got), defensive return solidly positive
(5-24%). Full results: `results/multi_def_summary.json` +
`results/multi_def_<name>_top10.json`.

**Step 3 — actual portfolio backtest, the real arbiter** (`portfolio_strategy.py`
wires the winning classifier per definition into a genuine 100%-aggressive-
or-100%-defensive `Strategy`, no partial allocation; `run_portfolio_comparison.py`
runs all 7 through the real 20-year backtest engine, 25% tax modeled, same
methodology as every other backtest in this repo):

| Strategy | CAGR | MaxDD | Sharpe | Total Return |
|---|---|---|---|---|
| Binary[alltime_05] | 20.7% | 41.3% | 4.32 | 5051% |
| Binary[alltime_10] | 19.2% | 41.3% | 3.91 | 3835% |
| Binary[alltime_15] | 21.3% | 47.6% | 4.22 | 5593% |
| Binary[alltime_20] | 17.8% | 47.6% | 3.63 | 3000% |
| **Binary[rolling126_10]** | **23.7%** | 47.2% | **4.43** | 8479% |
| Binary[rolling252_10] | 23.1% | 46.9% | 4.40 | 7653% |
| Binary[rolling252_15] | 22.7% | 47.2% | 4.26 | 7093% |
| **RegimeAdaptive (MacroMom, live, unmodified)** | **25.1%** | 66.1% | 4.51 | 10734% |
| SPY (no tax) | 10.7% | 50.7% | — | 731% |

**Honest result: by the user's own stated criterion ("the one that
maximizes growth is the winner"), the live, unmodified MacroMom strategy
wins.** Every one of the 7 binary classifier strategies has dramatically
better drawdown control (41-48% MaxDD vs. MacroMom's 66%) but LOWER raw
CAGR - none of the extensive signal engineering across 200,000+ combined
swept parameter combinations (this session, across the original 20%-ground-
truth work and this multi-definition work) produced a binary defensive
strategy that out-compounds the existing aggressive strategy over this
specific 20-year window.

**Why, mechanically**: a 100%-aggressive/100%-defensive strategy captures
*zero* upside on every day it's classified defensive - including every
false positive (6-15% of non-bear days, depending on definition) and every
correctly-called bear day the market happened to rally within (bear-market
rallies are real and frequent). MacroMom's graduated allocation (rarely
below 85% aggressive even when "cautious") keeps compounding through
almost all of that, and this specific 20-year window was dominated by an
extraordinary, concentrated tech/semis bull run (2023-2025 especially) -
the opportunity cost of any full exit, even a well-timed one, is unusually
high in a window this strong. A lower-volatility strategy will generally
show lower CAGR than a higher-volatility one that never quite got
punished badly enough for its drawdowns to outweigh its foregone upside -
that's what happened here, consistently, across every ground-truth
definition tried.

**Not a wasted result - a real, load-bearing finding**: binary (no-mix)
allocation and "maximize growth" are in genuine tension when the
underlying aggressive strategy is this strong. `Binary[rolling126_10]` is
the best of the 7 (23.7% CAGR, only 1.4pp behind MacroMom, comparable
Sharpe at 4.43 vs 4.51) while cutting max drawdown by 19 points (47.2% vs
66.1%) - a legitimate, different answer if risk reduction is valued
alongside growth, just not the winner under the literal maximize-growth
rule as stated.

Full artifacts: `results/portfolio_comparison_20yr.json` +
per-strategy CSVs, `portfolio_strategy.py`, `run_portfolio_comparison.py`.

---

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
