# Spec: Fast, Precise Bear-Market Detection with Binary Allocation

## 0. Reframe (9/17) — after 110,000+ combos hit a real ceiling

Sections 1-6 below are the original spec and stand as the methodology for
Steps 1-2. What changed: after exhausting the fixed 20%-rule ground truth
(only 3 true bears in 20 years — provably too small a sample to calibrate
a <5% false-positive classifier against, confirmed empirically across 4
signal architectures and 110,000+ swept parameter combinations — see
PROGRESS.md), the definition of "bear" itself is now a variable to search
over, not a fixed constant. New process:

1. **Define bear — several candidate definitions, not one.** Still
   mechanical/reproducible (no hand-picked dates), but now parameterized:
   drawdown threshold (try 5/10/15/20%) x peak style (all-time high vs. a
   rolling-window high, which resets faster and produces more frequent,
   shorter, more tactically-relevant episodes). See `ground_truth_v2.py`.
2. **Per definition, find a classifier** meeting a *relaxed* bar — 80%
   coverage, <5% FPR (was 85%/<5%) — using the same signal/classifier
   toolkit already built (`signals.py`, `classifier.py`, `sweep.py`).
3. **Portfolio-level backtest, not just classifier metrics.** Wire the
   winning classifier per definition into an actual `Strategy`
   (`backtest/backtest.py`'s engine, same pattern as the archived
   `BondRateAdaptiveStrategy`) — 100% aggressive or 100% defensive, no
   partial allocation, exactly per the original binary-allocation
   direction. Run the real 20-year backtest. **The definition+classifier
   combination that maximizes actual portfolio growth wins** — coverage/FPR
   are now a *filter* (only sufficiently-good classifiers are worth
   backtesting), not the final objective.

---

## 1. Context — why this exists

The prior approach (`archive/bond_rate_v2/`) mapped a continuous bear score to a
*graduated* allocation (e.g. score 65 → 60% aggressive / 40% defensive). That
solved the original ask ("don't lose value in a caught bear") reasonably well
for 2022 but left two open problems:

1. Partial hedging means even a "correctly caught" bear still bleeds money
   from whatever fraction stayed in high-beta stocks.
2. The signal's speed and precision were never measured directly — only
   inferred from equity-curve outcomes on 3 cherry-picked bear windows.

**This is a reset, not a refinement.** New goal, from the user directly:

> The ideal goal for the strategy is to correctly shift to safer assets (or
> even hold cash) during bears and invest in high betas otherwise. If I see
> that I'm in bear and I'm sure of it, it means that I shift 100% to
> defensive. There is no need for mix.

So the core deliverable is a **binary classifier** — `IN_BEAR` /
`NOT_IN_BEAR` — not a continuous score feeding an allocation curve. Allocation
becomes a trivial function of the classification: 100% aggressive or 100%
defensive (or cash). All the actual difficulty is in the classifier.

## 2. Success metrics (exact definitions)

Measured over a **20-year backtest (2005–2025)** against every historical
bear in that window (not a hand-picked subset).

| Metric | Target | Definition |
|---|---|---|
| **Coverage** | ≥ 85% | For each ground-truth bear period, (days classified `IN_BEAR` during that period) / (total days in that period), averaged across all bear periods (or pooled — see §3.3 for which). |
| **False-positive rate** | < 5% | (days classified `IN_BEAR` while NOT in a ground-truth bear period) / (total non-bear days in the 20yr window). |
| **Defensive-period return** | > 5% (stretch: positive at minimum) | Return of the portfolio's defensive allocation *specifically during the days it's deployed*, annualized or summed across all defensive episodes. Must not lose money; should ideally profit, since bears are net down periods and the defensive sleeve should be net short/uncorrelated. |

**Terminology note**: the user's brief calls false-positives "false negatives"
("times when it goes defensive when it's not in bear period"). By standard
classification terminology this is a **false positive** (predicted `BEAR`,
true label `NOT_BEAR`) — a true false-negative would be missing a real bear
(predicted `NOT_BEAR`, true label `BEAR`), which is the *inverse* of
coverage. I'm using standard terminology throughout the rest of this doc and
the code (`false_positive_rate`), and treating "coverage" as the complement
of the false-negative rate, to avoid ambiguity — but the target and its
intent are unchanged from the brief.

These three targets pull against each other by construction (higher coverage
usually costs false-positive rate, and both cost reaction speed vs.
precision). Hitting all three simultaneously across 20 years including both
fast shocks (2020) and slow grinds (2022) is a genuinely hard, possibly
not-fully-achievable target — treated here as the objective to iterate
toward, not a guarantee. If a true Pareto frontier is hit, the honest
trade-offs get reported, not hidden.

## 3. Ground truth: what counts as a "bear"

This is the methodological linchpin — the metrics above are meaningless
without an objective, reproducible bear-period definition. Prior work in
this repo used a hand-copied list of 6 dates from an old research note. That
list is a reasonable *sanity check* but isn't reproducible or objective
enough to certify metrics against.

### 3.1 Primary definition — the standard 20% rule

Researched: the conventional, widely-used definition (SoFi, Hartford Funds,
Invesco taxonomy, TradingView, and standard finance-industry usage) is:

> A bear market is a decline of ≥20% from the most recent closing high
> (all-time or cycle high). The bear period is backdated to that prior high.
> It ends at the first new closing high above the pre-bear peak (the trough
> is the lowest close in between).

This is mechanical and requires no judgment calls, so it's implemented as
code (`ground_truth.py`), run against daily SPY closes for 2005–2025, not
hand-maintained. Whatever episodes it finds *are* the primary ground truth.

**Which dates count as "in bear" for scoring — peak-to-trough, not peak-to-recovery.**
A full bear-market cycle by the 20% rule technically runs from the prior
peak until price recovers back *above that same peak* — but for 2008 that
recovery didn't happen until ~2012-2013, years after the March 2009 trough.
Labeling that entire multi-year grind as "should be defensive" would be
self-defeating: the classifier would be expected to stay short through a
multi-year bull recovery, which is exactly the opposite of "invest in high
betas otherwise." So the ground-truth `IN_BEAR` window used for scoring is
**peak-date → trough-date only** (the actual decline phase). Once the
trough is in, ground truth is `NOT_IN_BEAR` — the classifier should have
flipped back to aggressive by then, or shortly after. This is inherently a
retrospective label (the trough is only knowable in hindsight); the
classifier being scored never sees it in advance, same as any backtest.

### 3.2 Secondary definition — corrections (10–20%)

The strict 20% rule likely excludes 2011, 2015, and 2018 (historically
~19%, ~12%, ~20% depending on exact close-to-close measurement), which prior
analysis in this repo treated as bears worth defending against. These get
labeled separately as "corrections" (≥10%, <20%) using the same mechanical
peak/trough logic, and tracked as a **secondary, informational metric** —
not part of the primary 85%/5%/5% certification, but reported alongside it,
since a classifier that nails 20%+ bears but whipsaws through every 10-15%
correction isn't actually good in practice.

### 3.3 Coverage math — pooled, not averaged-per-episode

Coverage is computed **pooled** (total bear-days-covered / total bear-days),
not averaged per-episode. Averaging per-episode would let the classifier
nail a short 2011-style correction and weight it equally against missing
most of the 2008 crash, which is backwards — 2008 has ~150x more bear-days
than 2011 and should dominate the metric accordingly.

## 4. Research summary (this task came with explicit permission to look things up)

- **Bear market dating**: 20% peak-to-trough close-to-close, per above.
  [SoFi](https://www.sofi.com/learn/content/bear-market/),
  [Invesco taxonomy](https://www.invesco.com/content/dam/invesco/emea/en/pdf/T_con_zero-bear_necessities_apr_2020.pdf).
- **Trend-following baseline**: Mebane Faber's 2006 "A Quantitative Approach
  to Tactical Asset Allocation" — a 10-month SMA (≈200 trading days) rule:
  hold when price > SMA, move to cash/defensive when price < SMA. Simple,
  well-documented, and specifically shown to work "largely by avoiding most
  of the severe drawdown of 2008."
  [paper](https://www.trendfollowing.com/whitepaper/CMT-Simple.pdf),
  [10-years-later revisit](https://allocatortraining.com/wp-content/uploads/2023/06/A-Quantitative-Approach-to-Tactical-Asset-Allocation.pdf).
  Known weaknesses directly relevant here: whipsaws in choppy/sideways
  markets, and pure-price lag on fast shocks (2020-style crashes are mostly
  over before a 200-day SMA reacts).
- **Whipsaw reduction**: (a) volatility-adaptive bands instead of a hard
  crossing threshold — e.g. Keltner-style ATR bands around the trend line so
  price must clear a volatility-scaled buffer, not just cross a line; (b)
  multi-signal consensus — require 2-of-N independent signals to agree
  before flipping state, not a single indicator.
  [StockCharts on MA whipsaw](https://stockcharts.com/articles/arthurhill/2018/10/systemtrader---reducing-moving-average-whipsaws-with-smoothing-and-quantifying-filters-.html).
- **Regime-switching / HMM**: a legitimate, established alternative
  framing (bull/bear as hidden states inferred from observable returns/vol).
  Noted but deprioritized as the primary approach: HMMs are excellent at
  *retrospective* regime labeling but have the same real-time filtering lag
  problem as any statistical approach — the state estimate only firms up
  after enough data accumulates, which cuts against "as fast as possible."
  Worth a later experiment, not the first approach.
  [QuestDB overview](https://questdb.com/glossary/market-regime-detection-using-hidden-markov-models/),
  [arXiv: hierarchical HMM for bull/bear detection](https://arxiv.org/pdf/2007.14874).

## 5. Proposed architecture (first iteration — expect to revise from backtest results)

A layered, binary state machine, not a single indicator:

1. **Trend core** (Faber-style): price relative to a volatility-adaptive
   moving-average band, not a hard SMA crossing. This is the slow, reliable
   backbone — good coverage on grinding bears (2022), inherently laggy on
   fast ones (2020).
2. **Fast confirmation layer**: reuse the macro components already built in
   `archive/bond_rate_v2/data/bond_rate_score.py` (credit spread widening —
   the fastest-reacting signal found this session — real yield stress,
   curve shape) plus price-based fast signals (short-lookback drawdown
   velocity, VIX spikes) specifically to shave latency off the trend core
   for sudden shocks.
3. **State machine with asymmetric confirmation**: entering `IN_BEAR`
   requires stronger/multi-signal agreement (protects the false-positive
   budget); exiting back to `NOT_IN_BEAR` is comparatively easier (protects
   coverage — don't overstay in a bear that's clearly ending, since that
   both hurts the "quickly recognize when it's ending" goal and, if the
   defensive sleeve is short-biased, costs money in the recovery).
4. **Defensive sleeve**: short-biased (SH-weighted, per the confirmed
   research finding that SH is the one instrument positive across every
   historical bear type tested) when `IN_BEAR`, 100% high-beta when not.
   No blend.

## 6. Directory layout (this folder)

```
improvement/
  SPEC.md                 <- this file
  ground_truth.py         <- mechanical 20%-rule + 10%-rule bear labeler on SPY
  ground_truth_periods.py <- generated/cached output: the actual dated episodes
  signals/                <- candidate signal implementations, one file each
  classifier.py           <- the state machine combining signals -> IN_BEAR/NOT_IN_BEAR
  backtest_classifier.py  <- computes coverage / false-positive-rate / defensive-return
                              directly against ground truth (NOT a full portfolio
                              backtest at first - classify first, verify the
                              classifier is good on its own terms, THEN wire into
                              a strategy/portfolio backtest)
  results/                <- metric outputs per iteration, dated
```

Deliberately classifier-first, portfolio-backtest-second: the 3 target
metrics are about the *classifier*, not fund performance. Wiring a mediocre
classifier into the existing `Strategy`/`Backtest` machinery and reading
CAGR/Sharpe off the result would obscure exactly the thing being measured.
Get coverage/false-positive/defensive-return right against ground truth
first; only then plug the winning classifier into a real `Strategy` for an
end-to-end portfolio backtest (reusing `backtest/backtest.py`).
