"""
Simulates the live bot being unable to execute trades for stretches of
time - the concrete scenario described: Robinhood's session token expires
every few days, and a refresh sometimes silently fails, leaving the bot
offline until someone notices. During an outage the account doesn't
vanish - existing positions are still held and still gain/lose value
day to day - the bot just can't place new trades or react to a regime
change until it's back.

OutageSimulatingStrategy wraps an inner strategy: on an "available" day
it delegates to inner.execute() as normal; on an "outage" day it returns
context.portfolio completely unchanged (a no-op - the backtest engine
still marks existing positions to that day's market price when it builds
the next snapshot, so P&L still moves, just no rebalancing/regime
reaction happens).

Outage schedule is generated once per instance from a fixed seed, so a
given (mean_gap_days, mean_outage_days, seed) triple is fully
reproducible - not re-randomized on every execute() call.
"""
import random
from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategies'))
from strategy import Strategy, Portfolio, ExecutionContext


def generate_outage_windows(start_date, end_date, mean_gap_days: float, mean_outage_days: float,
                             seed: int = 0) -> List[Tuple]:
    """Poisson-ish process: outages start roughly every `mean_gap_days`
    on average (exponential inter-arrival), each lasting a random 1..2x
    `mean_outage_days` (uniform). Returns [(outage_start, outage_end), ...].
    """
    rng = random.Random(seed)
    windows = []
    t = start_date
    while t < end_date:
        gap = rng.expovariate(1.0 / mean_gap_days) if mean_gap_days > 0 else float('inf')
        t = t + __import__('datetime').timedelta(days=gap)
        if t >= end_date:
            break
        dur = rng.uniform(0.5, 1.5) * mean_outage_days
        outage_end = t + __import__('datetime').timedelta(days=dur)
        windows.append((t, outage_end))
        t = outage_end
    return windows


class OutageSimulatingStrategy(Strategy):
    def __init__(self, inner: Strategy, outage_windows: List[Tuple]):
        self.inner = inner
        self.outage_windows = outage_windows
        self._total_days = 0
        self._outage_days = 0

    def _in_outage(self, date) -> bool:
        for start, end in self.outage_windows:
            if start <= date <= end:
                return True
        return False

    def execute(self, context: ExecutionContext) -> Portfolio:
        self._total_days += 1
        if self._in_outage(context.date):
            self._outage_days += 1
            return context.portfolio  # no-op: hold whatever we're already holding
        return self.inner.execute(context)

    @property
    def outage_fraction(self) -> float:
        return self._outage_days / self._total_days if self._total_days else 0.0
