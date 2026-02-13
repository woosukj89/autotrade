from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class Position:
    """A holding in a single security."""
    ticker: str
    shares: float
    avg_cost: float

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost


@dataclass
class Portfolio:
    """Full state of a portfolio: cash plus positions."""
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def get_shares(self, ticker: str) -> float:
        pos = self.positions.get(ticker)
        return pos.shares if pos else 0.0

    def total_value(self, price_fn) -> float:
        positions_value = sum(
            pos.shares * (price_fn(ticker) or pos.avg_cost)
            for ticker, pos in self.positions.items()
        )
        return self.cash + positions_value


class ExecutionContext:
    """Data context provided to the strategy at each time step.

    Attributes:
        date: Current simulation date.
        portfolio: Deep copy of the current portfolio (read-only snapshot).
    """

    def __init__(self, date: datetime, portfolio: Portfolio,
                 get_price_fn, get_historical_fn, get_fundamentals_fn):
        self.date = date
        self.portfolio = portfolio
        self._get_price = get_price_fn
        self._get_historical = get_historical_fn
        self._get_fundamentals = get_fundamentals_fn

    def get_price(self, ticker: str) -> Optional[float]:
        """Get the latest available closing price for *ticker* as of the current date."""
        return self._get_price(ticker, self.date)

    def get_historical_prices(self, ticker: str, periods: int = 30) -> Optional[pd.DataFrame]:
        """Get up to *periods* rows of historical OHLCV data ending on the current date."""
        return self._get_historical(ticker, self.date, periods)

    def get_fundamentals(self, ticker: str, field_name: str = None):
        """Query the fundamentals SQLite table for *ticker* up to the current date.

        If *field_name* is given, only rows matching that field are returned.
        Returns a ``pandas.DataFrame``.
        """
        return self._get_fundamentals(ticker, self.date, field_name)


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclass this and implement :meth:`execute` to define a trading strategy
    that can be run by the :class:`Backtest` engine.
    """

    @abstractmethod
    def execute(self, context: ExecutionContext) -> Portfolio:
        """Run one step of the strategy.

        Args:
            context: An :class:`ExecutionContext` carrying the current date,
                     a read-only copy of the portfolio, and helper methods
                     for fetching price / fundamental data.

        Returns:
            A :class:`Portfolio` representing the *desired* state after this
            step.  The backtest engine will diff this against the current
            portfolio, execute the implied trades (applying buffer pricing
            if configured), and update the actual portfolio accordingly.
        """
        ...
