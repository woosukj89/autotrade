"""
Exchange Connector Interface
============================

Abstract base class defining the interface for exchange connections.
All exchange implementations (Robinhood, Alpaca, IBKR, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Position:
    """Represents a position in a security."""
    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass
class Order:
    """Represents an order."""
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]  # For limit orders
    status: OrderStatus
    filled_quantity: float
    filled_price: Optional[float]
    created_at: datetime
    updated_at: datetime


@dataclass
class AccountInfo:
    """Account information."""
    account_id: str
    cash: float
    buying_power: float
    portfolio_value: float
    equity: float
    margin_used: float = 0.0
    day_trades_remaining: int = 3  # PDT rule


@dataclass
class Quote:
    """Price quote for a security."""
    ticker: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime


class ExchangeConnector(ABC):
    """
    Abstract base class for exchange connectors.

    All exchange implementations must inherit from this class
    and implement all abstract methods.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the exchange.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the exchange."""
        pass

    # ─── Account Methods ───────────────────────────────────────────

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get account information including cash and buying power."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        pass

    @abstractmethod
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific ticker."""
        pass

    # ─── Market Data Methods ───────────────────────────────────────

    @abstractmethod
    def get_quote(self, ticker: str) -> Optional[Quote]:
        """Get current quote for a ticker."""
        pass

    @abstractmethod
    def get_quotes(self, tickers: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple tickers."""
        pass

    @abstractmethod
    def get_historical_prices(
        self,
        ticker: str,
        days: int = 365,
        interval: str = "day"
    ) -> Optional[List[Dict]]:
        """
        Get historical price data.

        Args:
            ticker: Stock symbol
            days: Number of days of history
            interval: "day", "week", "hour", etc.

        Returns:
            List of OHLCV dictionaries or None if not available.
        """
        pass

    # ─── Order Methods ─────────────────────────────────────────────

    @abstractmethod
    def place_market_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float
    ) -> Optional[Order]:
        """
        Place a market order.

        Args:
            ticker: Stock symbol
            side: BUY or SELL
            quantity: Number of shares

        Returns:
            Order object if successful, None otherwise.
        """
        pass

    @abstractmethod
    def place_limit_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        limit_price: float
    ) -> Optional[Order]:
        """
        Place a limit order.

        Args:
            ticker: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            limit_price: Limit price

        Returns:
            Order object if successful, None otherwise.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details by ID."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Get all open/pending orders."""
        pass

    # ─── Utility Methods ───────────────────────────────────────────

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        pass

    @abstractmethod
    def get_market_hours(self) -> Dict[str, datetime]:
        """
        Get market hours for today.

        Returns:
            Dict with 'open' and 'close' datetime keys.
        """
        pass

    def validate_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None
    ) -> tuple[bool, str]:
        """
        Validate an order before submission.

        Args:
            ticker: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            price: Optional limit price

        Returns:
            (is_valid, error_message) tuple
        """
        if quantity <= 0:
            return False, "Quantity must be positive"

        if price is not None and price <= 0:
            return False, "Price must be positive"

        quote = self.get_quote(ticker)
        if quote is None:
            return False, f"Could not get quote for {ticker}"

        if side == OrderSide.BUY:
            account = self.get_account_info()
            estimated_cost = quantity * (price or quote.ask)
            if estimated_cost > account.buying_power:
                return False, f"Insufficient buying power: need ${estimated_cost:.2f}, have ${account.buying_power:.2f}"

        elif side == OrderSide.SELL:
            position = self.get_position(ticker)
            if position is None or position.shares < quantity:
                current_shares = position.shares if position else 0
                return False, f"Insufficient shares: need {quantity}, have {current_shares}"

        return True, ""
