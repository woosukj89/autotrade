"""
Exchange Connectors Package
===========================

Provides abstract interface and implementations for connecting to various
trading platforms/brokers.

Available Connectors:
    - RobinhoodConnector: Robinhood trading platform

Usage:
    from connectors import RobinhoodConnector, create_robinhood_connector

    # Create and connect
    connector = create_robinhood_connector()
    connector.connect()

    # Use the connector
    account = connector.get_account_info()
    positions = connector.get_positions()

    # Place orders
    order = connector.place_market_order("AAPL", OrderSide.BUY, 1)

    # Disconnect when done
    connector.disconnect()
"""

from .base import (
    ExchangeConnector,
    Position,
    Order,
    AccountInfo,
    Quote,
    OrderSide,
    OrderType,
    OrderStatus,
)

from .robinhood import (
    RobinhoodConnector,
    create_robinhood_connector,
)

__all__ = [
    # Base classes
    'ExchangeConnector',
    'Position',
    'Order',
    'AccountInfo',
    'Quote',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    # Robinhood
    'RobinhoodConnector',
    'create_robinhood_connector',
]
