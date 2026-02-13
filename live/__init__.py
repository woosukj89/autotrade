"""
Live Trading Package
====================

Provides live trading runners for executing strategies with real brokers.

Available Traders:
    - LiveRegimeTrader: Live implementation of the Regime-Adaptive Strategy

The live trader uses the actual RegimeAdaptiveStrategy from strategies/regime_adaptive_strategy.py
to ensure consistency between backtesting and live trading.

Usage:
    from live import LiveRegimeTrader
    from connectors import create_robinhood_connector
    from notifications import create_email_notifier

    connector = create_robinhood_connector()
    notifier = create_email_notifier("your@email.com")

    trader = LiveRegimeTrader(
        connector=connector,
        email_notifier=notifier,
        dry_run=True,  # Set to False for real trades
    )

    trader.run()
"""

from .live_regime_trader import (
    LiveRegimeTrader,
    LiveExecutionContext,
    DryRunExecutionContext,
    MockConnector,
    TradeOrder,
)

__all__ = [
    'LiveRegimeTrader',
    'LiveExecutionContext',
    'DryRunExecutionContext',
    'MockConnector',
    'TradeOrder',
]
