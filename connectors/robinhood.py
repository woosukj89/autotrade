"""
Robinhood Exchange Connector
============================

Implementation of ExchangeConnector for Robinhood using robin_stocks library.

Installation:
    pip install robin_stocks pyotp python-dotenv

Authentication:
    Requires Robinhood username/password and optionally 2FA TOTP secret.
    Store credentials in .env file or as environment variables.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import warnings

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env in current dir and parent dirs
    load_dotenv()  # Load from current directory
    # Also try project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, '.env'))
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

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

# Suppress robin_stocks warnings
warnings.filterwarnings('ignore')
logging.getLogger('robin_stocks').setLevel(logging.WARNING)


class RobinhoodConnector(ExchangeConnector):
    """
    Robinhood exchange connector using robin_stocks library.

    Usage:
        connector = RobinhoodConnector(
            username="your_email",
            password="your_password",
            totp_secret="your_2fa_secret"  # Optional, for 2FA
        )
        connector.connect()

        # Get account info
        account = connector.get_account_info()
        print(f"Portfolio value: ${account.portfolio_value:,.2f}")

        # Get positions
        positions = connector.get_positions()

        # Place order
        order = connector.place_market_order("AAPL", OrderSide.BUY, 1)

        connector.disconnect()
    """

    def __init__(
        self,
        username: str = None,
        password: str = None,
        totp_secret: str = None,
        mfa_code: str = None,
        pickle_path: str = None,
    ):
        """
        Initialize Robinhood connector.

        Args:
            username: Robinhood email/username. If None, uses ROBINHOOD_USERNAME env var.
            password: Robinhood password. If None, uses ROBINHOOD_PASSWORD env var.
            totp_secret: 2FA TOTP secret for auto-generating MFA codes.
                         If None, uses ROBINHOOD_TOTP_SECRET env var.
            mfa_code: Manual MFA code (if not using TOTP secret).
            pickle_path: Directory to store/load session pickle file.
                         If None, uses ROBINHOOD_PICKLE_PATH env var or ~/.tokens/.
        """
        self.username = username or os.environ.get('ROBINHOOD_USERNAME')
        self.password = password or os.environ.get('ROBINHOOD_PASSWORD')
        self.totp_secret = totp_secret or os.environ.get('ROBINHOOD_TOTP_SECRET')
        self.mfa_code = mfa_code
        self.pickle_path = pickle_path or os.environ.get('ROBINHOOD_PICKLE_PATH', '')

        self._connected = False
        self._rs = None  # robin_stocks module

        if not self.username or not self.password:
            raise ValueError(
                "Robinhood credentials required. Set ROBINHOOD_USERNAME and "
                "ROBINHOOD_PASSWORD environment variables or pass them directly."
            )

    def connect(self) -> bool:
        """Connect to Robinhood."""
        try:
            import robin_stocks.robinhood as rs
            self._rs = rs

            # Generate TOTP code if secret provided
            mfa_code = self.mfa_code
            if self.totp_secret and not mfa_code:
                try:
                    import pyotp
                    totp = pyotp.TOTP(self.totp_secret)
                    mfa_code = totp.now()
                except ImportError:
                    print("Warning: pyotp not installed. Cannot generate 2FA code.")
                    print("Install with: pip install pyotp")

            # Login with session persistence
            login_kwargs = {
                'username': self.username,
                'password': self.password,
                'mfa_code': mfa_code,
                'store_session': True,
            }
            if self.pickle_path:
                login_kwargs['pickle_path'] = self.pickle_path

            login_result = rs.login(**login_kwargs)

            if login_result:
                self._connected = True
                print("[Robinhood] Connected successfully")
                return True
            else:
                print("[Robinhood] Connection failed")
                return False

        except Exception as e:
            print(f"[Robinhood] Connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Logout from Robinhood."""
        if self._rs and self._connected:
            try:
                self._rs.logout()
                print("[Robinhood] Disconnected")
            except Exception:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def _ensure_connected(self):
        """Raise error if not connected."""
        if not self._connected:
            raise RuntimeError("Not connected to Robinhood. Call connect() first.")

    # ─── Account Methods ───────────────────────────────────────────

    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        self._ensure_connected()

        profile = self._rs.profiles.load_account_profile()
        portfolio = self._rs.profiles.load_portfolio_profile()

        # Get cash and buying power
        cash = float(portfolio.get('withdrawable_amount', 0))
        buying_power = float(profile.get('buying_power', 0))
        portfolio_value = float(portfolio.get('equity', 0))

        return AccountInfo(
            account_id=profile.get('account_number', 'unknown'),
            cash=cash,
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            equity=portfolio_value,
            margin_used=float(profile.get('margin_balances', {}).get('margin_limit', 0)) if profile.get('margin_balances') else 0,
        )

    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        self._ensure_connected()

        positions = []
        holdings = self._rs.account.get_open_stock_positions()

        for holding in holdings:
            if float(holding.get('quantity', 0)) > 0:
                ticker = self._get_ticker_from_instrument(holding.get('instrument'))
                if ticker:
                    shares = float(holding.get('quantity', 0))
                    avg_cost = float(holding.get('average_buy_price', 0))

                    # Get current price
                    quote = self.get_quote(ticker)
                    current_price = quote.last if quote else avg_cost

                    market_value = shares * current_price
                    unrealized_pnl = market_value - (shares * avg_cost)
                    unrealized_pnl_pct = (unrealized_pnl / (shares * avg_cost) * 100) if avg_cost > 0 else 0

                    positions.append(Position(
                        ticker=ticker,
                        shares=shares,
                        avg_cost=avg_cost,
                        current_price=current_price,
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=unrealized_pnl_pct,
                    ))

        return positions

    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a specific ticker."""
        positions = self.get_positions()
        for pos in positions:
            if pos.ticker.upper() == ticker.upper():
                return pos
        return None

    def _get_ticker_from_instrument(self, instrument_url: str) -> Optional[str]:
        """Get ticker symbol from instrument URL."""
        if not instrument_url:
            return None
        try:
            instrument = self._rs.stocks.get_instrument_by_url(instrument_url)
            return instrument.get('symbol')
        except Exception:
            return None

    # ─── Market Data Methods ───────────────────────────────────────

    def get_quote(self, ticker: str) -> Optional[Quote]:
        """Get current quote for a ticker."""
        self._ensure_connected()

        try:
            quote_data = self._rs.stocks.get_quotes(ticker)
            if quote_data and len(quote_data) > 0:
                q = quote_data[0]
                if q is None:
                    # Ticker not available on Robinhood
                    return None
                return Quote(
                    ticker=ticker.upper(),
                    bid=float(q.get('bid_price', 0) or 0),
                    ask=float(q.get('ask_price', 0) or 0),
                    last=float(q.get('last_trade_price', 0) or 0),
                    volume=int(float(q.get('volume', 0) or 0)),
                    timestamp=datetime.now(),
                )
        except Exception as e:
            # Silently skip tickers not available on Robinhood
            pass

        return None

    def get_quotes(self, tickers: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple tickers."""
        self._ensure_connected()

        quotes = {}
        try:
            quote_data = self._rs.stocks.get_quotes(tickers)
            if quote_data:
                for q in quote_data:
                    if q and isinstance(q, dict):
                        ticker = q.get('symbol', '').upper()
                        if ticker:
                            quotes[ticker] = Quote(
                                ticker=ticker,
                                bid=float(q.get('bid_price', 0) or 0),
                                ask=float(q.get('ask_price', 0) or 0),
                                last=float(q.get('last_trade_price', 0) or 0),
                                volume=int(float(q.get('volume', 0) or 0)),
                                timestamp=datetime.now(),
                            )
        except Exception:
            # Silently handle errors - some tickers may not be on Robinhood
            pass

        return quotes

    def get_historical_prices(
        self,
        ticker: str,
        days: int = 365,
        interval: str = "day"
    ) -> Optional[List[Dict]]:
        """Get historical price data."""
        self._ensure_connected()

        try:
            # Map interval to Robinhood span/interval
            if days <= 7:
                span = "week"
                rh_interval = "hour" if interval == "hour" else "day"
            elif days <= 30:
                span = "month"
                rh_interval = "day"
            elif days <= 90:
                span = "3month"
                rh_interval = "day"
            elif days <= 365:
                span = "year"
                rh_interval = "day"
            else:
                span = "5year"
                rh_interval = "week"

            historicals = self._rs.stocks.get_stock_historicals(
                ticker,
                interval=rh_interval,
                span=span,
            )

            if historicals:
                return [
                    {
                        'date': h.get('begins_at'),
                        'open': float(h.get('open_price', 0)),
                        'high': float(h.get('high_price', 0)),
                        'low': float(h.get('low_price', 0)),
                        'close': float(h.get('close_price', 0)),
                        'volume': int(float(h.get('volume', 0))),
                    }
                    for h in historicals
                ]
        except Exception as e:
            print(f"[Robinhood] Error getting historical data for {ticker}: {e}")

        return None

    # ─── Order Methods ─────────────────────────────────────────────

    def place_market_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float
    ) -> Optional[Order]:
        """Place a market order."""
        self._ensure_connected()

        # Validate order
        is_valid, error = self.validate_order(ticker, side, quantity)
        if not is_valid:
            print(f"[Robinhood] Order validation failed: {error}")
            return None

        try:
            if side == OrderSide.BUY:
                result = self._rs.orders.order_buy_market(
                    ticker,
                    quantity,
                    timeInForce='gfd',  # Good for day
                )
            else:
                result = self._rs.orders.order_sell_market(
                    ticker,
                    quantity,
                    timeInForce='gfd',
                )

            if result:
                return self._parse_order(result)

        except Exception as e:
            print(f"[Robinhood] Error placing market order: {e}")

        return None

    def place_limit_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        limit_price: float
    ) -> Optional[Order]:
        """Place a limit order."""
        self._ensure_connected()

        # Validate order
        is_valid, error = self.validate_order(ticker, side, quantity, limit_price)
        if not is_valid:
            print(f"[Robinhood] Order validation failed: {error}")
            return None

        try:
            if side == OrderSide.BUY:
                result = self._rs.orders.order_buy_limit(
                    ticker,
                    quantity,
                    limit_price,
                    timeInForce='gfd',
                )
            else:
                result = self._rs.orders.order_sell_limit(
                    ticker,
                    quantity,
                    limit_price,
                    timeInForce='gfd',
                )

            if result:
                return self._parse_order(result)

        except Exception as e:
            print(f"[Robinhood] Error placing limit order: {e}")

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        self._ensure_connected()

        try:
            result = self._rs.orders.cancel_stock_order(order_id)
            return result is not None
        except Exception as e:
            print(f"[Robinhood] Error cancelling order: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details by ID."""
        self._ensure_connected()

        try:
            result = self._rs.orders.get_stock_order_info(order_id)
            if result:
                return self._parse_order(result)
        except Exception as e:
            print(f"[Robinhood] Error getting order: {e}")

        return None

    def get_open_orders(self) -> List[Order]:
        """Get all open/pending orders."""
        self._ensure_connected()

        orders = []
        try:
            open_orders = self._rs.orders.get_all_open_stock_orders()
            for o in open_orders:
                order = self._parse_order(o)
                if order:
                    orders.append(order)
        except Exception as e:
            print(f"[Robinhood] Error getting open orders: {e}")

        return orders

    def _parse_order(self, order_data: dict) -> Optional[Order]:
        """Parse Robinhood order data into Order object."""
        if not order_data:
            return None

        try:
            # Get ticker from instrument URL
            ticker = self._get_ticker_from_instrument(order_data.get('instrument'))

            # Parse status
            state = order_data.get('state', 'unknown').lower()
            status_map = {
                'queued': OrderStatus.PENDING,
                'confirmed': OrderStatus.PENDING,
                'pending': OrderStatus.PENDING,
                'filled': OrderStatus.FILLED,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'cancelled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED,
                'failed': OrderStatus.REJECTED,
            }
            status = status_map.get(state, OrderStatus.PENDING)

            # Parse side
            side = OrderSide.BUY if order_data.get('side') == 'buy' else OrderSide.SELL

            # Parse order type
            order_type_str = order_data.get('type', 'market').lower()
            type_map = {
                'market': OrderType.MARKET,
                'limit': OrderType.LIMIT,
                'stop': OrderType.STOP,
                'stop_limit': OrderType.STOP_LIMIT,
            }
            order_type = type_map.get(order_type_str, OrderType.MARKET)

            # Parse timestamps
            created_at = datetime.fromisoformat(
                order_data.get('created_at', '').replace('Z', '+00:00')
            ) if order_data.get('created_at') else datetime.now()

            updated_at = datetime.fromisoformat(
                order_data.get('updated_at', '').replace('Z', '+00:00')
            ) if order_data.get('updated_at') else created_at

            return Order(
                order_id=order_data.get('id', 'unknown'),
                ticker=ticker or 'UNKNOWN',
                side=side,
                order_type=order_type,
                quantity=float(order_data.get('quantity', 0)),
                price=float(order_data.get('price', 0)) if order_data.get('price') else None,
                status=status,
                filled_quantity=float(order_data.get('cumulative_quantity', 0)),
                filled_price=float(order_data.get('average_price', 0)) if order_data.get('average_price') else None,
                created_at=created_at,
                updated_at=updated_at,
            )
        except Exception as e:
            print(f"[Robinhood] Error parsing order: {e}")
            return None

    # ─── Utility Methods ───────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        self._ensure_connected()

        try:
            markets = self._rs.markets.get_market_hours('XNYS')  # NYSE
            if markets:
                is_open = markets.get('is_open', False)
                return is_open
        except Exception:
            pass

        # Fallback: check time (9:30 AM - 4:00 PM ET, Mon-Fri)
        from datetime import timezone
        now = datetime.now(timezone.utc)
        # Simple check (doesn't account for holidays)
        if now.weekday() >= 5:  # Weekend
            return False
        # This is a simplified check
        return True

    def get_market_hours(self) -> Dict[str, datetime]:
        """Get market hours for today."""
        self._ensure_connected()

        try:
            today = datetime.now().strftime('%Y-%m-%d')
            markets = self._rs.markets.get_market_hours('XNYS', today)
            if markets:
                opens_at = markets.get('opens_at')
                closes_at = markets.get('closes_at')

                return {
                    'open': datetime.fromisoformat(opens_at.replace('Z', '+00:00')) if opens_at else None,
                    'close': datetime.fromisoformat(closes_at.replace('Z', '+00:00')) if closes_at else None,
                }
        except Exception:
            pass

        # Fallback
        today = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        return {
            'open': today,
            'close': today.replace(hour=16, minute=0),
        }


# ─── Factory Function ──────────────────────────────────────────────

def create_robinhood_connector(
    username: str = None,
    password: str = None,
    totp_secret: str = None,
    pickle_path: str = None,
) -> RobinhoodConnector:
    """
    Factory function to create a Robinhood connector.

    Credentials can be passed directly or via environment variables:
        - ROBINHOOD_USERNAME
        - ROBINHOOD_PASSWORD
        - ROBINHOOD_TOTP_SECRET (optional, for 2FA)
        - ROBINHOOD_PICKLE_PATH (optional, directory for session pickle)

    Example:
        # Using environment variables
        connector = create_robinhood_connector()
        connector.connect()

        # Or with explicit credentials
        connector = create_robinhood_connector(
            username="email@example.com",
            password="password",
            totp_secret="your_2fa_secret"
        )
        connector.connect()
    """
    return RobinhoodConnector(
        username=username,
        password=password,
        totp_secret=totp_secret,
        pickle_path=pickle_path,
    )
