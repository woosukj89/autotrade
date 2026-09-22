"""
Toss Securities (토스증권) Open API Connector — UNVALIDATED SKELETON
====================================================================

Implementation of ExchangeConnector for Toss Securities' Open API,
which opened in 2026 and explicitly supports order placement on both
KRX and US-listed stocks — not just domestic Korean securities. This
is what makes it a real candidate for running this project's live
automation on a Korean brokerage account, per the user's request.

STATUS: UNVALIDATED. This has been written directly from Toss's public
API documentation (developers.tossinvest.com) and an unofficial
reference client (github.com/nbsp1221/tossinvest-openapi), but has
never been run against a real account — that requires the user to:
  1. Hold a Toss Securities account.
  2. Register as an API client via the Toss Securities web platform
     (WTS) settings to obtain a client_id/client_secret.
  3. Provide those credentials so this can actually be exercised.
Endpoint paths, field names, and response shapes below are believed
correct as of when this was written but have NOT been confirmed
against live responses. Treat every method here as "shaped correctly,
contents unverified" until tested against a real account — do not
route real money through this without that verification pass first.

Authentication: OAuth 2.0 Client Credentials grant. Order/account
endpoints require the bearer token plus an account-identification
header (X-Tossinvest-Account) identifying which brokerage account to
act on, since one API client can be linked to multiple accounts.

Installation:
    pip install requests python-dotenv

Environment Variables:
    TOSS_CLIENT_ID: OAuth client ID issued when registering the API client.
    TOSS_CLIENT_SECRET: OAuth client secret.
    TOSS_ACCOUNT_NUMBER: The brokerage account to trade (X-Tossinvest-Account).
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, '.env'))
except ImportError:
    pass

import requests

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

API_BASE = "https://openapi.tossinvest.com"


class TossConnector(ExchangeConnector):
    """
    Toss Securities Open API connector.

    UNVALIDATED — see module docstring. Written to the same
    ExchangeConnector interface as RobinhoodConnector so it can be
    dropped into live_momentum_trader.py / live_regime_trader.py's
    connector slot with no driver-code changes, once real credentials
    are available and this has been exercised against a paper or small
    real position first.

    Usage (once validated):
        connector = TossConnector()
        connector.connect()
        account = connector.get_account_info()
        order = connector.place_market_order("AAPL", OrderSide.BUY, 1)
        connector.disconnect()
    """

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        account_number: str = None,
    ):
        self.client_id = client_id or os.environ.get('TOSS_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('TOSS_CLIENT_SECRET')
        self.account_number = account_number or os.environ.get('TOSS_ACCOUNT_NUMBER')

        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._connected = False

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Toss API credentials required. Set TOSS_CLIENT_ID and "
                "TOSS_CLIENT_SECRET environment variables, or register an "
                "API client via the Toss Securities web platform (WTS) "
                "settings first if you haven't."
            )

    # ─── Connection ─────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        if self.account_number:
            headers["X-Tossinvest-Account"] = self.account_number
        return headers

    def connect(self) -> bool:
        """OAuth 2.0 Client Credentials token fetch."""
        try:
            resp = requests.post(
                f"{API_BASE}/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            self._connected = True
            print("[Toss] Connected successfully")
            return True
        except Exception as e:
            print(f"[Toss] Connection error: {e}")
            return False

    def disconnect(self) -> None:
        self._access_token = None
        self._connected = False
        print("[Toss] Disconnected")

    def is_connected(self) -> bool:
        if not self._connected or not self._access_token:
            return False
        if self._token_expires_at and datetime.now() >= self._token_expires_at:
            return False
        return True

    def _ensure_connected(self):
        if not self.is_connected():
            raise RuntimeError("Not connected to Toss. Call connect() first.")

    # ─── Account Methods ────────────────────────────────────────────

    def get_account_info(self) -> AccountInfo:
        self._ensure_connected()
        resp = requests.get(f"{API_BASE}/api/v1/accounts/balance", headers=self._headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()

        return AccountInfo(
            account_id=self.account_number or data.get("accountNumber", "unknown"),
            cash=float(data.get("cashBalance", 0)),
            buying_power=float(data.get("buyingPower", data.get("cashBalance", 0))),
            portfolio_value=float(data.get("totalEvaluationAmount", 0)),
            equity=float(data.get("totalEvaluationAmount", 0)),
        )

    def get_positions(self) -> List[Position]:
        self._ensure_connected()
        resp = requests.get(f"{API_BASE}/api/v1/accounts/positions", headers=self._headers(), timeout=15)
        resp.raise_for_status()
        holdings = resp.json().get("positions", [])

        positions = []
        for h in holdings:
            shares = float(h.get("quantity", 0))
            if shares <= 0:
                continue
            avg_cost = float(h.get("averagePrice", 0))
            current_price = float(h.get("currentPrice", avg_cost))
            market_value = shares * current_price
            unrealized_pnl = market_value - (shares * avg_cost)
            unrealized_pnl_pct = (unrealized_pnl / (shares * avg_cost) * 100) if avg_cost > 0 else 0

            positions.append(Position(
                ticker=h.get("symbol", "").upper(),
                shares=shares,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
            ))
        return positions

    def get_position(self, ticker: str) -> Optional[Position]:
        for pos in self.get_positions():
            if pos.ticker.upper() == ticker.upper():
                return pos
        return None

    # ─── Market Data Methods ────────────────────────────────────────

    def get_quote(self, ticker: str) -> Optional[Quote]:
        self._ensure_connected()
        try:
            resp = requests.get(
                f"{API_BASE}/api/v1/market/quote",
                params={"symbol": ticker, "market": "US"},
                headers=self._headers(),
                timeout=15,
            )
            resp.raise_for_status()
            q = resp.json()
            return Quote(
                ticker=ticker.upper(),
                bid=float(q.get("bidPrice", 0) or 0),
                ask=float(q.get("askPrice", 0) or 0),
                last=float(q.get("lastPrice", 0) or 0),
                volume=int(float(q.get("volume", 0) or 0)),
                timestamp=datetime.now(),
            )
        except Exception as e:
            print(f"[Toss] get_quote({ticker}) error: {e}")
            return None

    def get_quotes(self, tickers: List[str]) -> Dict[str, Quote]:
        # Toss's documented quote endpoint is single-symbol; no batch
        # endpoint confirmed in the public docs as of writing. Falls back
        # to sequential calls - fine for this project's ~10-25 ticker
        # universe, but flag for revisit if the real docs turn out to
        # expose a batch endpoint once tested against a real account.
        quotes = {}
        for ticker in tickers:
            q = self.get_quote(ticker)
            if q:
                quotes[ticker] = q
        return quotes

    def get_historical_prices(
        self,
        ticker: str,
        days: int = 365,
        interval: str = "day",
    ) -> Optional[List[Dict]]:
        self._ensure_connected()
        try:
            resp = requests.get(
                f"{API_BASE}/api/v1/market/candles",
                params={"symbol": ticker, "market": "US", "period": interval, "count": days},
                headers=self._headers(),
                timeout=15,
            )
            resp.raise_for_status()
            candles = resp.json().get("candles", [])
            return [
                {
                    "date": c.get("date"),
                    "open": float(c.get("open", 0)),
                    "high": float(c.get("high", 0)),
                    "low": float(c.get("low", 0)),
                    "close": float(c.get("close", 0)),
                    "volume": int(float(c.get("volume", 0) or 0)),
                }
                for c in candles
            ]
        except Exception as e:
            print(f"[Toss] get_historical_prices({ticker}) error: {e}")
            return None

    # ─── Order Methods ──────────────────────────────────────────────

    def place_market_order(self, ticker: str, side: OrderSide, quantity: float) -> Optional[Order]:
        self._ensure_connected()
        try:
            resp = requests.post(
                f"{API_BASE}/api/v1/orders",
                headers=self._headers(),
                json={
                    "symbol": ticker,
                    "market": "US",
                    "side": "BUY" if side == OrderSide.BUY else "SELL",
                    "orderType": "MARKET",
                    "quantity": quantity,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return self._order_from_response(ticker, side, OrderType.MARKET, quantity, data)
        except Exception as e:
            print(f"[Toss] place_market_order({ticker}, {side}, {quantity}) error: {e}")
            return None

    def place_limit_order(self, ticker: str, side: OrderSide, quantity: float, limit_price: float) -> Optional[Order]:
        self._ensure_connected()
        try:
            resp = requests.post(
                f"{API_BASE}/api/v1/orders",
                headers=self._headers(),
                json={
                    "symbol": ticker,
                    "market": "US",
                    "side": "BUY" if side == OrderSide.BUY else "SELL",
                    "orderType": "LIMIT",
                    "quantity": quantity,
                    "price": limit_price,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return self._order_from_response(ticker, side, OrderType.LIMIT, quantity, data, price=limit_price)
        except Exception as e:
            print(f"[Toss] place_limit_order({ticker}, {side}, {quantity}, {limit_price}) error: {e}")
            return None

    def _order_from_response(self, ticker, side, order_type, quantity, data, price=None) -> Order:
        now = datetime.now()
        status_map = {
            "PENDING": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
        }
        return Order(
            order_id=str(data.get("orderId", "")),
            ticker=ticker.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=status_map.get(data.get("status", "PENDING"), OrderStatus.PENDING),
            filled_quantity=float(data.get("filledQuantity", 0)),
            filled_price=float(data["filledPrice"]) if data.get("filledPrice") else None,
            created_at=now,
            updated_at=now,
        )

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_connected()
        try:
            resp = requests.delete(f"{API_BASE}/api/v1/orders/{order_id}", headers=self._headers(), timeout=15)
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"[Toss] cancel_order({order_id}) error: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        self._ensure_connected()
        try:
            resp = requests.get(f"{API_BASE}/api/v1/orders/{order_id}", headers=self._headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            side = OrderSide.BUY if data.get("side") == "BUY" else OrderSide.SELL
            order_type = OrderType.MARKET if data.get("orderType") == "MARKET" else OrderType.LIMIT
            return self._order_from_response(
                data.get("symbol", ""), side, order_type, float(data.get("quantity", 0)), data,
                price=float(data["price"]) if data.get("price") else None,
            )
        except Exception as e:
            print(f"[Toss] get_order({order_id}) error: {e}")
            return None

    def get_open_orders(self) -> List[Order]:
        self._ensure_connected()
        try:
            resp = requests.get(f"{API_BASE}/api/v1/orders", params={"status": "OPEN"}, headers=self._headers(), timeout=15)
            resp.raise_for_status()
            orders = []
            for data in resp.json().get("orders", []):
                side = OrderSide.BUY if data.get("side") == "BUY" else OrderSide.SELL
                order_type = OrderType.MARKET if data.get("orderType") == "MARKET" else OrderType.LIMIT
                orders.append(self._order_from_response(
                    data.get("symbol", ""), side, order_type, float(data.get("quantity", 0)), data,
                    price=float(data["price"]) if data.get("price") else None,
                ))
            return orders
        except Exception as e:
            print(f"[Toss] get_open_orders() error: {e}")
            return []

    # ─── Utility Methods ────────────────────────────────────────────

    def is_market_open(self) -> bool:
        # US market hours check via US Eastern time - matches the same
        # logic Robinhood-side driver code uses elsewhere in this
        # project, since orders here still target the US market.
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et <= market_close

    def get_market_hours(self) -> Dict[str, datetime]:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        return {
            "open": now_et.replace(hour=9, minute=30, second=0, microsecond=0),
            "close": now_et.replace(hour=16, minute=0, second=0, microsecond=0),
        }


def create_toss_connector() -> TossConnector:
    """Factory matching connectors.create_robinhood_connector()'s pattern."""
    return TossConnector()
