# exchange/kalshi.py
"""Kalshi prediction market client for BTC contracts."""
import base64
import json
import time
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID

BASE_URL = "https://api.elections.kalshi.com"
DEMO_URL = "https://demo-api.kalshi.co"


class KalshiClient:
    def __init__(self, api_key_id: str, private_key_path: str, demo: bool = False):
        self.api_key_id = api_key_id
        self.base_url = DEMO_URL if demo else BASE_URL
        self._private_key = self._load_key(private_key_path)

    def _load_key(self, path: str):
        with open(path, "rb") as f:
            raw = f.read()
        # The key file may contain an "API Key = ..." header line before the
        # PEM block.  Extract only the RSA private key portion.
        pem_start = raw.find(b"-----BEGIN")
        if pem_start > 0:
            raw = raw[pem_start:]
        return serialization.load_pem_private_key(raw, password=None)

    def _sign(self, method: str, path: str) -> tuple[str, str]:
        """Sign a request. Returns (timestamp_ms, signature_b64).

        Per official Kalshi spec (kalshi-starter-code-python):
        - Message = timestamp_ms + HTTP_METHOD + path (no delimiters, no body)
        - Algorithm = RSA-PSS with SHA-256
        - Salt length = PSS.DIGEST_LENGTH (not MAX_LENGTH)
        - Sign raw message bytes (not pre-hashed)
        - Query params stripped from path before signing
        """
        ts = str(int(time.time() * 1000))
        # Strip query parameters from path before signing
        path_clean = path.split("?")[0]
        message = f"{ts}{method}{path_clean}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return ts, base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str) -> dict:
        ts, sig = self._sign(method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        for attempt in range(3):
            headers = self._headers("GET", path)
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 500 and attempt < 2:
                time.sleep(0.5)
                continue
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, data: dict) -> dict:
        url = f"{self.base_url}{path}"
        for attempt in range(3):
            headers = self._headers("POST", path)
            resp = requests.post(url, headers=headers, json=data, timeout=10)
            if resp.status_code == 500 and attempt < 2:
                time.sleep(0.5)
                continue
            if resp.status_code in (400, 409):
                # Include response body in error for debugging
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                raise requests.HTTPError(
                    f"{resp.status_code} {resp.reason}: {body} (data={data})",
                    response=resp,
                )
            resp.raise_for_status()
            return resp.json()

    def _delete(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        headers = self._headers("DELETE", path)
        resp = requests.delete(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # --- Market Discovery ---

    def get_btc_events(self, status: str = "open") -> list[dict]:
        """Find open BTC prediction events."""
        # Try multiple series tickers that Kalshi might use
        for series in ["KXBTC", "KXBTCD", "KXBTCUSD", "BTC"]:
            try:
                resp = self._get("/trade-api/v2/events", {
                    "series_ticker": series,
                    "status": status,
                    "with_nested_markets": "true",
                    "limit": 20,
                })
                events = resp.get("events", [])
                if events:
                    return events
            except Exception:
                continue
        return []

    def get_markets(self, event_ticker: str = None, series_ticker: str = None,
                    status: str = "open") -> list[dict]:
        """List markets, optionally filtered."""
        params = {"status": status, "limit": 200}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        resp = self._get("/trade-api/v2/markets", params)
        return resp.get("markets", [])

    def get_market(self, ticker: str) -> dict:
        """Get a single market's details."""
        return self._get(f"/trade-api/v2/markets/{ticker}")

    def get_orderbook(self, ticker: str) -> dict:
        """Get the orderbook for a market."""
        return self._get(f"/trade-api/v2/markets/{ticker}/orderbook")

    def find_btc_15m_markets(self) -> list[dict]:
        """Find BTC markets expiring within the next ~30 minutes.
        These are the short-term contracts we want to trade."""
        events = self.get_btc_events()
        now = time.time()
        short_term = []
        for event in events:
            for market in event.get("markets", []):
                # Check if market expires within 30 minutes
                exp = market.get("expiration_time") or market.get("close_time", "")
                if exp:
                    try:
                        from datetime import datetime
                        if "T" in exp:
                            exp_ts = datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp()
                            mins_to_exp = (exp_ts - now) / 60
                            if 5 < mins_to_exp < 30:
                                market["_mins_to_expiry"] = round(mins_to_exp, 1)
                                short_term.append(market)
                    except Exception:
                        pass
        return sorted(short_term, key=lambda m: m.get("_mins_to_expiry", 999))

    # --- Trading ---

    def place_order(self, ticker: str, side: str, count: int,
                    price_cents: int | None = None, order_type: str = "market",
                    action: str = "buy") -> dict:
        """Place an order.
        side: 'yes' or 'no'
        count: number of contracts
        price_cents: 1-99 for limit orders
        order_type: 'market' or 'limit'
        action: 'buy' or 'sell'
        """
        data = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "type": order_type,
            "count": count,
        }
        if order_type == "limit" and price_cents is not None:
            if side == "yes":
                data["yes_price"] = price_cents
            else:
                data["no_price"] = price_cents
        return self._post("/trade-api/v2/portfolio/orders", data)

    def bet_btc_up(self, ticker: str, count: int, price_cents: int | None = None) -> dict:
        """Bet that BTC will be ABOVE the strike at expiry."""
        otype = "limit" if price_cents else "market"
        return self.place_order(ticker, "yes", count, price_cents, otype)

    def bet_btc_down(self, ticker: str, count: int, price_cents: int | None = None) -> dict:
        """Bet that BTC will be BELOW the strike at expiry."""
        otype = "limit" if price_cents else "market"
        return self.place_order(ticker, "no", count, price_cents, otype)

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f"/trade-api/v2/portfolio/orders/{order_id}")

    def get_order_status(self, order_id: str) -> dict:
        """Get status of a specific order (unwraps 'order' envelope)."""
        resp = self._get(f"/trade-api/v2/portfolio/orders/{order_id}")
        return resp.get("order", resp)

    def cancel_order_safe(self, order_id: str) -> dict:
        """Cancel an order. Returns filled status on 404 (order already filled)."""
        try:
            return self._delete(f"/trade-api/v2/portfolio/orders/{order_id}")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return {"status": "filled"}
            raise

    # --- Portfolio ---

    def get_balance(self) -> dict:
        """Get account balance in cents."""
        return self._get("/trade-api/v2/portfolio/balance")

    def get_positions(self, event_ticker: str = None) -> list[dict]:
        """Get open positions."""
        params = {"settlement_status": "unsettled", "limit": 100}
        if event_ticker:
            params["event_ticker"] = event_ticker
        resp = self._get("/trade-api/v2/portfolio/positions", params)
        return resp.get("market_positions", [])

    def get_orders(self, status: str = "resting") -> list[dict]:
        """Get open orders."""
        resp = self._get("/trade-api/v2/portfolio/orders", {"status": status, "limit": 100})
        return resp.get("orders", [])

    def get_fills(self, ticker: str = None, limit: int = 20) -> list[dict]:
        """Get recent trade fills with actual prices."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        resp = self._get("/trade-api/v2/portfolio/fills", params)
        return resp.get("fills", [])

    def get_settlements(self, limit: int = 20) -> list[dict]:
        """Get recent settlements with actual revenue and costs."""
        resp = self._get("/trade-api/v2/portfolio/settlements", {"limit": limit})
        return resp.get("settlements", [])
