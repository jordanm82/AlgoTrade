# exchange/kalshi_ws.py
"""Kalshi WebSocket client for real-time contract pricing.

Streams ticker updates (YES/NO bid/ask) for K15 UpDown markets.
Runs in a background thread so the daemon can read latest prices
without blocking API calls.

Usage:
    ws = KalshiWebSocket(api_key_id, private_key_path)
    ws.start()  # connects in background thread
    ws.subscribe_ticker("KXBTC15M-26MAR311200-00")
    price = ws.get_ticker("KXBTC15M-26MAR311200-00")
    # → {"yes_bid": 45, "yes_ask": 48, "no_bid": 52, "no_ask": 55, "ts": 1711900000}
    ws.stop()
"""
import base64
import json
import ssl
import threading
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

WS_URL_PROD = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_URL_DEMO = "wss://demo-api.kalshi.co/trade-api/ws/v2"


class KalshiWebSocket:
    def __init__(self, api_key_id: str, private_key_path: str, demo: bool = False):
        self.api_key_id = api_key_id
        self.ws_url = WS_URL_DEMO if demo else WS_URL_PROD
        self._private_key = self._load_key(private_key_path)
        self._ws = None
        self._thread = None
        self._running = False
        self._cmd_id = 0
        self._lock = threading.Lock()

        # Latest ticker data: {market_ticker: {yes_bid, yes_ask, no_bid, no_ask, ts}}
        self._tickers = {}
        # Subscribed market tickers
        self._subscribed = set()

    def _load_key(self, path: str):
        with open(path, "rb") as f:
            raw = f.read()
        pem_start = raw.find(b"-----BEGIN")
        if pem_start > 0:
            raw = raw[pem_start:]
        return serialization.load_pem_private_key(raw, password=None)

    def _sign(self) -> tuple[str, str]:
        """Sign the WebSocket handshake. Same as REST but path is /trade-api/ws/v2."""
        ts = str(int(time.time() * 1000))
        message = f"{ts}GET/trade-api/ws/v2".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return ts, base64.b64encode(signature).decode()

    def _next_id(self) -> int:
        self._cmd_id += 1
        return self._cmd_id

    def start(self):
        """Connect and start receiving in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Disconnect."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def subscribe_ticker(self, market_ticker: str):
        """Subscribe to ticker updates for a market.
        Safe to call before connection — queued and sent on connect."""
        self._subscribed.add(market_ticker)
        # Send immediately if already connected, otherwise on_open handles it
        if self._ws:
            self._send({
                "id": self._next_id(),
                "cmd": "subscribe",
                "params": {
                    "channels": ["ticker"],
                    "market_ticker": market_ticker,
                },
            })

    def subscribe_fills(self):
        """Subscribe to fill notifications for our orders."""
        self._send({
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["fill"],
            },
        })

    def unsubscribe_ticker(self, market_ticker: str):
        """Unsubscribe from ticker updates."""
        self._subscribed.discard(market_ticker)
        self._send({
            "id": self._next_id(),
            "cmd": "unsubscribe",
            "params": {
                "channels": ["ticker"],
                "market_ticker": market_ticker,
            },
        })

    def get_ticker(self, market_ticker: str) -> dict | None:
        """Get latest ticker data for a market. Thread-safe."""
        with self._lock:
            return self._tickers.get(market_ticker)

    def get_all_tickers(self) -> dict:
        """Get all latest ticker data. Thread-safe."""
        with self._lock:
            return dict(self._tickers)

    def is_connected(self) -> bool:
        return self._running and self._ws is not None

    def _send(self, msg: dict):
        """Send a JSON message. Thread-safe."""
        if self._ws:
            try:
                self._ws.send(json.dumps(msg))
            except Exception:
                pass

    def _run_loop(self):
        """Main WebSocket loop — connects, receives, reconnects on failure."""
        import websocket

        while self._running:
            try:
                ts, sig = self._sign()
                headers = {
                    "KALSHI-ACCESS-KEY": self.api_key_id,
                    "KALSHI-ACCESS-SIGNATURE": sig,
                    "KALSHI-ACCESS-TIMESTAMP": ts,
                }

                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    header=headers,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )

                # Run with SSL
                self._ws.run_forever(
                    sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                    ping_interval=30,
                    ping_timeout=10,
                )

            except Exception as e:
                print(f"  [WS] Connection error: {e}")

            # Reconnect after 2 seconds if still running
            if self._running:
                time.sleep(2)

    def _on_open(self, ws):
        """Re-subscribe to all tickers on (re)connect."""
        # Re-subscribe to previously subscribed tickers
        for ticker in list(self._subscribed):
            self._send({
                "id": self._next_id(),
                "cmd": "subscribe",
                "params": {
                    "channels": ["ticker"],
                    "market_ticker": ticker,
                },
            })
        # Subscribe to fills
        self._send({
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["fill"],
            },
        })

    def _on_message(self, ws, message):
        """Handle incoming messages."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "ticker":
                msg = data.get("msg", {})
                ticker = msg.get("market_ticker", "")
                if ticker:
                    yes_bid = self._parse_price(msg.get("yes_bid_dollars") or msg.get("yes_bid"))
                    yes_ask = self._parse_price(msg.get("yes_ask_dollars") or msg.get("yes_ask"))
                    # NO prices: derive from YES if not provided (NO = 100 - YES)
                    no_bid_raw = msg.get("no_bid_dollars") or msg.get("no_bid")
                    no_ask_raw = msg.get("no_ask_dollars") or msg.get("no_ask")
                    no_bid = self._parse_price(no_bid_raw) if no_bid_raw else (100 - yes_ask if yes_ask else 0)
                    no_ask = self._parse_price(no_ask_raw) if no_ask_raw else (100 - yes_bid if yes_bid else 0)

                    with self._lock:
                        self._tickers[ticker] = {
                            "yes_bid": yes_bid,
                            "yes_ask": yes_ask,
                            "no_bid": no_bid,
                            "no_ask": no_ask,
                            "last_price": self._parse_price(msg.get("price_dollars") or msg.get("price")),
                            "ts": time.time(),
                        }

            elif msg_type == "fill":
                # Store fill events for the daemon to pick up
                msg = data.get("msg", {})
                with self._lock:
                    if not hasattr(self, '_fills'):
                        self._fills = []
                    self._fills.append({
                        "ticker": msg.get("market_ticker", ""),
                        "side": msg.get("side", ""),
                        "count": msg.get("count", 0),
                        "price": self._parse_price(msg.get("yes_price", msg.get("no_price"))),
                        "ts": time.time(),
                    })

        except Exception:
            pass

    def _on_error(self, ws, error):
        pass  # reconnect handled by _run_loop

    def _on_close(self, ws, close_status_code, close_msg):
        self._ws = None

    @staticmethod
    def _parse_price(val) -> int:
        """Parse price to cents. Handles both cents (int) and dollars (float)."""
        if val is None:
            return 0
        val = float(val)
        if val < 1.5:  # dollars format (0.50)
            return int(val * 100)
        return int(val)  # already cents

    def get_pending_fills(self) -> list[dict]:
        """Get and clear pending fill events. Thread-safe."""
        with self._lock:
            fills = getattr(self, '_fills', [])
            self._fills = []
            return fills
