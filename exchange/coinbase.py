import uuid
from coinbase.rest import RESTClient


class CoinbaseExecutor:
    def __init__(self, key_file: str):
        self.client = RESTClient(key_file=key_file)

    def _order_id(self) -> str:
        return str(uuid.uuid4())

    def market_buy(self, symbol: str, usd_amount: float) -> dict:
        """Market buy by USD amount (spot)."""
        resp = self.client.market_order_buy(
            client_order_id=self._order_id(),
            product_id=symbol,
            quote_size=str(usd_amount),
        )
        return resp.to_dict() if hasattr(resp, "to_dict") else resp

    def market_sell(self, symbol: str, base_size: float) -> dict:
        """Market sell by base amount (spot)."""
        resp = self.client.market_order_sell(
            client_order_id=self._order_id(),
            product_id=symbol,
            base_size=str(base_size),
        )
        return resp.to_dict() if hasattr(resp, "to_dict") else resp

    def limit_buy(self, symbol: str, base_size: float, price: float) -> dict:
        """Limit buy order (GTC)."""
        resp = self.client.limit_order_gtc_buy(
            client_order_id=self._order_id(),
            product_id=symbol,
            base_size=str(base_size),
            limit_price=str(price),
        )
        return resp.to_dict() if hasattr(resp, "to_dict") else resp

    def limit_sell(self, symbol: str, base_size: float, price: float) -> dict:
        """Limit sell order (GTC)."""
        resp = self.client.limit_order_gtc_sell(
            client_order_id=self._order_id(),
            product_id=symbol,
            base_size=str(base_size),
            limit_price=str(price),
        )
        return resp.to_dict() if hasattr(resp, "to_dict") else resp

    def open_perp_long(self, symbol: str, base_size: float, leverage: int = 1) -> dict:
        """Open a perpetual long position."""
        resp = self.client.market_order_buy(
            client_order_id=self._order_id(),
            product_id=symbol,
            base_size=str(base_size),
            leverage=str(leverage),
            margin_type="CROSS",
        )
        return resp.to_dict() if hasattr(resp, "to_dict") else resp

    def open_perp_short(self, symbol: str, base_size: float, leverage: int = 1) -> dict:
        """Open a perpetual short position."""
        resp = self.client.market_order_sell(
            client_order_id=self._order_id(),
            product_id=symbol,
            base_size=str(base_size),
            leverage=str(leverage),
            margin_type="CROSS",
        )
        return resp.to_dict() if hasattr(resp, "to_dict") else resp

    def close_perp(self, symbol: str, size: str | None = None) -> dict:
        """Close a perpetual position."""
        resp = self.client.close_position(
            client_order_id=self._order_id(),
            product_id=symbol,
            size=size,
        )
        return resp.to_dict() if hasattr(resp, "to_dict") else resp

    def get_balances(self) -> dict[str, float]:
        """Get account balances (non-zero only)."""
        resp = self.client.get_accounts()
        balances = {}
        for acct in resp["accounts"]:
            val = float(acct["available_balance"]["value"])
            if val > 0:
                balances[acct["currency"]] = val
        return balances

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """List open orders, optionally filtered by symbol."""
        kwargs = {"order_status": ["OPEN"]}
        if symbol:
            kwargs["product_ids"] = [symbol]
        resp = self.client.list_orders(**kwargs)
        return resp.get("orders", [])

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an order by ID."""
        resp = self.client.cancel_orders(order_ids=[order_id])
        return resp.to_dict() if hasattr(resp, "to_dict") else resp
