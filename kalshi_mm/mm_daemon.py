# kalshi_mm/mm_daemon.py
"""Market maker daemon — state machine, dashboard, main loop.

State machine per asset:
    IDLE -> DISCOVERING -> QUOTING_BID -> QUOTING_ASK -> EXITING -> DARK -> IDLE
"""
import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from kalshi_mm.mm_config import (
    MM_ASSETS,
    IDLE, DISCOVERING, QUOTING_BID, QUOTING_ASK, EXITING, DARK,
    POLL_INTERVAL, QUOTE_WINDOW_MIN_MINUTES, QUOTE_WINDOW_MAX_MINUTES,
    HARD_CUTOFF_MINUTES, DISCOVERY_RETRY_SECONDS, FORCED_EXIT_RETRY_SECONDS,
    REQUOTE_DRIFT_CENTS, MAX_EXIT_LOSS_CENTS,
    VPIN_CAUTION,
)
from kalshi_mm.mm_inventory import MMInventory, compute_contracts, calc_round_trip_pnl
from kalshi_mm.mm_vpin import (
    compute_spot_vpin, compute_kalshi_ob_heuristic,
    compute_blended_vpin, KillSwitch,
)
from kalshi_mm.mm_strategy import (
    compute_mid_cents, compute_spread_cents, compute_bid_cents,
    parse_ob_as_dict, volume_consumed_at_or_above, volume_consumed_at_or_below,
    compute_ask_cents, parse_ob_total_volume,
)

logger = logging.getLogger("mm_daemon")

DATA_STORE = Path(__file__).resolve().parent.parent / "data" / "store"


# ---------------------------------------------------------------------------
# MMAssetRunner — one per asset (BTC, ETH)
# ---------------------------------------------------------------------------

class MMAssetRunner:
    """State-machine runner for one asset's market-making cycle."""

    def __init__(
        self,
        symbol: str,
        series_ticker: str,
        kalshi_client,
        dry_run: bool = True,
        safe_mode: bool = False,
    ):
        self.symbol = symbol
        self.series_ticker = series_ticker
        self.client = kalshi_client
        self.dry_run = dry_run
        self.safe_mode = safe_mode

        self.inv = MMInventory(asset=symbol)
        self.kill_switch = KillSwitch()
        self.vpin: float = 0.0

        # OB volume tracking for Kalshi heuristic
        self._prev_yes_vol: float = 0.0
        self._prev_no_vol: float = 0.0

        # OB snapshots for volume-based fill simulation (dry-run)
        self._prev_yes_bids: dict[int, int] = {}  # {price_cents: volume}
        self._prev_no_bids: dict[int, int] = {}   # {price_cents: volume}

        # Contract mid price tracking — detect rapid repricing
        self._prev_mid: int = 0

        # Discovery timing
        self._last_discovery_ts: float = 0.0

        # API error tracking (Fix I3)
        self._consecutive_api_errors: int = 0

        # Exit origin tracking (Fix I4)
        self._exit_from_kill_switch: bool = False

        # Trade log
        self.trades: list[dict] = []

        # Last status message for dashboard
        self.status_msg: str = ""

    # -- public interface ---------------------------------------------------

    @property
    def state(self) -> str:
        return self.inv.state

    def tick(self):
        """Dispatch to current state handler."""
        try:
            handler = {
                IDLE: self._tick_idle,
                DISCOVERING: self._tick_discovering,
                QUOTING_BID: self._tick_quoting_bid,
                QUOTING_ASK: self._tick_quoting_ask,
                EXITING: self._tick_exiting,
                DARK: self._tick_dark,
            }.get(self.inv.state, self._tick_idle)
            handler()
            self._consecutive_api_errors = 0  # reset on success
        except Exception as e:
            self._consecutive_api_errors += 1
            logger.warning(
                "%s tick error (%d consecutive): %s",
                self.symbol, self._consecutive_api_errors, e,
            )
            # Fix I3: go DARK after 2+ consecutive API errors
            if self._consecutive_api_errors >= 2:
                self.status_msg = f"DARK: {self._consecutive_api_errors} consecutive API errors"
                self.inv.state = DARK

    # -- state handlers -----------------------------------------------------

    def _tick_idle(self):
        """IDLE: check daily loss, then move to DISCOVERING."""
        if self.inv.is_daily_loss_hit():
            self.status_msg = "IDLE: daily loss limit hit"
            return
        self.inv.state = DISCOVERING
        self._last_discovery_ts = 0.0
        self.status_msg = "IDLE -> DISCOVERING"

    def _tick_discovering(self):
        """DISCOVERING: find a 15m market with 5-12 min to expiry."""
        now = time.time()
        if now - self._last_discovery_ts < DISCOVERY_RETRY_SECONDS:
            return
        self._last_discovery_ts = now

        market = self._find_market()
        if market is None:
            self.status_msg = "DISCOVERING: no suitable market"
            return

        # Fix I8: verify orderbook depth before transitioning
        ticker = market["ticker"]
        ob = self._get_orderbook(ticker)
        if ob is None:
            self.status_msg = "DISCOVERING: orderbook fetch failed"
            return

        ob_inner = ob.get("orderbook_fp", ob)
        yes_bids = ob_inner.get("yes_dollars", [])
        no_bids = ob_inner.get("no_dollars", [])
        if not yes_bids or not no_bids:
            self.status_msg = "DISCOVERING: orderbook lacks depth"
            return

        # Parse expiry
        # close_time = actual 15m window end; expiration_time = settlement (days later)
        exp_str = market.get("close_time") or market.get("expiration_time", "")
        try:
            exp_ts = datetime.fromisoformat(
                exp_str.replace("Z", "+00:00")
            ).timestamp()
        except (ValueError, AttributeError):
            self.status_msg = "DISCOVERING: cannot parse expiry"
            return

        self.inv.reset_window()
        self.inv.market_ticker = ticker
        self.inv.expiry_ts = exp_ts

        # Snapshot OB volumes for heuristic baseline
        self._prev_yes_vol = parse_ob_total_volume(yes_bids)
        self._prev_no_vol = parse_ob_total_volume(no_bids)

        # Initialize OB snapshots for volume-based fill simulation
        self._prev_yes_bids = parse_ob_as_dict(yes_bids)
        self._prev_no_bids = parse_ob_as_dict(no_bids)

        # Initialize mid for contract price volatility detection
        initial_mid = compute_mid_cents(ob)
        self._prev_mid = initial_mid or 0

        self.inv.state = QUOTING_BID
        self.status_msg = f"DISCOVERING -> QUOTING_BID ({ticker})"

    def _tick_quoting_bid(self):
        """QUOTING_BID: compute VPIN, quote bid, check fill."""
        # Fix I2: check window loss limit
        if self.inv.is_window_loss_hit():
            self.status_msg = "QUOTING_BID: window loss limit hit -> DARK"
            if self._safe_cancel_bid_or_manage():
                return  # bid was filled, now managing position
            self.inv.state = DARK
            return

        # Time check
        mins = self.inv.minutes_to_expiry()
        if mins < HARD_CUTOFF_MINUTES:
            self.status_msg = "QUOTING_BID: past hard cutoff -> IDLE"
            if self._safe_cancel_bid_or_manage():
                return
            self.inv.state = IDLE
            return

        # Compute VPIN
        self._compute_vpin()

        # Fix I1: check volatility spike alongside VPIN
        if self.kill_switch.should_go_dark(self.vpin) or self.kill_switch.volatility_spike():
            self.status_msg = f"QUOTING_BID: toxic flow (VPIN={self.vpin:.2f}) -> DARK"
            if self._safe_cancel_bid_or_manage():
                return
            self.inv.state = DARK
            return

        # Compute spread and bid
        spread = compute_spread_cents(self.vpin)
        if spread is None:
            self.status_msg = "QUOTING_BID: spread=None (toxic) -> DARK"
            if self._safe_cancel_bid_or_manage():
                return
            self.inv.state = DARK
            return

        ob = self._get_orderbook(self.inv.market_ticker)
        if ob is None:
            self.status_msg = "QUOTING_BID: orderbook fetch failed"
            return

        mid = compute_mid_cents(ob)
        if mid is None:
            self.status_msg = "QUOTING_BID: no mid price"
            return

        # Contract price volatility check — go dark if mid moved too fast
        from kalshi_mm.mm_config import MID_MOVE_DARK_CENTS
        if self._prev_mid > 0 and abs(mid - self._prev_mid) >= MID_MOVE_DARK_CENTS:
            self.status_msg = (
                f"QUOTING_BID: mid jumped {self._prev_mid}->{mid}c "
                f"({abs(mid - self._prev_mid)}c) -> DARK"
            )
            self._prev_mid = mid
            if self._safe_cancel_bid_or_manage():
                return
            self.inv.state = DARK
            return
        self._prev_mid = mid

        bid = compute_bid_cents(mid, spread)
        if bid is None:
            self.status_msg = "QUOTING_BID: bid out of bounds"
            return

        # Check if existing bid needs re-quoting
        if self.inv.pending_bid_id is not None:
            # Check fill first
            filled = self._check_bid_filled(ob)
            if filled:
                return  # state already transitioned

            # Only re-quote UPWARD (mid rose = we can bid higher = better fill chance).
            # Never chase price down — that's directional betting, not market making.
            if bid > self.inv.bid_price_cents and (bid - self.inv.bid_price_cents) >= REQUOTE_DRIFT_CENTS:
                cancel_result = self._cancel_pending_bid()
                if cancel_result and cancel_result.get("status") == "filled":
                    # Bid filled during cancel — treat as fill
                    self._handle_bid_fill_from_cancel(self.inv.pending_bid_id or "")
                    return
                self._place_bid(bid, ob)
            else:
                self.status_msg = (
                    f"QUOTING_BID: resting bid@{self.inv.bid_price_cents}c "
                    f"mid={mid} vpin={self.vpin:.2f}"
                )
        else:
            self._place_bid(bid, ob)

    def _tick_quoting_ask(self):
        """QUOTING_ASK: we hold inventory, quote ask to sell."""
        # Fix I2: check window loss limit
        if self.inv.is_window_loss_hit():
            self.status_msg = "QUOTING_ASK: window loss limit hit -> EXITING"
            self._cancel_pending_ask()
            self._exit_from_kill_switch = True
            self.inv.state = EXITING
            return

        # Time check — must exit before expiry
        mins = self.inv.minutes_to_expiry()
        if mins < HARD_CUTOFF_MINUTES:
            self.status_msg = "QUOTING_ASK: hard cutoff -> EXITING"
            self._cancel_pending_ask()
            self.inv.state = EXITING
            return

        # Compute VPIN
        self._compute_vpin()

        # Fix I1: check volatility spike alongside VPIN
        if self.kill_switch.should_go_dark(self.vpin) or self.kill_switch.volatility_spike():
            self.status_msg = f"QUOTING_ASK: toxic flow -> EXITING"
            # Fix C1: check cancel result for fill race
            cancel_result = self._cancel_pending_ask()
            if cancel_result and cancel_result.get("status") == "filled":
                self.inv.record_sell_fill(self.inv.ask_price_cents)
                self._log_trade("sell", self.inv.ask_price_cents, "ask-cancel-fill-race")
                self.inv.state = QUOTING_BID
                self.status_msg = "QUOTING_ASK: cancel returned filled -> QUOTING_BID"
                return
            self._exit_from_kill_switch = True
            self.inv.state = EXITING
            return

        # Contract price volatility check — exit if mid is crashing while we hold
        ob_for_mid = self._get_orderbook(self.inv.market_ticker)
        if ob_for_mid:
            mid = compute_mid_cents(ob_for_mid)
            if mid and self._prev_mid > 0 and abs(mid - self._prev_mid) >= MID_MOVE_DARK_CENTS:
                self.status_msg = (
                    f"QUOTING_ASK: mid jumped {self._prev_mid}->{mid}c -> EXITING"
                )
                self._prev_mid = mid
                cancel_result = self._cancel_pending_ask()
                if cancel_result and cancel_result.get("status") == "filled":
                    self.inv.record_sell_fill(self.inv.ask_price_cents)
                    self._log_trade("sell", self.inv.ask_price_cents, "ask-cancel-fill-mid-jump")
                    self.inv.state = IDLE
                    return
                self._exit_from_kill_switch = True
                self.inv.state = EXITING
                return
            if mid:
                self._prev_mid = mid

        # Compute spread and ask
        spread = compute_spread_cents(self.vpin)
        if spread is None:
            # Fix C1 on cancel
            cancel_result = self._cancel_pending_ask()
            if cancel_result and cancel_result.get("status") == "filled":
                self.inv.record_sell_fill(self.inv.ask_price_cents)
                self._log_trade("sell", self.inv.ask_price_cents, "ask-cancel-fill-race")
                self.inv.state = QUOTING_BID
                self.status_msg = "QUOTING_ASK: cancel-filled during toxic spread -> QUOTING_BID"
                return
            self._exit_from_kill_switch = True
            self.inv.state = EXITING
            self.status_msg = "QUOTING_ASK: spread=None -> EXITING"
            return

        ask = compute_ask_cents(self.inv.entry_price_cents, spread)
        if ask is None:
            self.status_msg = "QUOTING_ASK: ask out of bounds -> EXITING"
            cancel_result = self._cancel_pending_ask()
            if cancel_result and cancel_result.get("status") == "filled":
                self.inv.record_sell_fill(self.inv.ask_price_cents)
                self._log_trade("sell", self.inv.ask_price_cents, "ask-cancel-fill-race")
                self.inv.state = QUOTING_BID
                return
            self.inv.state = EXITING
            return

        ob = self._get_orderbook(self.inv.market_ticker)
        if ob is None:
            self.status_msg = "QUOTING_ASK: orderbook fetch failed"
            return

        if self.inv.pending_ask_id is not None:
            filled = self._check_ask_filled(ob)
            if filled:
                return

            # Re-quote if drifted
            if abs(ask - self.inv.ask_price_cents) >= REQUOTE_DRIFT_CENTS:
                # Fix C1: check cancel result
                cancel_result = self._cancel_pending_ask()
                if cancel_result and cancel_result.get("status") == "filled":
                    self.inv.record_sell_fill(self.inv.ask_price_cents)
                    self._log_trade("sell", self.inv.ask_price_cents, "ask-cancel-fill-race")
                    self.inv.state = QUOTING_BID
                    self.status_msg = "QUOTING_ASK: re-quote cancel returned filled"
                    return
                self._place_ask(ask)
            else:
                self.status_msg = (
                    f"QUOTING_ASK: resting ask@{self.inv.ask_price_cents}c "
                    f"entry={self.inv.entry_price_cents}c vpin={self.vpin:.2f}"
                )
        else:
            self._place_ask(ask)

    def _tick_exiting(self):
        """EXITING: forced sell to dump inventory before expiry."""
        if not self.inv.has_inventory():
            # Fix I4: go DARK if exit was from kill switch, else QUOTING_BID
            if self._exit_from_kill_switch:
                self._exit_from_kill_switch = False
                self.inv.state = DARK
                self.status_msg = "EXITING: done (kill switch) -> DARK"
            else:
                self.inv.state = IDLE
                self.status_msg = "EXITING: no inventory -> IDLE"
            return

        now = time.time()
        # Retry throttle
        if (now - self.inv.last_exit_attempt_ts) < FORCED_EXIT_RETRY_SECONDS:
            return

        self._exit_at_market()

    def _tick_dark(self):
        """DARK: wait until flow calms down, then resume."""
        self._compute_vpin()
        if self.vpin < VPIN_CAUTION and not self.kill_switch.volatility_spike():
            self.inv.state = IDLE
            self.status_msg = "DARK: flow calmed -> IDLE"
        else:
            self.status_msg = f"DARK: waiting (VPIN={self.vpin:.2f})"

    # -- VPIN computation ---------------------------------------------------

    def _compute_vpin(self):
        """Compute blended VPIN from spot flow + Kalshi OB heuristic.
        Fix I1: also record price for volatility spike detection.
        """
        try:
            from data.market_data import get_trade_flow
            flow = get_trade_flow(self.symbol, limit=200)
        except Exception:
            flow = {"net_flow": 0, "buy_ratio": 0.5}

        spot_vpin = compute_spot_vpin(flow)

        # Kalshi OB heuristic
        ob = self._get_orderbook(self.inv.market_ticker) if self.inv.market_ticker else None
        kalshi_h = 0.0
        if ob is not None:
            ob_inner = ob.get("orderbook_fp", ob)
            yes_bids = ob_inner.get("yes_dollars", [])
            no_bids = ob_inner.get("no_dollars", [])
            curr_yes_vol = parse_ob_total_volume(yes_bids) if yes_bids else 0
            curr_no_vol = parse_ob_total_volume(no_bids) if no_bids else 0
            kalshi_h = compute_kalshi_ob_heuristic(
                self._prev_yes_vol, curr_yes_vol,
                self._prev_no_vol, curr_no_vol,
            )
            self._prev_yes_vol = curr_yes_vol
            self._prev_no_vol = curr_no_vol

        self.vpin = compute_blended_vpin(spot_vpin, kalshi_h)
        self.kill_switch.record_vpin(self.vpin)

        # Fix I1: record spot price for volatility spike detection
        try:
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()
            tick = fetcher.ticker(self.symbol)
            if tick and "last" in tick:
                self.kill_switch.record_price(tick["last"])
        except Exception:
            pass

    # -- market discovery ---------------------------------------------------

    def _find_market(self) -> dict | None:
        """Find a 15m market with 5-12 min to expiry."""
        try:
            markets = self.client.get_markets(series_ticker=self.series_ticker)
        except Exception as e:
            logger.warning("%s market discovery failed: %s", self.symbol, e)
            return None

        now = time.time()
        best = None
        best_mins = 999.0

        for m in markets:
            # close_time = actual 15m window end; expiration_time = settlement (days later)
            exp_str = m.get("close_time") or m.get("expiration_time", "")
            if not exp_str:
                continue
            try:
                exp_ts = datetime.fromisoformat(
                    exp_str.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, AttributeError):
                continue

            mins = (exp_ts - now) / 60
            if QUOTE_WINDOW_MIN_MINUTES <= mins <= QUOTE_WINDOW_MAX_MINUTES:
                if mins < best_mins:
                    best = m
                    best_mins = mins

        return best

    # -- order placement ----------------------------------------------------

    def _place_bid(self, bid_cents: int, ob: dict):
        """Place a YES limit buy (bid)."""
        balance = self._get_balance_cents()
        if balance is None:
            self.status_msg = "QUOTING_BID: balance fetch failed"
            return

        contracts = compute_contracts(balance, bid_cents)
        if contracts is None:
            self.status_msg = "QUOTING_BID: insufficient budget for bid"
            return

        if self.safe_mode:
            contracts = 1

        if self.dry_run:
            # Simulate: bid rests immediately
            order_id = f"dry-bid-{int(time.time()*1000)}"
            self.inv.pending_bid_id = order_id
            self.inv.bid_price_cents = bid_cents
            self.status_msg = f"QUOTING_BID: [DRY] bid@{bid_cents}c x{contracts}"
        else:
            try:
                resp = self.client.place_order(
                    self.inv.market_ticker, "yes", contracts,
                    price_cents=bid_cents, order_type="limit", action="buy",
                )
                order = resp.get("order", resp)
                self.inv.pending_bid_id = order.get("order_id", order.get("id"))
                self.inv.bid_price_cents = bid_cents
                self.status_msg = f"QUOTING_BID: bid@{bid_cents}c x{contracts}"
            except Exception as e:
                logger.warning("%s place bid failed: %s", self.symbol, e)
                raise

    def _place_ask(self, ask_cents: int):
        """Place a YES limit sell (ask)."""
        if self.dry_run:
            order_id = f"dry-ask-{int(time.time()*1000)}"
            self.inv.pending_ask_id = order_id
            self.inv.ask_price_cents = ask_cents
            self.status_msg = (
                f"QUOTING_ASK: [DRY] ask@{ask_cents}c "
                f"x{self.inv.yes_held}"
            )
        else:
            try:
                resp = self.client.place_order(
                    self.inv.market_ticker, "yes", self.inv.yes_held,
                    price_cents=ask_cents, order_type="limit", action="sell",
                )
                order = resp.get("order", resp)
                self.inv.pending_ask_id = order.get("order_id", order.get("id"))
                self.inv.ask_price_cents = ask_cents
                self.status_msg = (
                    f"QUOTING_ASK: ask@{ask_cents}c x{self.inv.yes_held}"
                )
            except Exception as e:
                logger.warning("%s place ask failed: %s", self.symbol, e)
                raise

    def _place_exit_sell(self, price_cents: int):
        """Place a forced exit sell at a potentially losing price."""
        if self.dry_run:
            order_id = f"dry-exit-{int(time.time()*1000)}"
            self.inv.pending_ask_id = order_id
            self.inv.ask_price_cents = price_cents
            # In dry-run, simulate immediate fill
            # Fix C2: capture entry_price before record_sell_fill zeroes it
            entry = self.inv.entry_price_cents
            rt = self.inv.record_sell_fill(price_cents)
            self._log_trade("exit-sell", price_cents, f"forced-exit entry={entry}")
            self.status_msg = (
                f"EXITING: [DRY] exit@{price_cents}c pnl={rt['pnl_cents']}c"
            )
        else:
            try:
                resp = self.client.place_order(
                    self.inv.market_ticker, "yes", self.inv.yes_held,
                    price_cents=price_cents, order_type="limit", action="sell",
                )
                order = resp.get("order", resp)
                self.inv.pending_ask_id = order.get("order_id", order.get("id"))
                self.inv.ask_price_cents = price_cents
                self.status_msg = f"EXITING: exit sell@{price_cents}c"
            except Exception as e:
                logger.warning("%s exit sell failed: %s", self.symbol, e)
                raise

    def _exit_at_market(self):
        """Forced exit with progressive pricing (Fix I5)."""
        self.inv.exit_attempt_count += 1
        self.inv.last_exit_attempt_ts = time.time()

        # Fix I5: progressive pricing — entry - (attempt+1), clamped
        exit_price = self.inv.entry_price_cents - (self.inv.exit_attempt_count + 1)
        min_price = self.inv.entry_price_cents - MAX_EXIT_LOSS_CENTS
        exit_price = max(exit_price, min_price, 1)

        if self.dry_run:
            # Fix C2: capture entry_price before record_sell_fill zeroes it
            entry = self.inv.entry_price_cents
            rt = self.inv.record_sell_fill(exit_price)
            self._log_trade("exit-market", exit_price, f"forced entry={entry}")
            self.status_msg = f"EXITING: [DRY] market exit@{exit_price}c pnl={rt['pnl_cents']}c"
        else:
            try:
                # Fix C2: capture entry_price before record_sell_fill zeroes it
                entry = self.inv.entry_price_cents
                resp = self.client.place_order(
                    self.inv.market_ticker, "yes", self.inv.yes_held,
                    price_cents=exit_price, order_type="limit", action="sell",
                )
                order = resp.get("order", resp)
                # Check if filled immediately
                status = order.get("status", "")
                if status == "filled":
                    rt = self.inv.record_sell_fill(exit_price)
                    self._log_trade("exit-market", exit_price, f"forced entry={entry}")
                    self.status_msg = f"EXITING: exit filled@{exit_price}c"
                else:
                    self.inv.pending_ask_id = order.get("order_id", order.get("id"))
                    self.inv.ask_price_cents = exit_price
                    self.status_msg = f"EXITING: exit resting@{exit_price}c attempt#{self.inv.exit_attempt_count}"
            except Exception as e:
                logger.warning("%s exit_at_market failed: %s", self.symbol, e)
                raise

    # -- fill checks --------------------------------------------------------

    def _check_bid_filled(self, ob: dict) -> bool:
        """Check if our bid was filled. Returns True if filled."""
        if self.inv.pending_bid_id is None:
            return False

        if self.dry_run:
            # Volume-based fill simulation:
            # Our bid is BUY YES at X. For this to fill, someone must SELL YES at X.
            # Selling YES = buying NO at (100-X). So if NO bid volume decreased
            # at (100-X) or above, contracts were traded → our bid would fill.
            ob_inner = ob.get("orderbook_fp", ob)
            no_bids_raw = ob_inner.get("no_dollars", [])
            curr_no_bids = parse_ob_as_dict(no_bids_raw)

            # The NO price corresponding to our YES bid
            no_price_for_our_bid = 100 - self.inv.bid_price_cents

            # Check if volume was consumed at or above that NO price
            consumed = volume_consumed_at_or_above(
                self._prev_no_bids, curr_no_bids, no_price_for_our_bid
            )

            # Update snapshot for next poll
            self._prev_no_bids = curr_no_bids

            if consumed > 0:
                contracts = compute_contracts(
                    self._get_balance_cents() or 10000,
                    self.inv.bid_price_cents,
                )
                if contracts is None:
                    contracts = 1
                # Fill at most what was consumed
                contracts = min(contracts, consumed)
                self.inv.record_buy_fill(
                    contracts, self.inv.bid_price_cents, self.inv.pending_bid_id,
                )
                self._log_trade("buy", self.inv.bid_price_cents,
                                f"bid-filled-dry vol={consumed}")
                self.inv.state = QUOTING_ASK
                self.status_msg = (
                    f"QUOTING_BID: [DRY] bid filled@{self.inv.entry_price_cents}c"
                    f" x{self.inv.yes_held} (vol={consumed}) -> QUOTING_ASK"
                )
                return True
            return False
        else:
            # Live: check order status
            try:
                resp = self.client.get_order_status(self.inv.pending_bid_id)
                order = resp.get("order", resp)
                status = order.get("status", "")
                if status == "filled":
                    # Fix C3: query actual fill count in live mode
                    fill_count = order.get("count", order.get("filled_count", 0))
                    if not fill_count:
                        fill_count = compute_contracts(
                            self._get_balance_cents() or 10000,
                            self.inv.bid_price_cents,
                        ) or 10
                    self.inv.record_buy_fill(
                        fill_count, self.inv.bid_price_cents, self.inv.pending_bid_id,
                    )
                    self._log_trade("buy", self.inv.bid_price_cents, "bid-filled-live")
                    self.inv.state = QUOTING_ASK
                    self.status_msg = f"QUOTING_BID: bid filled -> QUOTING_ASK"
                    return True
            except Exception as e:
                logger.warning("%s check bid fill failed: %s", self.symbol, e)
            return False

    def _check_ask_filled(self, ob: dict) -> bool:
        """Check if our ask was filled. Returns True if filled."""
        if self.inv.pending_ask_id is None:
            return False

        if self.dry_run:
            # Volume-based fill simulation:
            # Our ask is SELL YES at X. For this to fill, someone must BUY YES at X.
            # If YES bid volume decreased at X or above, contracts were bought → our ask fills.
            ob_inner = ob.get("orderbook_fp", ob)
            yes_bids_raw = ob_inner.get("yes_dollars", [])
            curr_yes_bids = parse_ob_as_dict(yes_bids_raw)

            consumed = volume_consumed_at_or_above(
                self._prev_yes_bids, curr_yes_bids, self.inv.ask_price_cents
            )

            # Update snapshot for next poll
            self._prev_yes_bids = curr_yes_bids

            if consumed > 0:
                rt = self.inv.record_sell_fill(self.inv.ask_price_cents)
                self._log_trade(
                    "sell", self.inv.ask_price_cents,
                    f"ask-filled-dry vol={consumed} pnl={rt['pnl_cents']}c",
                )
                self.inv.state = QUOTING_BID
                self.status_msg = (
                    f"QUOTING_ASK: [DRY] ask filled@{self.inv.ask_price_cents}c "
                    f"vol={consumed} pnl={rt['pnl_cents']}c -> QUOTING_BID"
                )
                return True
            return False
        else:
            try:
                resp = self.client.get_order_status(self.inv.pending_ask_id)
                order = resp.get("order", resp)
                status = order.get("status", "")
                if status == "filled":
                    rt = self.inv.record_sell_fill(self.inv.ask_price_cents)
                    self._log_trade(
                        "sell", self.inv.ask_price_cents,
                        f"ask-filled-live pnl={rt['pnl_cents']}c",
                    )
                    self.inv.state = QUOTING_BID
                    self.status_msg = f"QUOTING_ASK: ask filled -> QUOTING_BID"
                    return True
            except Exception as e:
                logger.warning("%s check ask fill failed: %s", self.symbol, e)
            return False

    # -- cancel helpers -----------------------------------------------------

    def _cancel_pending_bid(self) -> dict | None:
        """Check if bid filled, then cancel if still resting.

        Returns:
            {"status": "filled"} if already filled (caller must handle inventory)
            {"status": "cancelled"} if successfully cancelled
            None on error
        """
        if self.inv.pending_bid_id is None:
            return None
        if self.dry_run:
            self.inv.pending_bid_id = None
            return {"status": "cancelled"}
        try:
            # Check fill status BEFORE attempting cancel
            resp = self.client.get_order_status(self.inv.pending_bid_id)
            order = resp.get("order", resp)
            status = order.get("status", "")
            if status == "filled":
                logger.info("%s bid %s already filled — must manage position",
                            self.symbol, self.inv.pending_bid_id)
                return {"status": "filled", "order": order}
            # Still resting — safe to cancel
            result = self.client.cancel_order_safe(self.inv.pending_bid_id)
            self.inv.pending_bid_id = None
            return result
        except Exception as e:
            logger.warning("%s cancel bid failed: %s", self.symbol, e)
            self.inv.pending_bid_id = None
            return None

    def _cancel_pending_ask(self) -> dict | None:
        """Check if ask filled, then cancel if still resting.

        Returns:
            {"status": "filled"} if already filled
            {"status": "cancelled"} if successfully cancelled
            None on error
        """
        if self.inv.pending_ask_id is None:
            return None
        if self.dry_run:
            self.inv.pending_ask_id = None
            return {"status": "cancelled"}
        try:
            # Check fill status BEFORE attempting cancel
            resp = self.client.get_order_status(self.inv.pending_ask_id)
            order = resp.get("order", resp)
            status = order.get("status", "")
            if status == "filled":
                logger.info("%s ask %s already filled — round trip complete",
                            self.symbol, self.inv.pending_ask_id)
                return {"status": "filled", "order": order}
            # Still resting — safe to cancel
            result = self.client.cancel_order_safe(self.inv.pending_ask_id)
            self.inv.pending_ask_id = None
            return result
        except Exception as e:
            logger.warning("%s cancel ask failed: %s", self.symbol, e)
            self.inv.pending_ask_id = None
            return None

    def _safe_cancel_bid_or_manage(self) -> bool:
        """Cancel bid, but if already filled, transition to QUOTING_ASK.

        Returns True if bid was filled (caller should return, state is now QUOTING_ASK).
        Returns False if bid was cancelled or no bid pending (caller continues normally).
        """
        result = self._cancel_pending_bid()
        if result and result.get("status") == "filled":
            order = result.get("order", {})
            fill_count = order.get("count", 1) if not self.dry_run else 1
            if self.safe_mode:
                fill_count = 1
            self.inv.record_buy_fill(fill_count, self.inv.bid_price_cents)
            self._log_trade("buy", self.inv.bid_price_cents, "bid-filled-on-cancel")
            # Post ask to try to profit, or at least exit
            spread = compute_spread_cents(self.vpin) or 2
            ask = compute_ask_cents(self.inv.entry_price_cents, spread)
            if ask:
                self._place_ask(ask)
                self.inv.state = QUOTING_ASK
                self.status_msg = f"bid filled on cancel — managing position, ask@{ask}c"
            else:
                self.inv.state = EXITING
                self.status_msg = "bid filled on cancel — ask invalid, exiting"
            return True
        return False

    # -- handle bid fill from cancel (Fix C3) --------------------------------

    def _handle_bid_fill_from_cancel(self, original_bid_id: str):
        """When a bid cancel returns 'filled', record the fill."""
        if self.dry_run:
            contracts = compute_contracts(
                self._get_balance_cents() or 10000,
                self.inv.bid_price_cents,
            ) or 10
        else:
            # Fix C3: query actual fill count in live mode
            try:
                resp = self.client.get_order_status(original_bid_id)
                order = resp.get("order", resp)
                contracts = order.get("count", order.get("filled_count", 0))
                if not contracts:
                    contracts = compute_contracts(
                        self._get_balance_cents() or 10000,
                        self.inv.bid_price_cents,
                    ) or 10
            except Exception:
                contracts = compute_contracts(
                    self._get_balance_cents() or 10000,
                    self.inv.bid_price_cents,
                ) or 10

        self.inv.record_buy_fill(contracts, self.inv.bid_price_cents, original_bid_id)
        self._log_trade("buy", self.inv.bid_price_cents, "bid-cancel-fill-race")
        self.inv.state = QUOTING_ASK

    # -- balance / orderbook helpers ----------------------------------------

    def _get_balance_cents(self) -> int | None:
        """Get Kalshi balance in cents (real balance even in dry-run)."""
        try:
            resp = self.client.get_balance()
            return resp.get("balance", 0)
        except Exception as e:
            logger.warning("%s balance fetch failed: %s", self.symbol, e)
            return None

    def _get_orderbook(self, ticker: str) -> dict | None:
        """Fetch orderbook for a market ticker."""
        if not ticker:
            return None
        try:
            return self.client.get_orderbook(ticker)
        except Exception as e:
            logger.warning("%s orderbook fetch failed: %s", self.symbol, e)
            return None

    # -- trade logging ------------------------------------------------------

    def _log_trade(self, side: str, price_cents: int, note: str = ""):
        """Append to internal trade log."""
        trade = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "asset": self.symbol,
            "market": self.inv.market_ticker,
            "side": side,
            "price_cents": price_cents,
            "contracts": self.inv.yes_held if side == "buy" else 0,
            "note": note,
        }
        self.trades.append(trade)
        logger.info("TRADE: %s", trade)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(
    runners: list[MMAssetRunner],
    mode: str,
    start_time: float,
    balance_cents: int,
) -> str:
    """Render ASCII dashboard for the market maker."""
    now = time.time()
    uptime = int(now - start_time)
    h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60

    tag = "[DRY-RUN]" if mode == "dry-run" else "[LIVE]"

    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"  Kalshi Market Maker {tag}")
    lines.append(f"  Uptime: {h:02d}:{m:02d}:{s:02d}  |  Balance: ${balance_cents / 100:.2f}")
    lines.append(f"{'=' * 60}")

    for r in runners:
        inv = r.inv
        pnl_str = f"{inv.window_pnl_cents:+d}c" if inv.window_pnl_cents != 0 else "0c"
        day_pnl_str = f"{inv.day_pnl_cents:+d}c"
        mins_exp = inv.minutes_to_expiry()
        exp_str = f"{mins_exp:.1f}m" if mins_exp < 900 else "--"
        held_str = f"{inv.yes_held}@{inv.entry_price_cents}c" if inv.has_inventory() else "flat"

        lines.append(f"  {r.symbol:<10} {inv.state:<14} VPIN={r.vpin:.2f}")
        lines.append(f"    Market: {inv.market_ticker or '---':<20} Exp: {exp_str}")
        lines.append(f"    Pos: {held_str:<16} Win PnL: {pnl_str:<8} Day PnL: {day_pnl_str}")
        lines.append(f"    Trades: {inv.window_round_trips}  |  {r.status_msg}")
        lines.append(f"  {'-' * 56}")

    total_day_pnl = sum(r.inv.day_pnl_cents for r in runners)
    total_trades = sum(len(r.trades) for r in runners)
    lines.append(f"  Total Day PnL: {total_day_pnl:+d}c  |  Total Trades: {total_trades}")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence (Fix I6)
# ---------------------------------------------------------------------------

def _write_trades_csv(runners: list[MMAssetRunner]):
    """Append new trades to mm_trades.csv."""
    csv_path = DATA_STORE / "mm_trades.csv"
    write_header = not csv_path.exists()
    all_trades = []
    for r in runners:
        all_trades.extend(r.trades)
    if not all_trades:
        return
    DATA_STORE.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ts", "asset", "market", "side", "price_cents", "contracts", "note"],
        )
        if write_header:
            writer.writeheader()
        for t in all_trades:
            writer.writerow(t)


def _write_session_json(runners: list[MMAssetRunner], mode: str, start_time: float, balance_cents: int):
    """Write mm_session.json snapshot."""
    DATA_STORE.mkdir(parents=True, exist_ok=True)
    session = {
        "mode": mode,
        "start_time": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
        "balance_cents": balance_cents,
        "runners": [],
    }
    for r in runners:
        inv = r.inv
        session["runners"].append({
            "asset": r.symbol,
            "state": inv.state,
            "market_ticker": inv.market_ticker,
            "yes_held": inv.yes_held,
            "entry_price_cents": inv.entry_price_cents,
            "window_pnl_cents": inv.window_pnl_cents,
            "day_pnl_cents": inv.day_pnl_cents,
            "window_round_trips": inv.window_round_trips,
            "vpin": round(r.vpin, 4),
            "total_trades": len(r.trades),
        })
    with open(DATA_STORE / "mm_session.json", "w") as f:
        json.dump(session, f, indent=2)


def _append_vpin_csv(runners: list[MMAssetRunner]):
    """Append VPIN readings to mm_vpin.csv."""
    csv_path = DATA_STORE / "mm_vpin.csv"
    write_header = not csv_path.exists()
    DATA_STORE.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "asset", "vpin", "state"])
        if write_header:
            writer.writeheader()
        ts = datetime.now(timezone.utc).isoformat()
        for r in runners:
            writer.writerow({
                "ts": ts,
                "asset": r.symbol,
                "vpin": round(r.vpin, 4),
                "state": r.inv.state,
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point: parse args, build runners, run poll loop."""
    parser = argparse.ArgumentParser(description="Kalshi Market Maker Daemon")
    parser.add_argument(
        "--mode", choices=["live", "dry-run"], default="dry-run",
        help="Trading mode (default: dry-run)",
    )
    parser.add_argument(
        "--cycles", type=int, default=0,
        help="Number of poll cycles (0 = infinite)",
    )
    parser.add_argument(
        "--safe-mode", action="store_true",
        help="Stop after first profitable round trip (safety test mode)",
    )
    args = parser.parse_args()

    dry_run = args.mode == "dry-run"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Build Kalshi client
    from exchange.kalshi import KalshiClient
    from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
    client = KalshiClient(KALSHI_API_KEY_ID, str(KALSHI_KEY_FILE))

    # Build runners for each asset
    runners: list[MMAssetRunner] = []
    for symbol, series in MM_ASSETS.items():
        runner = MMAssetRunner(
            symbol=symbol,
            series_ticker=series,
            kalshi_client=client,
            dry_run=dry_run,
            safe_mode=args.safe_mode,
        )
        runners.append(runner)

    start_time = time.time()
    cycle = 0
    tag = "[DRY-RUN]" if dry_run else "[LIVE]"
    logger.info("Market Maker started %s — %d assets", tag, len(runners))

    try:
        while True:
            cycle += 1
            if args.cycles > 0 and cycle > args.cycles:
                break

            # Tick runners with position limiting.
            # Safe mode: only tick ONE runner per cycle. Pick the first one
            # that has capital deployed, or the first idle one if none do.
            # Normal mode: up to MAX_CONCURRENT_POSITIONS runners active.
            from kalshi_mm.mm_config import MAX_CONCURRENT_POSITIONS

            if args.safe_mode:
                # Safe mode: strictly one runner. Find the one with capital,
                # or pick the first idle one to advance.
                active_runner = None
                for r in runners:
                    if r.inv.has_inventory() or r.inv.pending_bid_id is not None:
                        active_runner = r
                        break
                if active_runner:
                    active_runner.tick()
                else:
                    # Nobody has capital — tick just the first runner
                    runners[0].tick()
            else:
                # Normal mode: max N concurrent
                def _has_capital(r):
                    return r.inv.has_inventory() or r.inv.pending_bid_id is not None

                active_count = sum(1 for r in runners if _has_capital(r))

                for runner in runners:
                    if _has_capital(runner):
                        runner.tick()
                    elif runner.inv.state == EXITING:
                        runner.tick()
                    elif active_count < MAX_CONCURRENT_POSITIONS:
                        runner.tick()
                        if _has_capital(runner):
                            active_count += 1

            # Get balance for dashboard
            balance_cents = 10000 if dry_run else _fetch_balance(client)

            # Render dashboard
            dashboard = render_dashboard(runners, args.mode, start_time, balance_cents)
            print("\033[2J\033[H" + dashboard, flush=True)

            # Fix I6: persistence
            _write_trades_csv(runners)
            _write_session_json(runners, args.mode, start_time, balance_cents)
            _append_vpin_csv(runners)

            # Safe mode: stop after first profitable round trip
            if args.safe_mode:
                total_day_pnl = sum(r.inv.day_pnl_cents for r in runners)
                total_rts = sum(r.inv.window_round_trips for r in runners)
                if total_rts > 0 and total_day_pnl > 0:
                    logger.info("SAFE MODE: profitable round trip completed (pnl=+%dc). Stopping.", total_day_pnl)
                    # Cancel any resting orders before exit
                    for r in runners:
                        r._cancel_pending_bid()
                        r._cancel_pending_ask()
                    break

            # Clear trade logs after persisting (avoid re-writing)
            for runner in runners:
                runner.trades.clear()

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Market Maker stopped by user")
    finally:
        logger.info("Market Maker shutdown — writing final state")
        _write_session_json(runners, args.mode, start_time, balance_cents if 'balance_cents' in dir() else 0)


def _fetch_balance(client) -> int:
    """Fetch balance with fallback."""
    try:
        resp = client.get_balance()
        return resp.get("balance", 0)
    except Exception:
        return 0


if __name__ == "__main__":
    main()
