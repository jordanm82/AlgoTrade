# tests/test_mm_daemon.py
"""Integration tests for MMAssetRunner state machine and dashboard."""
import time
import pytest
from unittest.mock import MagicMock, patch

from kalshi_mm.mm_config import (
    IDLE, DISCOVERING, QUOTING_BID, QUOTING_ASK, EXITING, DARK,
    VPIN_CAUTION, MAX_EXIT_LOSS_CENTS,
)
from kalshi_mm.mm_daemon import MMAssetRunner, render_dashboard


def _mock_client():
    """Create a mocked KalshiClient."""
    client = MagicMock()
    client.get_balance.return_value = {"balance": 10000}
    client.get_orderbook.return_value = {
        "orderbook_fp": {
            "yes_dollars": [["0.4800", "20"], ["0.4700", "15"]],
            "no_dollars": [["0.5000", "10"], ["0.5100", "8"]],
        }
    }
    client.get_markets.return_value = [
        {
            "ticker": "KXBTC15M-26MAR25-T08",
            "expiration_time": _expiry_in_minutes(8),
        }
    ]
    client.place_order.return_value = {"order": {"order_id": "ord-123", "status": "resting"}}
    client.cancel_order_safe.return_value = {"status": "cancelled"}
    client.get_order_status.return_value = {"order": {"status": "resting"}}
    return client


def _expiry_in_minutes(minutes: float) -> str:
    """Return ISO timestamp `minutes` from now."""
    from datetime import datetime, timezone, timedelta
    dt = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return dt.isoformat()


def _make_runner(client=None, dry_run=True) -> MMAssetRunner:
    """Create a test runner with mocked externals."""
    if client is None:
        client = _mock_client()
    runner = MMAssetRunner(
        symbol="BTC/USDT",
        series_ticker="KXBTC15M",
        kalshi_client=client,
        dry_run=dry_run,
    )
    return runner


class TestIdleToDiscovering:
    def test_initial_state_is_idle(self):
        runner = _make_runner()
        assert runner.state == IDLE

    def test_idle_transitions_to_discovering(self):
        runner = _make_runner()
        with patch.object(runner, "_compute_vpin"):
            runner.tick()
        # After IDLE tick, should be DISCOVERING (or further if discovery succeeds)
        assert runner.state in (DISCOVERING, QUOTING_BID)


class TestDiscoveringToQuotingBid:
    def test_discovers_market_and_transitions(self):
        runner = _make_runner()
        # Tick IDLE -> DISCOVERING
        runner.inv.state = DISCOVERING
        runner._last_discovery_ts = 0

        with patch.object(runner, "_compute_vpin"):
            runner.tick()

        assert runner.state == QUOTING_BID
        assert runner.inv.market_ticker == "KXBTC15M-26MAR25-T08"

    def test_no_market_stays_discovering(self):
        client = _mock_client()
        client.get_markets.return_value = []
        runner = _make_runner(client)
        runner.inv.state = DISCOVERING
        runner._last_discovery_ts = 0

        runner.tick()

        assert runner.state == DISCOVERING

    def test_empty_orderbook_stays_discovering(self):
        """Fix I8: reject markets with no depth."""
        client = _mock_client()
        client.get_orderbook.return_value = {
            "orderbook_fp": {"yes_dollars": [], "no_dollars": []}
        }
        runner = _make_runner(client)
        runner.inv.state = DISCOVERING
        runner._last_discovery_ts = 0

        runner.tick()

        assert runner.state == DISCOVERING
        assert "lacks depth" in runner.status_msg


class TestQuotingBidToAskOnFill:
    def test_bid_fill_transitions_to_quoting_ask(self):
        runner = _make_runner()
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600  # 10 min
        runner.inv.pending_bid_id = "dry-bid-123"
        runner.inv.bid_price_cents = 49

        # Volume-based fill: our bid is YES@49, so we need NO volume consumed
        # at (100-49)=51 or above. Set prev snapshot with volume at 51,
        # then current snapshot with less volume → consumed.
        runner._prev_no_bids = {51: 50, 52: 30}  # previous: 50 contracts at 51c NO

        fill_ob = {
            "orderbook_fp": {
                "yes_dollars": [["0.4800", "20"]],
                "no_dollars": [["0.5100", "20"], ["0.5200", "10"]],  # 51→20 (was 50, consumed 30), 52→10 (was 30, consumed 20)
            }
        }
        runner.client.get_orderbook.return_value = fill_ob

        with patch.object(runner, "_compute_vpin"):
            runner.tick()

        assert runner.state == QUOTING_ASK
        assert runner.inv.yes_held > 0
        assert runner.inv.entry_price_cents == 49

    def test_bid_no_fill_stays_quoting_bid(self):
        runner = _make_runner()
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600
        runner.inv.pending_bid_id = "dry-bid-123"
        runner.inv.bid_price_cents = 40

        # YES ask = 100 - 50 = 50, which is > 40 (no fill)
        no_fill_ob = {
            "orderbook_fp": {
                "yes_dollars": [["0.4800", "20"]],
                "no_dollars": [["0.5000", "10"]],
            }
        }
        runner.client.get_orderbook.return_value = no_fill_ob

        with patch.object(runner, "_compute_vpin"):
            runner.tick()

        assert runner.state == QUOTING_BID


class TestDarkOnToxicVpin:
    def test_toxic_vpin_sends_quoting_bid_to_dark(self):
        runner = _make_runner()
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600

        def set_toxic_vpin():
            runner.vpin = VPIN_CAUTION + 0.1

        with patch.object(runner, "_compute_vpin", side_effect=set_toxic_vpin):
            runner.tick()

        assert runner.state == DARK

    def test_toxic_vpin_sends_quoting_ask_to_exiting(self):
        runner = _make_runner()
        runner.inv.state = QUOTING_ASK
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600
        runner.inv.yes_held = 10
        runner.inv.entry_price_cents = 48

        def set_toxic_vpin():
            runner.vpin = VPIN_CAUTION + 0.1

        with patch.object(runner, "_compute_vpin", side_effect=set_toxic_vpin):
            runner.tick()

        assert runner.state == EXITING
        assert runner._exit_from_kill_switch is True

    def test_volatility_spike_sends_to_dark(self):
        """Fix I1: volatility spike triggers DARK."""
        runner = _make_runner()
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600

        def set_low_vpin_but_spike():
            runner.vpin = 0.1  # low VPIN
            runner.kill_switch.record_price(100.0)
            runner.kill_switch.record_price(101.0)  # 1% spike > 0.5% threshold

        with patch.object(runner, "_compute_vpin", side_effect=set_low_vpin_but_spike):
            runner.tick()

        assert runner.state == DARK


class TestAskCancelFillRace:
    """Fix C1: cancel returning 'filled' should NOT go to EXITING."""

    def test_ask_cancel_fill_race_toxic_vpin(self):
        runner = _make_runner()
        runner.inv.state = QUOTING_ASK
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600
        runner.inv.yes_held = 10
        runner.inv.entry_price_cents = 48
        runner.inv.pending_ask_id = "dry-ask-999"
        runner.inv.ask_price_cents = 51

        # Make cancel return "filled"
        runner.client.cancel_order_safe.return_value = {"status": "filled"}
        runner.dry_run = False  # need live mode to hit cancel_order_safe

        def set_toxic_vpin():
            runner.vpin = VPIN_CAUTION + 0.1

        with patch.object(runner, "_compute_vpin", side_effect=set_toxic_vpin):
            runner.tick()

        # Should NOT be EXITING — the ask was already filled
        assert runner.state != EXITING
        assert runner.state == QUOTING_BID
        # Inventory should be cleared (sell fill recorded)
        assert runner.inv.yes_held == 0

    def test_ask_cancel_fill_race_requote(self):
        """Cancel during re-quote returns filled."""
        runner = _make_runner()
        runner.inv.state = QUOTING_ASK
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600
        runner.inv.yes_held = 10
        runner.inv.entry_price_cents = 48
        runner.inv.pending_ask_id = "ask-old"
        runner.inv.ask_price_cents = 51

        # Make cancel return "filled"
        runner.client.cancel_order_safe.return_value = {"status": "filled"}
        runner.dry_run = False

        # Return an OB that produces a different ask (triggers re-quote)
        runner.client.get_orderbook.return_value = {
            "orderbook_fp": {
                "yes_dollars": [["0.4800", "20"]],
                "no_dollars": [["0.5000", "10"]],
            }
        }

        def set_safe_vpin():
            runner.vpin = 0.1

        with patch.object(runner, "_compute_vpin", side_effect=set_safe_vpin):
            runner.tick()

        assert runner.state == QUOTING_BID
        assert runner.inv.yes_held == 0


class TestExitingState:
    def test_exiting_no_inventory_goes_idle(self):
        runner = _make_runner()
        runner.inv.state = EXITING
        runner.inv.yes_held = 0
        runner._exit_from_kill_switch = False

        runner.tick()

        assert runner.state == IDLE

    def test_exiting_from_kill_switch_goes_dark(self):
        """Fix I4: exit from kill switch -> DARK, not IDLE."""
        runner = _make_runner()
        runner.inv.state = EXITING
        runner.inv.yes_held = 0
        runner._exit_from_kill_switch = True

        runner.tick()

        assert runner.state == DARK
        assert runner._exit_from_kill_switch is False

    def test_progressive_exit_pricing(self):
        """Fix I5: each exit attempt lowers price by 1 more cent."""
        runner = _make_runner()
        runner.inv.state = EXITING
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.yes_held = 10
        runner.inv.entry_price_cents = 50
        runner.inv.last_exit_attempt_ts = 0

        runner.tick()

        # First attempt: price = 50 - (1+1) = 48
        assert runner.inv.exit_attempt_count == 0  # reset after dry-run fill
        assert len(runner.trades) == 1
        assert runner.trades[0]["price_cents"] == 48


class TestWindowLossLimit:
    """Fix I2: window loss limit check."""

    def test_quoting_bid_window_loss_goes_dark(self):
        runner = _make_runner()
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600
        runner.inv.window_pnl_cents = -1001  # exceeds MAX_WINDOW_LOSS_CENTS

        with patch.object(runner, "_compute_vpin"):
            runner.tick()

        assert runner.state == DARK

    def test_quoting_ask_window_loss_goes_exiting(self):
        runner = _make_runner()
        runner.inv.state = QUOTING_ASK
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600
        runner.inv.yes_held = 10
        runner.inv.entry_price_cents = 48
        runner.inv.window_pnl_cents = -1001

        with patch.object(runner, "_compute_vpin"):
            runner.tick()

        assert runner.state == EXITING


class TestConsecutiveApiErrors:
    """Fix I3: go DARK after 2 consecutive API errors."""

    def test_two_errors_goes_dark(self):
        runner = _make_runner()
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600

        # Make _compute_vpin raise to trigger error path
        with patch.object(runner, "_tick_quoting_bid", side_effect=RuntimeError("API down")):
            runner.tick()  # error 1
            assert runner._consecutive_api_errors == 1
            assert runner.state == QUOTING_BID  # not dark yet

            runner.tick()  # error 2
            assert runner._consecutive_api_errors >= 2
            assert runner.state == DARK


class TestDarkState:
    def test_dark_recovers_when_flow_calms(self):
        runner = _make_runner()
        runner.inv.state = DARK

        def set_calm():
            runner.vpin = 0.1

        with patch.object(runner, "_compute_vpin", side_effect=set_calm):
            runner.tick()

        assert runner.state == IDLE

    def test_dark_stays_when_still_toxic(self):
        runner = _make_runner()
        runner.inv.state = DARK

        def set_toxic():
            runner.vpin = VPIN_CAUTION + 0.1

        with patch.object(runner, "_compute_vpin", side_effect=set_toxic):
            runner.tick()

        assert runner.state == DARK


class TestLiveModeBidFill:
    """Fix C3: query actual fill count in live mode."""

    def test_live_bid_fill_uses_order_status(self):
        client = _mock_client()
        client.get_order_status.return_value = {
            "order": {"status": "filled", "count": 15}
        }
        runner = _make_runner(client, dry_run=False)
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600
        runner.inv.pending_bid_id = "live-bid-123"
        runner.inv.bid_price_cents = 49

        def set_safe():
            runner.vpin = 0.1

        with patch.object(runner, "_compute_vpin", side_effect=set_safe):
            runner.tick()

        assert runner.state == QUOTING_ASK
        assert runner.inv.yes_held == 15  # from order status, not computed


class TestDashboard:
    def test_dashboard_renders(self):
        runner = _make_runner()
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.vpin = 0.25
        runner.status_msg = "test status"

        output = render_dashboard([runner], "dry-run", time.time() - 120, 10000)

        assert "[DRY-RUN]" in output
        assert "BTC/USDT" in output
        assert "QUOTING_BID" in output
        assert "$100.00" in output

    def test_dashboard_live_tag(self):
        runner = _make_runner()
        output = render_dashboard([runner], "live", time.time(), 5000)
        assert "[LIVE]" in output
        assert "$50.00" in output


class TestEntryPriceCapture:
    """Fix C2: entry_price captured before record_sell_fill zeroes it."""

    def test_exit_at_market_captures_entry(self):
        runner = _make_runner()
        runner.inv.state = EXITING
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.yes_held = 10
        runner.inv.entry_price_cents = 50
        runner.inv.last_exit_attempt_ts = 0

        runner.tick()

        assert len(runner.trades) == 1
        assert "entry=50" in runner.trades[0]["note"]
