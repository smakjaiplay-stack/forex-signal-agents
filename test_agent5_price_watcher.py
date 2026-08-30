"""
Unit tests for Agent 5 - Price Watcher.

Agent 5 decides what actually happened to a trade, and the R-multiple it writes
into trades_log.jsonl is the only out-of-sample number this project will ever
have. It had no tests. The TIMEOUT path in particular was carrying 65% of all
exits under the old trade geometry with nothing exercising it.

No network: fetch_recent_bars is replaced with a synthetic frame per test.

    python -m unittest test_agent5_price_watcher -v
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

import agent5_price_watcher as a5


# Relative to now, because watch() times a trade out against the wall clock:
# a fixture pinned to a past date would come back TIMEOUT from every test.
OPENED = datetime.now(timezone.utc) - timedelta(hours=1)


def bars(rows, start=None):
    """rows = [(high, low, close), ...] on 5-minute spacing."""
    start = start or (OPENED + timedelta(minutes=5))
    index = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, columns=["High", "Low", "Close"], index=index)


def make_trade(**overrides):
    """A 1R/2R/3R Buy on a 100-pip risk, as Agent 3 now builds them."""
    trade = {
        "pair": "EURUSD",
        "action": "Buy",
        "entry_price": 1.10000,
        "take_profit_1": 1.10100,
        "take_profit_2": 1.10200,
        "take_profit_3": 1.10300,
        "stop_loss": 1.09900,
        "current_sl": 1.09900,
        "opened_at": OPENED.isoformat(),
        "closed": False,
        "alerts_sent": {"tp1": False, "tp2": False, "tp3": False, "sl": False},
        "score": 1.83,
        "components": {"ema_sep": 0.5},
    }
    trade.update(overrides)
    return trade


class WatchHarness(unittest.TestCase):
    """Runs watch() against synthetic bars, capturing alerts and the log."""

    def setUp(self):
        self.alerts = []
        fd, self.log_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.addCleanup(os.unlink, self.log_path)

        self._real_fetch = a5.fetch_recent_bars
        self._real_notify = a5.notify
        a5.notify = lambda token, text, dry_run: self.alerts.append(text)
        self.addCleanup(self._restore)

    def _restore(self):
        a5.fetch_recent_bars = self._real_fetch
        a5.notify = self._real_notify

    def run_watch(self, frame, trade=None):
        a5.fetch_recent_bars = lambda symbol, last_checked=None: frame
        return a5.watch(trade or make_trade(), None, True, self.log_path)

    def logged(self):
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class TestStopLoss(WatchHarness):
    def test_the_stop_closes_the_trade_at_minus_one_r(self):
        trade = self.run_watch(bars([(1.10010, 1.09890, 1.09895)]))
        self.assertTrue(trade["closed"])
        self.assertEqual(trade["outcome"], "SL")
        self.assertAlmostEqual(trade["r_multiple"], -1.0, places=3)

    def test_a_sell_stops_out_on_the_high(self):
        sell = make_trade(action="Sell", take_profit_1=1.09900, take_profit_2=1.09800,
                          take_profit_3=1.09700, stop_loss=1.10100, current_sl=1.10100)
        trade = self.run_watch(bars([(1.10110, 1.09990, 1.10105)]), sell)
        self.assertEqual(trade["outcome"], "SL")
        self.assertAlmostEqual(trade["r_multiple"], -1.0, places=3)

    def test_the_stop_wins_a_bar_that_spans_both_levels(self):
        """The true intrabar order is unknown; assuming the stop hit first is
        the non-optimistic choice, and it has to match what backtest.py does."""
        trade = self.run_watch(bars([(1.10350, 1.09890, 1.10000)]))
        self.assertEqual(trade["outcome"], "SL")


class TestTrailingStop(WatchHarness):
    def test_tp1_moves_the_stop_to_breakeven(self):
        trade = self.run_watch(bars([(1.10110, 1.10000, 1.10100)]))
        self.assertTrue(trade["alerts_sent"]["tp1"])
        self.assertEqual(trade["current_sl"], trade["entry_price"])
        self.assertFalse(trade["closed"])
        self.assertIn("breakeven", self.alerts[0])

    def test_tp2_moves_the_stop_to_tp1(self):
        trade = self.run_watch(bars([(1.10110, 1.10000, 1.10100),
                                     (1.10210, 1.10100, 1.10200)]))
        self.assertTrue(trade["alerts_sent"]["tp2"])
        self.assertEqual(trade["current_sl"], trade["take_profit_1"])

    def test_a_trailed_stop_closes_in_profit_and_is_still_called_sl(self):
        """The outcome label is where the exit came from, not whether it won -
        the trade log's first three records are 'SL' at a positive R."""
        trade = self.run_watch(bars([(1.10110, 1.10000, 1.10100),
                                     (1.10210, 1.10100, 1.10200),
                                     (1.10200, 1.10090, 1.10095)]))
        self.assertEqual(trade["outcome"], "SL")
        self.assertGreater(trade["r_multiple"], 0)

    def test_tp3_closes_the_trade_at_three_r(self):
        trade = self.run_watch(bars([(1.10110, 1.10000, 1.10100),
                                     (1.10210, 1.10100, 1.10200),
                                     (1.10310, 1.10200, 1.10300)]))
        self.assertEqual(trade["outcome"], "TP3")
        self.assertAlmostEqual(trade["r_multiple"], 3.0, places=2)

    def test_targets_are_taken_in_order_within_one_bar(self):
        trade = self.run_watch(bars([(1.10310, 1.10000, 1.10300)]))
        self.assertEqual(trade["outcome"], "TP3")
        self.assertTrue(all(trade["alerts_sent"][k] for k in ("tp1", "tp2", "tp3")))


class TestTimeout(WatchHarness):
    def test_a_trade_past_the_hold_window_is_closed_at_market(self):
        old = make_trade(opened_at=(datetime.now(timezone.utc)
                                    - timedelta(hours=a5.MAX_HOLD_HOURS + 1)).isoformat())
        trade = self.run_watch(bars([(1.10050, 1.09950, 1.10020)],
                                    start=datetime.now(timezone.utc) - timedelta(minutes=5)),
                               old)
        self.assertEqual(trade["outcome"], "TIMEOUT")
        self.assertAlmostEqual(trade["exit_price"], 1.10020, places=5)

    def test_a_young_trade_is_not_timed_out(self):
        trade = self.run_watch(bars([(1.10050, 1.09950, 1.10020)]))
        self.assertFalse(trade["closed"])


class TestTradeLog(WatchHarness):
    def test_a_closed_trade_is_appended_with_score_and_components(self):
        """Without these two fields the forward samples can only ever answer
        'did it make money', never 'which indicator was carrying it'."""
        self.run_watch(bars([(1.10010, 1.09890, 1.09895)]))
        record = self.logged()[0]
        self.assertEqual(record["outcome"], "SL")
        self.assertEqual(record["score"], 1.83)
        self.assertEqual(record["components"], {"ema_sep": 0.5})

    def test_an_open_trade_writes_nothing(self):
        self.run_watch(bars([(1.10050, 1.09950, 1.10020)]))
        self.assertEqual(self.logged(), [])

    def test_a_gap_in_coverage_is_recorded_on_the_trade(self):
        """A weekend used to leave the missed bars simply unaccounted for; the
        trade then timed out at whatever price was current and logged an
        R-multiple that had nothing to do with the levels it traded through."""
        late = bars([(1.10050, 1.09950, 1.10020)],
                    start=OPENED + timedelta(days=3))
        trade = self.run_watch(late)
        self.assertGreater(trade["coverage_gap_hours"], 0)

    def test_no_gap_is_recorded_when_the_feed_reaches_back(self):
        """The live fetch pulls days of history, so it starts well before the
        last check; only the leftover minutes are tolerated as normal."""
        early = bars([(1.10050, 1.09950, 1.10020)],
                     start=OPENED - timedelta(days=1))
        self.assertIsNone(self.run_watch(early).get("coverage_gap_hours"))

    def test_a_few_minutes_after_the_last_check_is_not_a_gap(self):
        trade = self.run_watch(bars([(1.10050, 1.09950, 1.10020)]))
        self.assertIsNone(trade.get("coverage_gap_hours"))


class TestFetchWindow(unittest.TestCase):
    """A fixed 2-day window did not survive the weekend the watcher skips."""

    def setUp(self):
        self.now = pd.Timestamp("2026-08-31 12:00", tz="UTC")

    def window(self, hours):
        return a5.fetch_window_days(self.now - pd.Timedelta(hours=hours), self.now)

    def test_a_routine_gap_asks_for_the_minimum(self):
        self.assertEqual(self.window(0.5), a5.MIN_FETCH_DAYS)

    def test_a_weekend_gap_asks_for_more(self):
        self.assertGreaterEqual(self.window(72), 4)

    def test_the_window_is_capped(self):
        self.assertEqual(self.window(24 * 90), a5.MAX_FETCH_DAYS)

    def test_a_trade_with_no_last_check_falls_back_to_the_minimum(self):
        self.assertEqual(a5.fetch_window_days(None, self.now), a5.MIN_FETCH_DAYS)


class TestStateFile(unittest.TestCase):
    def write(self, payload):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    def test_a_list_of_trades_is_read(self):
        path = self.write([make_trade(), make_trade(pair="GBPUSD")])
        self.assertEqual(len(a5.load_open_trades(path)), 2)

    def test_the_legacy_single_dict_is_read(self):
        self.assertEqual(len(a5.load_open_trades(self.write(make_trade()))), 1)

    def test_a_missing_file_is_none_not_a_crash(self):
        self.assertIsNone(a5.load_open_trades("no-such-file.json"))

    def test_junk_entries_are_dropped(self):
        path = self.write([make_trade(), "not a trade", None])
        self.assertEqual(len(a5.load_open_trades(path)), 1)


if __name__ == "__main__":
    unittest.main()
