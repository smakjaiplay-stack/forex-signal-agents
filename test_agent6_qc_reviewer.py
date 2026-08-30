"""
Unit tests for Agent 6 - Signal QC Reviewer.

Stdlib unittest only, so CI needs no extra dependency:

    python -m unittest test_agent6_qc_reviewer -v

Every test pins `now` explicitly and builds its own fixtures, so the suite
never depends on the checked-in signal.json / news_summary.json (which go
stale by design and would otherwise fail rule 5 forever).
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import agent6_qc_reviewer as qc

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


def iso(dt):
    return dt.isoformat()


def make_signal(**overrides):
    """A clean signal that passes every rule, so each test can break exactly one."""
    signal = {
        "pair": "EURUSD",
        "action": "Buy",
        "direction": "bullish",
        "confidence": "medium",
        "possibility_percent": 70,
        "status": "Active",
        "open_price": 1.10000,
        "take_profit_1": 1.10300,
        "take_profit_2": 1.10600,
        "take_profit_3": 1.10900,
        "stop_loss": 1.09900,   # risk 100 pips, TP1 reward 300 -> R:R 3.0
        "support": 1.09900,
        "resistance": 1.10900,
        "reasons": ["EMA20 above EMA50 (short-term uptrend)"],
        "new": True,
    }
    signal.update(overrides)
    return signal


def make_news(events=None, generated_at=None):
    return {
        "generated_at": generated_at or iso(NOW - timedelta(minutes=2)),
        "calendar_events_today": events or [],
        "news_headlines": [],
    }


def make_tech(generated_at=None):
    return {
        "generated_at": generated_at or iso(NOW - timedelta(minutes=2)),
        "interval": "15m",
        "results": [],
    }


def make_signal_data(signals, generated_at=None):
    return {
        "generated_at": generated_at or iso(NOW - timedelta(minutes=1)),
        "signals": signals,
        "signal": signals[0] if signals else None,
    }


def review_one(signal, *, news=None, tech=None, open_trades=None, **kwargs):
    """Review a single signal and hand back the reviewed card."""
    result = qc.review(
        make_signal_data([signal]),
        news if news is not None else make_news(),
        tech if tech is not None else make_tech(),
        open_trades or [],
        now=NOW,
        **kwargs,
    )
    return result["signals"][0]


class TestRule1NewsConflict(unittest.TestCase):
    """High-impact release imminent for one of the pair's currencies."""

    def high_impact(self, currency, minutes_ahead, actual=None):
        return {
            "time": iso(NOW + timedelta(minutes=minutes_ahead)),
            "currency": currency,
            "title": "Core CPI m/m",
            "impact": "High",
            "actual": actual,
        }

    def test_imminent_event_flags_high_risk_and_cuts_possibility(self):
        news = make_news([self.high_impact("USD", 45)])
        reviewed = review_one(make_signal(possibility_percent=90), news=news)

        self.assertIn(qc.FLAG_HIGH_RISK, reviewed["qc_flags"])
        self.assertEqual(reviewed["qc_status"], qc.STATUS_DOWNGRADED)
        self.assertEqual(reviewed["possibility_percent"], 90 - qc.NEWS_RISK_PENALTY)
        self.assertEqual(reviewed["original_possibility_percent"], 90)
        self.assertIn("Core CPI", reviewed["qc_notes"])

    def test_matches_the_base_currency_too(self):
        news = make_news([self.high_impact("EUR", 30)])
        reviewed = review_one(make_signal(pair="EURUSD", possibility_percent=90), news=news)
        self.assertIn(qc.FLAG_HIGH_RISK, reviewed["qc_flags"])

    def test_event_beyond_the_window_is_ignored(self):
        news = make_news([self.high_impact("USD", 180)])   # 3h out, window is 2h
        reviewed = review_one(make_signal(), news=news)

        self.assertEqual(reviewed["qc_flags"], [])
        self.assertEqual(reviewed["qc_status"], qc.STATUS_APPROVED)

    def test_already_released_event_is_ignored(self):
        news = make_news([self.high_impact("USD", 30, actual="0.3%")])
        reviewed = review_one(make_signal(), news=news)
        self.assertEqual(reviewed["qc_flags"], [])

    def test_past_event_is_ignored(self):
        news = make_news([self.high_impact("USD", -30)])
        reviewed = review_one(make_signal(), news=news)
        self.assertEqual(reviewed["qc_flags"], [])

    def test_medium_impact_event_is_ignored(self):
        event = self.high_impact("USD", 30)
        event["impact"] = "Medium"
        reviewed = review_one(make_signal(), news=make_news([event]))
        self.assertEqual(reviewed["qc_flags"], [])

    def test_unrelated_currency_is_ignored(self):
        news = make_news([self.high_impact("AUD", 30)])
        reviewed = review_one(make_signal(pair="EURUSD"), news=news)
        self.assertEqual(reviewed["qc_flags"], [])

    def test_wait_signal_is_not_flagged_high_risk(self):
        """Nothing is being risked on a Wait, so pending news is irrelevant."""
        news = make_news([self.high_impact("USD", 30)])
        reviewed = review_one(make_signal(action="Wait", direction="neutral"), news=news)
        self.assertNotIn(qc.FLAG_HIGH_RISK, reviewed["qc_flags"])

    def test_news_penalty_can_push_a_signal_under_the_confidence_floor(self):
        """Rule 1 feeds rule 2: 70% minus the 15-point penalty is below 60%."""
        news = make_news([self.high_impact("USD", 20)])
        reviewed = review_one(make_signal(possibility_percent=70), news=news)

        self.assertEqual(reviewed["possibility_percent"], 55)
        self.assertIn(qc.FLAG_HIGH_RISK, reviewed["qc_flags"])
        self.assertIn(qc.FLAG_LOW_CONFIDENCE, reviewed["qc_flags"])
        self.assertEqual(reviewed["final_action"], "Wait")


class TestRule2ConfidenceThreshold(unittest.TestCase):
    """The FALLBACK path: no calibration attached to the signal.

    These signals carry no "calibration" block, which is what a run looks like
    before validate.py has ever produced score_calibration.json. In that state
    possibility_percent is Agent 3's uncalibrated linear guess, and the flat
    DEFAULT_MIN_POSSIBILITY floor is the only thing available to gate on. Once
    a calibration exists, TestRule2UnprovenEdge and TestRule2BreakevenFloor
    describe what happens instead.
    """

    def test_below_threshold_is_forced_to_wait(self):
        reviewed = review_one(make_signal(possibility_percent=55))

        self.assertIn(qc.FLAG_LOW_CONFIDENCE, reviewed["qc_flags"])
        self.assertEqual(reviewed["original_action"], "Buy")
        self.assertEqual(reviewed["final_action"], "Wait")
        self.assertEqual(reviewed["action"], "Wait")
        self.assertEqual(reviewed["status"], "No Trade")
        self.assertEqual(reviewed["qc_status"], qc.STATUS_DOWNGRADED)

    def test_exactly_at_threshold_passes(self):
        reviewed = review_one(make_signal(possibility_percent=60))
        self.assertEqual(reviewed["qc_flags"], [])
        self.assertEqual(reviewed["final_action"], "Buy")

    def test_threshold_is_configurable(self):
        reviewed = review_one(make_signal(possibility_percent=70), min_possibility=75)
        self.assertIn(qc.FLAG_LOW_CONFIDENCE, reviewed["qc_flags"])
        self.assertEqual(reviewed["final_action"], "Wait")

    def test_wait_signal_is_left_alone(self):
        reviewed = review_one(make_signal(action="Wait", direction="neutral",
                                          possibility_percent=40))
        self.assertEqual(reviewed["qc_flags"], [])
        self.assertEqual(reviewed["qc_status"], qc.STATUS_APPROVED)


def calibration(**overrides):
    """The block Agent 3 attaches from score_calibration.json.

    Defaults describe a strategy with a proven edge, so each test below breaks
    exactly one thing - the mirror of make_signal's contract.
    """
    calib = {
        "breakeven_win_rate": 46.7,
        "payoff_ratio": 1.14,
        "expectancy_r": 0.05,
        "expectancy_t": 2.4,
        "edge_significant": True,
        "trades": 1312,
    }
    calib.update(overrides)
    return calib


class TestRule2UnprovenEdge(unittest.TestCase):
    """No demonstrated edge means nothing ships, whatever the possibility says.

    The old rule compared possibility_percent against a flat 60. That number
    was picked while possibility_percent was the constant 70 - a bar set just
    under a decoration. These tests pin the two things that replaced it.
    """

    def test_unproven_edge_forces_wait_even_at_high_possibility(self):
        reviewed = review_one(make_signal(
            possibility_percent=95,
            calibration=calibration(edge_significant=False, expectancy_r=-0.036,
                                    expectancy_t=-1.44)))

        self.assertIn(qc.FLAG_UNPROVEN_EDGE, reviewed["qc_flags"])
        self.assertEqual(reviewed["final_action"], "Wait")
        self.assertEqual(reviewed["status"], "No Trade")

    def test_allow_unproven_edge_publishes_but_still_flags(self):
        """The override collects live samples; it must not launder the signal."""
        reviewed = review_one(
            make_signal(possibility_percent=52,
                        calibration=calibration(edge_significant=False)),
            allow_unproven_edge=True)

        self.assertIn(qc.FLAG_UNPROVEN_EDGE, reviewed["qc_flags"])
        self.assertEqual(reviewed["final_action"], "Buy")
        self.assertEqual(reviewed["qc_status"], qc.STATUS_DOWNGRADED)

    def test_gate_rearms_itself_when_a_later_calibration_proves_an_edge(self):
        """Nobody should have to remember to switch the gate back on."""
        reviewed = review_one(make_signal(possibility_percent=52,
                                          calibration=calibration()))
        self.assertNotIn(qc.FLAG_UNPROVEN_EDGE, reviewed["qc_flags"])
        self.assertEqual(reviewed["final_action"], "Buy")

    def test_missing_edge_flag_counts_as_unproven(self):
        """A calibration file written before the field existed has not
        demonstrated an edge, so it must not be read as having one."""
        calib = calibration()
        del calib["edge_significant"]
        reviewed = review_one(make_signal(possibility_percent=95, calibration=calib))
        self.assertIn(qc.FLAG_UNPROVEN_EDGE, reviewed["qc_flags"])
        self.assertEqual(reviewed["final_action"], "Wait")


class TestRule2BreakevenFloor(unittest.TestCase):
    """With an edge proven, the floor is the breakeven win rate - not 60."""

    def test_below_breakeven_is_forced_to_wait(self):
        reviewed = review_one(make_signal(possibility_percent=44,
                                          calibration=calibration()))
        self.assertIn(qc.FLAG_LOW_CONFIDENCE, reviewed["qc_flags"])
        self.assertEqual(reviewed["final_action"], "Wait")

    def test_above_breakeven_but_below_the_old_60_now_passes(self):
        """52% loses money at a 1:1 payoff and makes money at 1.14:1.

        This is the case the flat 60% bar got wrong, and the reason the floor
        had to become a derived number rather than a chosen one.
        """
        reviewed = review_one(make_signal(possibility_percent=52,
                                          calibration=calibration()))
        self.assertEqual(reviewed["qc_flags"], [])
        self.assertEqual(reviewed["final_action"], "Buy")

    def test_a_worse_payoff_raises_the_floor_on_the_same_signal(self):
        """Same 52% possibility, worse payoff -> blocked. The win rate alone
        never carried enough information to make this call."""
        reviewed = review_one(make_signal(
            possibility_percent=52,
            calibration=calibration(payoff_ratio=0.8, breakeven_win_rate=55.6)))
        self.assertIn(qc.FLAG_LOW_CONFIDENCE, reviewed["qc_flags"])
        self.assertEqual(reviewed["final_action"], "Wait")

    def test_missing_breakeven_falls_back_to_the_hand_set_floor(self):
        reviewed = review_one(make_signal(
            possibility_percent=55,
            calibration=calibration(breakeven_win_rate=None)))
        self.assertIn(qc.FLAG_LOW_CONFIDENCE, reviewed["qc_flags"])
        self.assertEqual(reviewed["final_action"], "Wait")


class TestRule3RiskReward(unittest.TestCase):
    """Reward/risk is measured entry -> TP1 against entry -> SL."""

    def test_poor_rr_is_flagged(self):
        # risk 100 pips, TP1 reward 100 pips -> 1.0, below the 1.5 floor
        reviewed = review_one(make_signal(take_profit_1=1.10100, stop_loss=1.09900))

        self.assertIn(qc.FLAG_POOR_RR, reviewed["qc_flags"])
        self.assertEqual(reviewed["qc_risk_reward_tp1"], 1.0)
        self.assertEqual(reviewed["qc_status"], qc.STATUS_DOWNGRADED)
        # A poor ratio is a warning, not a veto — the action stands.
        self.assertEqual(reviewed["final_action"], "Buy")

    def test_good_rr_passes(self):
        reviewed = review_one(make_signal())
        self.assertEqual(reviewed["qc_flags"], [])
        self.assertEqual(reviewed["qc_risk_reward_tp1"], 3.0)

    def test_sell_side_rr_is_measured_the_same_way(self):
        reviewed = review_one(make_signal(
            action="Sell", direction="bearish",
            take_profit_1=1.09800, stop_loss=1.10100,   # reward 200, risk 100
        ))
        self.assertEqual(reviewed["qc_risk_reward_tp1"], 2.0)
        self.assertNotIn(qc.FLAG_POOR_RR, reviewed["qc_flags"])

    def test_zero_risk_is_flagged_rather_than_dividing_by_zero(self):
        reviewed = review_one(make_signal(stop_loss=1.10000))   # SL == entry
        self.assertIn(qc.FLAG_POOR_RR, reviewed["qc_flags"])
        self.assertIsNone(reviewed["qc_risk_reward_tp1"])

    def test_missing_levels_are_flagged(self):
        signal = make_signal()
        del signal["take_profit_1"]
        reviewed = review_one(signal)
        self.assertIn(qc.FLAG_POOR_RR, reviewed["qc_flags"])

    def test_min_rr_zero_disables_the_check(self):
        reviewed = review_one(make_signal(take_profit_1=1.10050), min_rr=0)
        self.assertNotIn(qc.FLAG_POOR_RR, reviewed["qc_flags"])


class TestRule4ConflictingSignal(unittest.TestCase):
    """A new signal must not point the other way to a live trade."""

    def open_trade(self, pair="EURUSD", action="Sell", closed=False):
        return {
            "pair": pair,
            "action": action,
            "entry_price": 1.10500,
            "stop_loss": 1.10800,
            "opened_at": iso(NOW - timedelta(hours=3)),
            "closed": closed,
        }

    def test_opposite_open_trade_is_flagged(self):
        reviewed = review_one(make_signal(action="Buy"),
                              open_trades=[self.open_trade(action="Sell")])

        self.assertIn(qc.FLAG_CONFLICTING, reviewed["qc_flags"])
        self.assertEqual(reviewed["qc_status"], qc.STATUS_DOWNGRADED)
        self.assertIn("EURUSD", reviewed["qc_notes"])

    def test_same_direction_open_trade_is_fine(self):
        reviewed = review_one(make_signal(action="Buy"),
                              open_trades=[self.open_trade(action="Buy")])
        self.assertEqual(reviewed["qc_flags"], [])

    def test_other_pair_is_not_a_conflict(self):
        reviewed = review_one(make_signal(pair="EURUSD", action="Buy"),
                              open_trades=[self.open_trade(pair="GBPUSD", action="Sell")])
        self.assertEqual(reviewed["qc_flags"], [])

    def test_closed_trades_are_ignored(self):
        raw = [self.open_trade(action="Sell", closed=True)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "open_trade.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(raw, f)
            self.assertEqual(qc.load_open_trades(path), [])

    def test_load_open_trades_accepts_the_legacy_single_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "open_trade.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.open_trade(), f)
            self.assertEqual(len(qc.load_open_trades(path)), 1)

    def test_load_open_trades_tolerates_a_missing_file(self):
        self.assertEqual(qc.load_open_trades("definitely_not_here.json"), [])


class TestRule5DataFreshness(unittest.TestCase):
    """Stale inputs block the run outright."""

    def test_stale_technical_data_blocks(self):
        tech = make_tech(generated_at=iso(NOW - timedelta(minutes=45)))
        reviewed = review_one(make_signal(), tech=tech)

        self.assertIn(qc.FLAG_STALE_DATA, reviewed["qc_flags"])
        self.assertEqual(reviewed["qc_status"], qc.STATUS_BLOCKED)
        self.assertFalse(reviewed["new"])
        self.assertIn("blocked by QC", reviewed["skip_reason"])

    def test_stale_news_data_blocks(self):
        news = make_news(generated_at=iso(NOW - timedelta(hours=6)))
        reviewed = review_one(make_signal(), news=news)
        self.assertEqual(reviewed["qc_status"], qc.STATUS_BLOCKED)

    def test_fresh_data_passes(self):
        reviewed = review_one(make_signal())
        self.assertEqual(reviewed["qc_status"], qc.STATUS_APPROVED)

    def test_missing_timestamp_counts_as_stale(self):
        """Can't verify freshness == don't trust it."""
        reviewed = review_one(make_signal(), tech={"interval": "15m", "results": []})
        self.assertIn(qc.FLAG_STALE_DATA, reviewed["qc_flags"])
        self.assertEqual(reviewed["qc_status"], qc.STATUS_BLOCKED)

    def test_missing_file_counts_as_stale(self):
        reviewed = review_one(make_signal(), news={})
        self.assertEqual(reviewed["qc_status"], qc.STATUS_BLOCKED)

    def test_age_limit_is_configurable(self):
        tech = make_tech(generated_at=iso(NOW - timedelta(minutes=45)))
        reviewed = review_one(make_signal(), tech=tech, max_data_age_minutes=60)
        self.assertEqual(reviewed["qc_status"], qc.STATUS_APPROVED)

    def test_blocked_wins_over_every_other_flag(self):
        """A stale run is blocked even if it also trips rules 2 and 3."""
        tech = make_tech(generated_at=iso(NOW - timedelta(minutes=45)))
        reviewed = review_one(
            make_signal(possibility_percent=40, take_profit_1=1.10050), tech=tech)

        self.assertEqual(reviewed["qc_status"], qc.STATUS_BLOCKED)
        self.assertIn(qc.FLAG_LOW_CONFIDENCE, reviewed["qc_flags"])
        self.assertIn(qc.FLAG_POOR_RR, reviewed["qc_flags"])
        self.assertIn(qc.FLAG_STALE_DATA, reviewed["qc_flags"])


class TestReviewPayload(unittest.TestCase):
    """Shape of signal_reviewed.json as a whole."""

    def test_clean_run_is_approved_and_unmutated(self):
        original = make_signal()
        snapshot = json.loads(json.dumps(original))
        result = qc.review(make_signal_data([original]), make_news(), make_tech(), [], now=NOW)

        self.assertEqual(result["qc_status"], qc.STATUS_APPROVED)
        self.assertEqual(result["qc"]["counts"], {"approved": 1, "downgraded": 0, "blocked": 0})
        self.assertEqual(original, snapshot, "review() must not mutate its input")

    def test_run_status_is_the_worst_signal_status(self):
        signals = [make_signal(pair="EURUSD"),
                   make_signal(pair="GBPUSD", possibility_percent=50)]
        result = qc.review(make_signal_data(signals), make_news(), make_tech(), [], now=NOW)

        self.assertEqual(result["qc_status"], qc.STATUS_DOWNGRADED)
        self.assertEqual(result["qc"]["counts"]["approved"], 1)
        self.assertEqual(result["qc"]["counts"]["downgraded"], 1)

    def test_run_is_blocked_only_when_every_signal_is(self):
        tech = make_tech(generated_at=iso(NOW - timedelta(hours=2)))
        signals = [make_signal(pair="EURUSD"), make_signal(pair="GBPUSD")]
        result = qc.review(make_signal_data(signals), make_news(), tech, [], now=NOW)
        self.assertEqual(result["qc_status"], qc.STATUS_BLOCKED)

    def test_legacy_single_signal_input_is_reviewed(self):
        payload = {"generated_at": iso(NOW), "signal": make_signal(possibility_percent=50)}
        result = qc.review(payload, make_news(), make_tech(), [], now=NOW)

        self.assertEqual(len(result["signals"]), 1)
        self.assertEqual(result["signal"]["final_action"], "Wait")

    def test_empty_signal_list_is_approved(self):
        result = qc.review(make_signal_data([]), make_news(), make_tech(), [], now=NOW)
        self.assertEqual(result["signals"], [])
        self.assertIsNone(result["signal"])
        self.assertEqual(result["qc_status"], qc.STATUS_APPROVED)

    def test_unrelated_top_level_keys_survive(self):
        payload = make_signal_data([make_signal()])
        payload["rejected_low_rr"] = [{"pair": "USDJPY"}]
        result = qc.review(payload, make_news(), make_tech(), [], now=NOW)
        self.assertEqual(result["rejected_low_rr"], [{"pair": "USDJPY"}])


class TestQcLog(unittest.TestCase):
    def test_only_flagged_signals_are_logged_by_default(self):
        signals = [make_signal(pair="EURUSD"),
                   make_signal(pair="GBPUSD", possibility_percent=50)]
        result = qc.review(make_signal_data(signals), make_news(), make_tech(), [], now=NOW)

        records = qc.log_records(result)
        self.assertEqual([r["pair"] for r in records], ["GBPUSD"])
        self.assertEqual(records[0]["original_action"], "Buy")
        self.assertEqual(records[0]["final_action"], "Wait")
        self.assertIn(qc.FLAG_LOW_CONFIDENCE, records[0]["qc_flags"])

    def test_log_all_includes_approved_signals(self):
        result = qc.review(make_signal_data([make_signal()]), make_news(), make_tech(), [], now=NOW)
        self.assertEqual(len(qc.log_records(result, log_all=True)), 1)

    def test_append_qc_log_appends_one_json_line_per_record(self):
        result = qc.review(make_signal_data([make_signal(possibility_percent=50)]),
                           make_news(), make_tech(), [], now=NOW)
        records = qc.log_records(result)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "qc_log.jsonl")
            self.assertEqual(qc.append_qc_log(path, records), 1)
            self.assertEqual(qc.append_qc_log(path, records), 1)   # appends, doesn't truncate

            with open(path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["pair"], "EURUSD")
        self.assertEqual(lines[0]["qc_status"], qc.STATUS_DOWNGRADED)

    def test_append_qc_log_creates_nothing_when_there_is_nothing_to_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "qc_log.jsonl")
            self.assertEqual(qc.append_qc_log(path, []), 0)
            self.assertFalse(os.path.exists(path))


class TestOpenTradeRollback(unittest.TestCase):
    """Trades Agent 3 opened for a signal QC then rejected must not survive."""

    def trade(self, pair, action, opened_at):
        return {"pair": pair, "action": action, "entry_price": 1.1, "opened_at": opened_at,
                "closed": False}

    def test_blocked_signal_rolls_back_this_runs_trade(self):
        generated_at = iso(NOW - timedelta(minutes=1))
        tech = make_tech(generated_at=iso(NOW - timedelta(hours=2)))
        payload = make_signal_data([make_signal(pair="EURUSD")], generated_at=generated_at)
        result = qc.review(payload, make_news(), tech, [], now=NOW)

        open_trades = [self.trade("EURUSD", "Buy", generated_at)]
        kept, dropped = qc.rollback_open_trades(open_trades, result)

        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 1)

    def test_downgrade_to_wait_also_rolls_back(self):
        generated_at = iso(NOW - timedelta(minutes=1))
        payload = make_signal_data([make_signal(pair="EURUSD", possibility_percent=50)],
                                   generated_at=generated_at)
        result = qc.review(payload, make_news(), make_tech(), [], now=NOW)

        kept, dropped = qc.rollback_open_trades([self.trade("EURUSD", "Buy", generated_at)], result)
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 1)

    def test_older_trades_on_the_same_pair_are_left_alone(self):
        """Only trades stamped with this run's generated_at are rolled back —
        a position Agent 5 is already managing must survive."""
        generated_at = iso(NOW - timedelta(minutes=1))
        older = iso(NOW - timedelta(hours=8))
        payload = make_signal_data([make_signal(pair="EURUSD", possibility_percent=50)],
                                   generated_at=generated_at)
        result = qc.review(payload, make_news(), make_tech(), [], now=NOW)

        kept, dropped = qc.rollback_open_trades([self.trade("EURUSD", "Buy", older)], result)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])

    def test_approved_signal_keeps_its_trade(self):
        generated_at = iso(NOW - timedelta(minutes=1))
        payload = make_signal_data([make_signal(pair="EURUSD")], generated_at=generated_at)
        result = qc.review(payload, make_news(), make_tech(), [], now=NOW)

        open_trades = [self.trade("EURUSD", "Buy", generated_at)]
        kept, dropped = qc.rollback_open_trades(open_trades, result)
        self.assertEqual(kept, open_trades)
        self.assertEqual(dropped, [])

    def test_a_downgraded_but_still_tradable_signal_keeps_its_trade(self):
        """poor_rr warns; it doesn't cancel the trade."""
        generated_at = iso(NOW - timedelta(minutes=1))
        payload = make_signal_data(
            [make_signal(pair="EURUSD", take_profit_1=1.10050)], generated_at=generated_at)
        result = qc.review(payload, make_news(), make_tech(), [], now=NOW)

        self.assertEqual(result["signals"][0]["qc_status"], qc.STATUS_DOWNGRADED)
        open_trades = [self.trade("EURUSD", "Buy", generated_at)]
        kept, _ = qc.rollback_open_trades(open_trades, result)
        self.assertEqual(kept, open_trades)


class TestHelpers(unittest.TestCase):
    def test_parse_time_handles_offsets_z_and_naive(self):
        self.assertEqual(qc.parse_time("2026-08-13T02:00:00-04:00").utcoffset(),
                         timedelta(hours=-4))
        self.assertEqual(qc.parse_time("2026-08-13T06:00:00Z").tzinfo, timezone.utc)
        self.assertEqual(qc.parse_time("2026-08-13T06:00:00").tzinfo, timezone.utc)

    def test_parse_time_returns_none_for_junk(self):
        for junk in (None, "", "not a date", 12345.0):
            self.assertIsNone(qc.parse_time(junk))

    def test_currencies_in_pair(self):
        self.assertEqual(qc.currencies_in_pair("EURUSD"), ("EUR", "USD"))
        self.assertEqual(qc.currencies_in_pair("xauusd"), ("XAU", "USD"))

    def test_malformed_calendar_entries_do_not_crash(self):
        news = {
            "generated_at": iso(NOW),
            "calendar_events_today": [
                None, "nonsense", {},
                {"impact": "High", "currency": "USD", "time": "garbage"},
                {"impact": "High", "time": iso(NOW + timedelta(minutes=10))},   # no currency
            ],
        }
        self.assertEqual(qc.imminent_high_impact_events(news, NOW, 120), {})


if __name__ == "__main__":
    unittest.main()
