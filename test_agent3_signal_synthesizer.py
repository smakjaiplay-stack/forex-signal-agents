"""
Unit tests for Agent 3 - Signal Synthesizer.

Agent 3 had no tests, and it is the file that decides what a trade actually
looks like. That is where the project's largest measured defect lived: for its
entire recorded history the levels on 98.6% of trades came out of a fallback
branch nobody knew was firing, and no test - and no QC rule - was positioned to
notice. Most of what follows pins the geometry.

    python -m unittest test_agent3_signal_synthesizer -v
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import agent3_signal_synthesizer as a3
import scoring


def iso(dt):
    return dt.isoformat()


NOW = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


def make_result(pair="EURUSD", score=1.8, bias="bullish", close=1.10000,
                atr=0.00080, **overrides):
    result = {
        "pair": pair,
        "bias": bias,
        "score": score,
        "last_close": close,
        "support": close - 0.0050,
        "resistance": close + 0.0050,
        "atr14": atr,
        "reasons": ["EMA20 above EMA50 (short-term uptrend)"],
        "last_bar_time": "2026-08-13 11:00 UTC",
        "data_age_minutes": 12.0,
        "components": {"ema_sep": 0.5},
    }
    result.update(overrides)
    return result


def make_tech(results, interval="1h"):
    return {"generated_at": iso(NOW), "interval": interval, "results": results}


class TestTradeGeometry(unittest.TestCase):
    """The stop is ATR-scaled and the targets are multiples of the risk."""

    def test_stop_sits_at_the_configured_atr_multiple(self):
        levels = a3.build_trade_levels("Buy", 1.10000, 5, atr=0.00100)
        self.assertAlmostEqual(1.10000 - levels["stop_loss"],
                               0.00100 * scoring.SL_ATR_MULT, places=6)

    def test_targets_are_one_two_and_three_times_the_risk(self):
        entry, atr = 1.10000, 0.00100
        levels = a3.build_trade_levels("Buy", entry, 5, atr=atr)
        risk = entry - levels["stop_loss"]
        for mult, key in zip(scoring.TP_R_MULTS,
                             ("take_profit_1", "take_profit_2", "take_profit_3")):
            self.assertAlmostEqual(levels[key] - entry, risk * mult, places=6)

    def test_sell_mirrors_buy(self):
        buy = a3.build_trade_levels("Buy", 1.10000, 5, atr=0.00100)
        sell = a3.build_trade_levels("Sell", 1.10000, 5, atr=0.00100)
        self.assertAlmostEqual(buy["take_profit_3"] - 1.10000,
                               1.10000 - sell["take_profit_3"], places=6)
        self.assertAlmostEqual(buy["stop_loss"] - 1.10000,
                               1.10000 - sell["stop_loss"], places=6)

    def test_reward_risk_is_a_stated_constant_not_an_accident(self):
        """It used to be whatever the range happened to be, which is how a
        card came to say 28.4."""
        levels = a3.build_trade_levels("Buy", 1.10000, 5, atr=0.00080)
        rr1 = a3.risk_reward(1.10000, levels["take_profit_1"], levels["stop_loss"])
        rr3 = a3.risk_reward(1.10000, levels["take_profit_3"], levels["stop_loss"])
        self.assertAlmostEqual(rr1, scoring.TP_R_MULTS[0], places=2)
        self.assertAlmostEqual(rr3, scoring.TP_R_MULTS[2], places=2)

    def test_rounding_never_pushes_the_ratio_past_qc_tolerance(self):
        """Levels are placed at exact R-multiples and THEN rounded, so the
        rounding has to stay small against the risk. Silver caught this: at
        ~28 with a 0.11 ATR, 2dp moved the ratio to 1.06 and Agent 6 would
        have read every XAGUSD card as broken geometry."""
        import agent6_qc_reviewer as qc
        cases = (("EURUSD", 1.10000, 0.00080), ("GBPUSD", 1.27000, 0.00090),
                 ("USDJPY", 157.250, 0.14000), ("AUDUSD", 0.65400, 0.00050),
                 ("USDCAD", 1.37000, 0.00070), ("NZDUSD", 0.59800, 0.00045),
                 ("USDCHF", 0.88000, 0.00060), ("XAUUSD", 2431.00, 3.33000),
                 ("XAGUSD", 28.4400, 0.11000))
        self.assertEqual({c[0] for c in cases},
                         set(__import__("agent2_technical_analyzer").DEFAULT_PAIRS),
                         "every supported pair needs a case here")
        for pair, entry, atr in cases:
            levels = a3.build_trade_levels("Buy", entry, a3.decimals_for_pair(pair), atr=atr)
            rr1 = a3.risk_reward(entry, levels["take_profit_1"], levels["stop_loss"])
            self.assertAlmostEqual(rr1, 1.0, delta=qc.RR_TOLERANCE, msg=f"{pair} rr1={rr1}")

    def test_no_atr_means_no_levels(self):
        """The removed fallback invented a target at a flat 1.5% of price -
        on FX roughly ten ATRs away - and it produced 98.6% of every trade the
        project has measured. There is no honest stop without a volatility
        estimate, so there is no trade."""
        for bad_atr in (None, 0, -1, float("nan")):
            self.assertIsNone(a3.build_trade_levels("Buy", 1.10000, 5, atr=bad_atr),
                              f"atr={bad_atr!r} must not produce levels")

    def test_wait_has_no_levels(self):
        self.assertIsNone(a3.build_trade_levels("Wait", 1.10000, 5, atr=0.001))

    def test_zero_atr_mult_would_collapse_the_stop_and_is_refused(self):
        self.assertIsNone(a3.build_trade_levels("Buy", 1.10000, 5, atr=0.001, atr_mult=0))

    def test_stop_can_never_be_closer_than_the_floor(self):
        levels = a3.build_trade_levels("Buy", 1.10000, 5, atr=0.00100, atr_mult=0.01)
        self.assertAlmostEqual(levels["risk_atr_mult"], scoring.MIN_SL_ATR_MULT, places=3)

    def test_risk_atr_mult_is_reported_for_qc(self):
        """Agent 6 verifies the stop is ATR-scaled from this field rather than
        inferring it from the ratio, which is what makes the check direct."""
        levels = a3.build_trade_levels("Buy", 1.10000, 5, atr=0.00080)
        self.assertAlmostEqual(levels["risk_atr_mult"], scoring.SL_ATR_MULT, places=3)


class TestPublicationGate(unittest.TestCase):
    def test_a_pair_without_atr_is_rejected_not_published(self):
        result = a3.synthesize(None, make_tech([make_result(atr=None)]))
        self.assertEqual(result["signals"], [])
        self.assertEqual(len(result["rejected"]), 1)
        self.assertIn("no usable ATR", result["rejected"][0]["reason"])

    def test_scores_below_the_threshold_do_not_publish_a_trade(self):
        weak = make_result(score=0.4, bias="neutral")
        result = a3.synthesize(None, make_tech([weak]))
        self.assertEqual(result["signals"][0]["action"], "Wait")

    def test_top_n_caps_how_many_are_published(self):
        results = [make_result(pair=p, score=1.9) for p in ("EURUSD", "GBPUSD", "AUDUSD")]
        self.assertEqual(len(a3.synthesize(None, make_tech(results), top_n=2)["signals"]), 2)

    def test_pairs_are_ranked_by_absolute_score(self):
        results = [make_result(pair="EURUSD", score=1.3),
                   make_result(pair="GBPUSD", score=-1.9, bias="bearish"),
                   make_result(pair="AUDUSD", score=1.6)]
        signals = a3.synthesize(None, make_tech(results), top_n=3)["signals"]
        self.assertEqual([s["pair"] for s in signals], ["GBPUSD", "AUDUSD", "EURUSD"])

    def test_errored_pairs_are_skipped(self):
        tech = make_tech([{"pair": "EURUSD", "error": "insufficient data"},
                          make_result(pair="GBPUSD")])
        signals = a3.synthesize(None, tech)["signals"]
        self.assertEqual([s["pair"] for s in signals], ["GBPUSD"])

    def test_no_usable_results_returns_a_note_rather_than_a_signal(self):
        result = a3.synthesize(None, make_tech([]))
        self.assertEqual(result["signals"], [])
        self.assertIn("No valid technical results", result["note"])

    def test_the_card_carries_both_reward_risk_ratios(self):
        signal = a3.synthesize(None, make_tech([make_result()]))["signals"][0]
        self.assertAlmostEqual(signal["risk_reward_tp1"], 1.0, delta=0.05)
        self.assertAlmostEqual(signal["risk_reward"], 3.0, delta=0.05)


class TestNewsRisk(unittest.TestCase):
    def news(self, currency, actual=None):
        return {"calendar_events_today": [
            {"impact": "High", "currency": currency, "title": "CPI", "actual": actual}]}

    def test_pending_high_impact_event_flags_the_pair(self):
        signal = a3.synthesize(self.news("EUR"), make_tech([make_result()]))["signals"][0]
        self.assertTrue(signal["pending_high_impact_news"])
        self.assertEqual(signal["confidence"], "low")

    def test_a_released_event_is_not_a_risk(self):
        signal = a3.synthesize(self.news("EUR", actual="2.4%"),
                               make_tech([make_result()]))["signals"][0]
        self.assertFalse(signal["pending_high_impact_news"])

    def test_a_clean_pair_is_preferred_over_a_risky_one(self):
        results = [make_result(pair="EURUSD", score=1.9),
                   make_result(pair="AUDUSD", score=1.5)]
        signals = a3.synthesize(self.news("EUR"), make_tech(results), top_n=1)["signals"]
        self.assertEqual(signals[0]["pair"], "AUDUSD")


class TestPossibilityPercent(unittest.TestCase):
    """The number used to be the constant 70 on every card ever sent."""

    CALIBRATION = {"points": [(1.2, 30.0), (1.6, 40.0), (2.0, 50.0)]}

    def test_uncalibrated_is_the_linear_guess_and_says_so(self):
        signal = a3.synthesize(None, make_tech([make_result()]))["signals"][0]
        self.assertFalse(signal["possibility_calibrated"])

    def test_calibrated_interpolates_the_measured_win_rate(self):
        self.assertEqual(a3.possibility_percent(1.4, False, self.CALIBRATION), 35)

    def test_calibration_is_clamped_outside_the_measured_range(self):
        self.assertEqual(a3.possibility_percent(0.1, False, self.CALIBRATION), 30)
        self.assertEqual(a3.possibility_percent(9.9, False, self.CALIBRATION), 50)

    def test_pending_news_costs_the_documented_penalty(self):
        clean = a3.possibility_percent(2.0, False, self.CALIBRATION)
        risky = a3.possibility_percent(2.0, True, self.CALIBRATION)
        self.assertEqual(clean - risky, 15)

    def test_the_result_stays_inside_30_to_95(self):
        self.assertEqual(a3.possibility_percent(0.0, True, None), 35)
        self.assertEqual(a3.possibility_percent(99.0, False, None), 95)

    def test_calibration_travels_with_the_signal_for_agent6(self):
        """Agent 6 must not have to re-open and re-interpret the file: a win
        rate cannot say whether a trade makes money without its payoff."""
        calib = dict(self.CALIBRATION, breakeven_win_rate=34.3, payoff_ratio=1.91,
                     expectancy_r=0.0043, expectancy_t=0.13, edge_significant=False,
                     trades=1535)
        signal = a3.synthesize(None, make_tech([make_result()]), calibration=calib)["signals"][0]
        self.assertEqual(signal["calibration"]["breakeven_win_rate"], 34.3)
        self.assertFalse(signal["calibration"]["edge_significant"])


class TestCalibrationLoading(unittest.TestCase):
    def write(self, payload):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    def test_missing_file_is_not_an_error(self):
        self.assertIsNone(a3.load_calibration("does-not-exist.json"))

    def test_a_file_with_no_buckets_is_unusable(self):
        self.assertIsNone(a3.load_calibration(self.write({"trades": 10})))

    def test_missing_edge_significant_counts_as_false(self):
        """A calibration that never measured an edge has not demonstrated one."""
        path = self.write({"buckets": [{"score_mid": 1.5, "win_rate": 40}]})
        self.assertFalse(a3.load_calibration(path)["edge_significant"])

    def test_missing_breakeven_is_none_not_zero(self):
        """0 would mean "breaks even at 0%" and wave everything through."""
        path = self.write({"buckets": [{"score_mid": 1.5, "win_rate": 40}]})
        self.assertIsNone(a3.load_calibration(path)["breakeven_win_rate"])


class TestOpenTradeMerging(unittest.TestCase):
    """The pipeline runs six times a day; open_trade.json is merged, not
    overwritten, or Agent 5 loses the trades it is still watching."""

    def write_log(self, records):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_a_pair_that_just_closed_is_in_cooldown(self):
        path = self.write_log([{"pair": "EURUSD",
                                "closed_at": iso(datetime.now(timezone.utc) - timedelta(hours=1))}])
        self.assertEqual(a3.pairs_in_cooldown(path, 6), {"EURUSD"})

    def test_cooldown_expires(self):
        path = self.write_log([{"pair": "EURUSD",
                                "closed_at": iso(datetime.now(timezone.utc) - timedelta(hours=9))}])
        self.assertEqual(a3.pairs_in_cooldown(path, 6), set())

    def test_zero_hours_disables_the_cooldown(self):
        path = self.write_log([{"pair": "EURUSD",
                                "closed_at": iso(datetime.now(timezone.utc))}])
        self.assertEqual(a3.pairs_in_cooldown(path, 0), set())

    def test_a_corrupt_log_line_does_not_take_the_run_down(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json\n")
            f.write(json.dumps({"pair": "GBPUSD",
                                "closed_at": iso(datetime.now(timezone.utc))}) + "\n")
        self.assertEqual(a3.pairs_in_cooldown(path, 6), {"GBPUSD"})

    def test_missing_log_is_not_an_error(self):
        self.assertEqual(a3.pairs_in_cooldown("no-such-log.jsonl", 6), set())

    def test_closed_and_non_trade_entries_are_dropped_on_load(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"pair": "EURUSD", "action": "Buy", "closed": False},
                       {"pair": "GBPUSD", "action": "Sell", "closed": True},
                       {"pair": "AUDUSD", "action": "Wait"}], f)
        self.assertEqual([t["pair"] for t in a3.load_open_trades(path)], ["EURUSD"])

    def test_the_legacy_single_dict_format_is_still_readable(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"pair": "EURUSD", "action": "Buy", "closed": False}, f)
        self.assertEqual(len(a3.load_open_trades(path)), 1)


if __name__ == "__main__":
    unittest.main()
