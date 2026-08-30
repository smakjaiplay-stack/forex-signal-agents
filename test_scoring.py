"""
Unit tests for scoring.py and validate.py's statistics.

The pipeline's GitHub Actions workflow runs the whole test suite as a gate
before anything reaches LINE, so these cover the properties that would break
signals silently: a NaN indicator poisoning the score, a component escaping
its [-1, +1] range, or the rank statistics mis-handling ties.
"""

import json
import math
import os
import tempfile
import unittest

import scoring
import validate


def base_row(**overrides):
    """A neutral, self-consistent indicator row. Override one field per test
    so each assertion isolates a single component."""
    row = {
        "Close": 1.1000,
        "ema20": 1.1000, "ema50": 1.1000, "ema80": 1.1000, "ema200": 1.1000,
        "rsi14": 50.0, "macd_histogram": 0.0, "atr14": 0.0010,
        "plus_di": 25.0, "minus_di": 25.0,
        "support": 1.0900, "resistance": 1.1100,
    }
    row.update(overrides)
    return row


class TestComponentRanges(unittest.TestCase):
    def test_every_component_stays_within_unit_range(self):
        """Extreme inputs must saturate, not escape.

        A component that can exceed 1 would dominate the weighted sum no
        matter what the weights say, which is how one indicator quietly
        becomes the whole strategy."""
        extremes = [
            base_row(ema20=99.0, ema200=99.0, macd_histogram=50.0, rsi14=100.0,
                     plus_di=100.0, minus_di=0.0, Close=1.1100),
            base_row(ema20=-99.0, ema200=-99.0, macd_histogram=-50.0, rsi14=0.0,
                     plus_di=0.0, minus_di=100.0, Close=1.0900),
        ]
        for row in extremes:
            _, values, _ = scoring.score_from_row(row)
            for name, v in values.items():
                self.assertGreaterEqual(v, -1.0, f"{name} below -1")
                self.assertLessEqual(v, 1.0, f"{name} above +1")

    def test_score_stays_within_score_scale(self):
        row = base_row(ema20=99.0, ema200=99.0, macd_histogram=50.0, rsi14=100.0,
                       plus_di=100.0, minus_di=0.0, Close=1.1100)
        score, _, _ = scoring.score_from_row(row)
        self.assertLessEqual(abs(score), scoring.SCORE_SCALE + 1e-9)

    def test_neutral_row_scores_zero(self):
        score, values, _ = scoring.score_from_row(base_row())
        self.assertAlmostEqual(score, 0.0, places=9)
        self.assertTrue(all(abs(v) < 1e-9 for v in values.values()))


class TestNaNHandling(unittest.TestCase):
    def test_missing_indicators_do_not_produce_nan(self):
        """An unavailable indicator must mean 'no opinion', not NaN.

        A single NaN makes the whole score NaN, and a NaN score compares
        False against every threshold - so the pair vanishes from the ranking
        instead of merely scoring weakly."""
        for field in ("rsi14", "atr14", "plus_di", "minus_di"):
            with self.subTest(field=field):
                score, values, _ = scoring.score_from_row(base_row(**{field: None}))
                self.assertTrue(math.isfinite(score))
                self.assertTrue(all(math.isfinite(v) for v in values.values()))

    def test_nan_atr_does_not_propagate(self):
        score, values, _ = scoring.score_from_row(base_row(atr14=float("nan")))
        self.assertTrue(math.isfinite(score))

    def test_zero_atr_does_not_divide_by_zero(self):
        score, values, _ = scoring.score_from_row(base_row(atr14=0.0))
        self.assertTrue(math.isfinite(score))
        self.assertEqual(values["ema_sep"], 0.0)

    def test_flat_range_does_not_divide_by_zero(self):
        score, values, _ = scoring.score_from_row(
            base_row(support=1.1000, resistance=1.1000))
        self.assertTrue(math.isfinite(score))
        self.assertEqual(values["range_pos"], 0.0)


class TestComponentSigns(unittest.TestCase):
    def test_ema_separation_is_bullish_when_fast_leads(self):
        _, values, _ = scoring.score_from_row(base_row(ema20=1.1010))
        self.assertGreater(values["ema_sep"], 0)

    def test_rsi_components_point_in_opposite_directions(self):
        """rsi_dev reads momentum, rsi_extreme reads mean-reversion.

        They are deliberately opposed so validate.py can measure which sign
        is right instead of the choice being baked in - see scoring.py."""
        _, values, _ = scoring.score_from_row(base_row(rsi14=85.0))
        self.assertGreater(values["rsi_dev"], 0)
        self.assertLess(values["rsi_extreme"], 0)

    def test_rsi_extreme_is_silent_in_the_middle(self):
        for rsi in (35.0, 50.0, 65.0):
            _, values, _ = scoring.score_from_row(base_row(rsi14=rsi))
            self.assertEqual(values["rsi_extreme"], 0.0)

    def test_range_pos_maps_support_to_minus_one(self):
        _, values, _ = scoring.score_from_row(base_row(Close=1.0900))
        self.assertAlmostEqual(values["range_pos"], -1.0, places=6)

    def test_di_spread_follows_the_dominant_side(self):
        _, values, _ = scoring.score_from_row(base_row(plus_di=40.0, minus_di=10.0))
        self.assertGreater(values["di_spread"], 0)


class TestWeights(unittest.TestCase):
    def test_zero_weights_do_not_divide_by_zero(self):
        score, contributions = scoring.compute_score(
            {c: 1.0 for c in scoring.COMPONENTS},
            weights={c: 0.0 for c in scoring.COMPONENTS})
        self.assertEqual(score, 0.0)

    def test_rescaling_keeps_range_fixed_when_weights_change(self):
        """Halving a weight must not silently make --min-score stricter."""
        values = {c: 1.0 for c in scoring.COMPONENTS}
        full, _ = scoring.compute_score(values, weights=scoring.DEFAULT_WEIGHTS)
        halved = dict(scoring.DEFAULT_WEIGHTS)
        halved["rsi_extreme"] = 0.5
        partial, _ = scoring.compute_score(values, weights=halved)
        self.assertAlmostEqual(full, scoring.SCORE_SCALE, places=9)
        self.assertAlmostEqual(partial, scoring.SCORE_SCALE, places=9)

    def test_missing_weights_file_falls_back_to_defaults(self):
        self.assertIsNone(scoring.load_ic_weights("definitely-not-a-file.json"))

    def test_malformed_weights_file_is_ignored(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            f.write("{not json at all")
            path = f.name
        try:
            self.assertIsNone(scoring.load_ic_weights(path))
        finally:
            os.unlink(path)


class TestBias(unittest.TestCase):
    def test_threshold_is_inclusive_on_both_sides(self):
        self.assertEqual(scoring.bias_for_score(1.2, 1.2), "bullish")
        self.assertEqual(scoring.bias_for_score(-1.2, 1.2), "bearish")
        self.assertEqual(scoring.bias_for_score(1.19, 1.2), "neutral")


class TestSpearman(unittest.TestCase):
    def test_perfect_monotonic_relationships(self):
        self.assertAlmostEqual(validate.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(validate.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_ties_are_rank_averaged(self):
        """Without tie-averaging, a component that is 0.0 on half the trades
        would correlate with nothing more than the file's row order."""
        self.assertAlmostEqual(validate.spearman([1, 1, 1, 1], [1, 2, 3, 4]), 0.0)

    def test_too_few_points_returns_zero(self):
        self.assertEqual(validate.spearman([1, 2], [1, 2]), 0.0)

    def test_ranks_average_tied_positions(self):
        self.assertEqual(validate._ranks([5, 5, 9]), [1.5, 1.5, 3.0])


class TestExpectancyStats(unittest.TestCase):
    def test_interval_containing_zero_is_reported_as_such(self):
        s = validate.mean_stats([1.0, -1.0] * 50)
        self.assertLess(s["ci"][0], 0)
        self.assertGreater(s["ci"][1], 0)
        self.assertAlmostEqual(s["mean"], 0.0, places=9)

    def test_single_trade_has_no_spread(self):
        s = validate.mean_stats([0.5])
        self.assertEqual(s["n"], 1)
        self.assertEqual(s["sd"], 0.0)
        self.assertEqual(s["t"], 0.0)

    def test_empty_input_is_safe(self):
        self.assertEqual(validate.mean_stats([])["n"], 0)

    def test_bootstrap_is_deterministic_for_a_fixed_seed(self):
        values = [1.0, -1.0, 0.5, -0.5, 2.0] * 10
        self.assertEqual(validate.bootstrap_ci(values, iterations=200),
                         validate.bootstrap_ci(values, iterations=200))


class TestNegativeICWeighting(unittest.TestCase):
    def test_negative_ic_is_halved_never_zeroed(self):
        """Zeroing anti-predictive components is what inverted the scoring in
        the system this technique came from - each IC is measured while the
        others are active, so removing one changes what the rest do."""
        results = [
            ("rsi_dev", 0.13, 2.1, 255, ""),
            ("rsi_extreme", -0.13, -2.0, 255, ""),
        ]
        weights = validate.build_weights(results)
        self.assertEqual(weights["rsi_dev"], 1.0)
        self.assertEqual(weights["rsi_extreme"], validate.NEGATIVE_IC_WEIGHT)
        self.assertGreater(weights["rsi_extreme"], 0.0)


class TestDirection(unittest.TestCase):
    def test_sell_flips_component_agreement(self):
        """A bearish component on a Sell agrees with the trade, so it must
        enter the IC with a positive sign."""
        self.assertEqual(validate.direction({"action": "Buy"}), 1.0)
        self.assertEqual(validate.direction({"action": "Sell"}), -1.0)


class TestTimeframeIsShared(unittest.TestCase):
    """The live path and the backtest must score the same bar size.

    They did not. Agent 2 defaulted to 15m, backtest.py defaulted to 1h, and
    each file carried a comment asserting the other agreed - so the live system
    scored 15m bars against DEFAULT_MIN_SCORE, score_calibration.json and a set
    of component ICs that were every one of them measured on 1h bars. Nothing
    failed; the numbers were simply describing a different system.

    A comment cannot hold that invariant. These assertions can.
    """

    def setUp(self):
        # Imported here rather than at module scope: both pull in yfinance, and
        # the rest of this file must stay importable without it.
        import agent2_technical_analyzer
        import backtest
        self.agent2 = agent2_technical_analyzer
        self.backtest = backtest

    def test_agent2_and_backtest_read_the_same_interval(self):
        live = self.agent2.build_parser().get_default("interval")
        historical = self.backtest.build_parser().get_default("interval")
        self.assertEqual(live, historical)
        self.assertEqual(live, scoring.DEFAULT_INTERVAL)

    def test_live_period_clears_the_ema200_warmup(self):
        """A period shorter than MIN_BARS makes every pair error out.

        Measured against Yahoo, 1h returns ~23.8 bars per calendar day of
        forex history, so the check below is deliberately conservative.
        """
        period = self.agent2.build_parser().get_default("period")
        self.assertTrue(period.endswith("d"), f"expected a day-based period, got {period!r}")
        days = int(period[:-1])
        bars_per_calendar_day = 20  # under the ~23.8 observed, to leave holiday slack
        self.assertGreaterEqual(days * bars_per_calendar_day, self.agent2.MIN_BARS)

    def test_backtest_hold_window_matches_agent5(self):
        """--bars-per-day is also the maximum hold, and Agent 5 times a live
        trade out at MAX_HOLD_HOURS. On hourly bars the two are the same number,
        and they stop being the same number the moment the interval changes."""
        self.assertEqual(scoring.DEFAULT_INTERVAL, "1h")
        self.assertEqual(self.backtest.build_parser().get_default("bars_per_day"),
                         scoring.BARS_PER_DAY)
        import agent5_price_watcher
        self.assertEqual(scoring.BARS_PER_DAY, agent5_price_watcher.MAX_HOLD_HOURS)


if __name__ == "__main__":
    unittest.main()
