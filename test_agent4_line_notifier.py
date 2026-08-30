"""
Unit tests for Agent 4 - LINE Notifier.

Agent 4 is the only agent whose output a human reads, so what it chooses to
put on a card is a correctness question, not a presentation one. The card used
to lead with a reward/risk measured to TP3 - a target being hit on 1.9% of
trades, and reading "R:R 28.4" on the run that produced it.

No network: send_broadcast is injected with a fake in every test that needs it.

    python -m unittest test_agent4_line_notifier -v
"""

import unittest

import agent4_line_notifier as a4


def make_signal(**overrides):
    signal = {
        "pair": "EURUSD",
        "action": "Buy",
        "direction": "bullish",
        "confidence": "medium",
        "possibility_percent": 55,
        "status": "Active",
        "time_frame": "1h",
        "open_price": 1.10000,
        "take_profit_1": 1.10120,
        "take_profit_2": 1.10240,
        "take_profit_3": 1.10360,
        "stop_loss": 1.09880,
        "risk_reward_tp1": 1.0,
        "risk_reward": 3.0,
        "support": 1.09500,
        "resistance": 1.10500,
        "last_bar_time": "2026-08-13 11:00 UTC",
        "data_age_minutes": 12.0,
        "reasons": ["EMA20 above EMA50 (short-term uptrend)"],
        "qc_status": "approved",
        "qc_flags": [],
        "new": True,
    }
    signal.update(overrides)
    return signal


class FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class TestCardContents(unittest.TestCase):
    def card(self, **overrides):
        return "\n".join(a4.build_signal_block(make_signal(**overrides)))

    def test_reward_risk_leads_with_tp1(self):
        """TP1 is the target most trades actually reach. Quoting only the TP3
        ratio is how a card came to advertise 28.4 to 1."""
        text = self.card()
        self.assertIn("R:R 1.0 to TP1", text)
        self.assertIn("(3.0 to TP3)", text)

    def test_a_card_without_a_tp1_ratio_still_shows_what_it_has(self):
        text = self.card(risk_reward_tp1=None)
        self.assertIn("R:R 3.0 to TP3", text)

    def test_price_age_is_stated_not_hidden(self):
        self.assertIn("12 min ago", self.card())

    def test_a_wait_card_carries_no_targets(self):
        text = self.card(action="Wait", status="No Trade")
        self.assertNotIn("TP1:", text)

    def test_a_qc_downgrade_says_do_not_enter(self):
        text = self.card(action="Wait", original_action="Buy", qc_status="downgraded",
                         qc_flags=["low_confidence"], qc_notes="below breakeven")
        self.assertIn("QC WARNING", text)
        self.assertIn("do not enter this trade", text)
        self.assertIn("NO TRADE (QC held back a Buy)", text)

    def test_unproven_edge_is_spelled_out_on_a_still_tradable_card(self):
        """Published under --allow-unproven-edge the action is NOT downgraded,
        so without this the card reads like any other approved trade."""
        text = self.card(qc_status="downgraded", qc_flags=["unproven_edge"])
        self.assertIn("UNPROVEN EDGE", text)

    def test_stale_price_is_spelled_out(self):
        text = self.card(qc_status="downgraded", qc_flags=["stale_price"])
        self.assertIn("STALE PRICE", text)

    def test_unverified_news_is_spelled_out(self):
        """"No events" and "we could not look" must not read the same."""
        text = self.card(qc_status="downgraded", qc_flags=["news_unverified"])
        self.assertIn("NEWS NOT CHECKED", text)

    def test_an_approved_card_carries_no_qc_noise(self):
        self.assertNotIn("QC WARNING", self.card())


class TestSelection(unittest.TestCase):
    def test_only_new_signals_are_sent_by_default(self):
        data = {"signals": [make_signal(pair="EURUSD", new=True),
                            make_signal(pair="GBPUSD", new=False)]}
        self.assertEqual([s["pair"] for s in a4.select_signals(data, False)], ["EURUSD"])

    def test_all_overrides_the_new_filter(self):
        data = {"signals": [make_signal(new=False)]}
        self.assertEqual(len(a4.select_signals(data, True)), 1)

    def test_blocked_signals_are_dropped_even_under_all(self):
        """The entire point of the QC layer is that these never reach a phone."""
        data = {"signals": [make_signal(qc_status="blocked", new=True)]}
        self.assertEqual(a4.select_signals(data, True), [])

    def test_the_legacy_single_signal_shape_is_still_read(self):
        data = {"signal": make_signal()}
        self.assertEqual(len(a4.select_signals(data, True)), 1)

    def test_no_signals_produces_a_message_rather_than_a_crash(self):
        self.assertIn("no valid signal", a4.build_message_text({"note": "quiet"}, []))


class TestLineMessageLimits(unittest.TestCase):
    """LINE rejects a text over 5000 chars, and the old code sent one blob."""

    def test_a_short_message_is_left_alone(self):
        self.assertEqual(a4.split_for_line("hello"), ["hello"])

    def test_a_long_message_is_split_on_card_boundaries(self):
        text = "\n─────────────".join(["HEADER"] + ["x" * 2000] * 6)
        parts = a4.split_for_line(text)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part), a4.LINE_MAX_TEXT_CHARS)

    def test_no_card_is_torn_in_half(self):
        text = "\n─────────────".join(["HEADER"] + [f"CARD{i} " + "x" * 1500
                                                    for i in range(6)])
        rejoined = "".join(p.split("\n", 1)[1] for p in a4.split_for_line(text))
        for i in range(6):
            self.assertIn(f"CARD{i}", rejoined)

    def test_a_single_oversized_card_is_cut_rather_than_rejected(self):
        parts = a4.split_for_line("y" * 12000)
        self.assertTrue(all(len(p) <= a4.LINE_MAX_TEXT_CHARS for p in parts))

    def test_overflow_past_the_message_cap_says_so(self):
        text = "\n─────────────".join(["HEADER"] + ["z" * 4500] * 12)
        parts = a4.split_for_line(text)
        self.assertEqual(len(parts), a4.LINE_MAX_MESSAGES)
        self.assertIn("did not fit", parts[-1])

    def test_parts_are_numbered_so_a_split_run_reads_in_order(self):
        text = "\n─────────────".join(["HEADER"] + ["x" * 2000] * 6)
        self.assertTrue(a4.split_for_line(text)[0].startswith("(1/"))


class TestSendRetries(unittest.TestCase):
    def send(self, responses):
        """Return (result, attempt_count) for a scripted sequence."""
        calls = {"n": 0}

        def fake_post(*args, **kwargs):
            resp = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            if isinstance(resp, Exception):
                raise resp
            return resp

        original = a4.requests.post
        a4.requests.post = fake_post
        try:
            resp = a4.send_broadcast("token", "text", sleep=lambda s: None)
        finally:
            a4.requests.post = original
        return resp, calls["n"]

    def test_a_success_is_not_retried(self):
        resp, attempts = self.send([FakeResponse(200)])
        self.assertEqual((resp.status_code, attempts), (200, 1))

    def test_rate_limiting_is_retried_and_can_succeed(self):
        resp, attempts = self.send([FakeResponse(429), FakeResponse(200)])
        self.assertEqual((resp.status_code, attempts), (200, 2))

    def test_a_server_error_is_retried_up_to_the_limit(self):
        resp, attempts = self.send([FakeResponse(503)])
        self.assertEqual((resp.status_code, attempts), (503, a4.RETRY_ATTEMPTS))

    def test_a_bad_request_is_not_retried(self):
        """400 and 401 mean this code or the token is wrong. Sending the same
        broken request three times just fails three times."""
        resp, attempts = self.send([FakeResponse(400, "bad token")])
        self.assertEqual((resp.status_code, attempts), (400, 1))

    def test_a_connection_error_is_retried_then_raised(self):
        with self.assertRaises(a4.requests.RequestException):
            self.send([a4.requests.RequestException("boom")])


if __name__ == "__main__":
    unittest.main()
