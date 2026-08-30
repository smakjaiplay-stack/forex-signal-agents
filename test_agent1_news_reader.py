"""
Unit tests for Agent 1 - News Reader.

Agent 1 had no tests, and its failure mode was the quiet kind: an unreachable
calendar produced a well-formed news_summary.json with a fresh timestamp and an
empty event list, so "nothing is scheduled" and "we never found out" were the
same file downstream. Most of what follows is about keeping those two apart.

No network: requests.get and feedparser.parse are replaced per test.

    python -m unittest test_agent1_news_reader -v
"""

import unittest
from datetime import datetime, timedelta, timezone

import agent1_news_reader as a1


TODAY = datetime.now(timezone.utc)


def event(currency="USD", impact="High", title="Non-Farm Payrolls",
          when=None, actual=None):
    return {
        "date": (when or TODAY).isoformat(),
        "country": currency,
        "impact": impact,
        "title": title,
        "forecast": "180K",
        "previous": "175K",
        "actual": actual,
    }


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload if payload is not None else []
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


class CalendarHarness(unittest.TestCase):
    def fetch(self, responses, impact=("high", "medium")):
        """Run fetch_calendar_events against a scripted requests.get."""
        calls = {"n": 0}

        def fake_get(*args, **kwargs):
            item = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            if isinstance(item, Exception):
                raise item
            return item

        original = a1.requests.get
        a1.requests.get = fake_get
        try:
            events, error = a1.fetch_calendar_events(set(impact), sleep=lambda s: None)
        finally:
            a1.requests.get = original
        return events, error, calls["n"]


class TestCalendarFetch(CalendarHarness):
    def test_todays_high_impact_event_is_kept(self):
        events, error, _ = self.fetch([FakeResponse([event()])])
        self.assertIsNone(error)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["currency"], "USD")

    def test_impact_filter_drops_what_it_should(self):
        events, _, _ = self.fetch([FakeResponse([event(impact="Low")])])
        self.assertEqual(events, [])

    def test_events_on_other_days_are_dropped(self):
        payload = [event(when=TODAY + timedelta(days=2))]
        events, _, _ = self.fetch([FakeResponse(payload)])
        self.assertEqual(events, [])

    def test_an_unparseable_date_is_skipped_not_fatal(self):
        bad = event()
        bad["date"] = "whenever"
        events, error, _ = self.fetch([FakeResponse([bad, event()])])
        self.assertIsNone(error)
        self.assertEqual(len(events), 1)

    def test_a_released_figure_travels_with_the_event(self):
        """Agent 3 and Agent 6 both key off `actual` being empty."""
        events, _, _ = self.fetch([FakeResponse([event(actual="192K")])])
        self.assertEqual(events[0]["actual"], "192K")


class TestCalendarRetries(CalendarHarness):
    def test_a_transient_failure_is_retried_and_can_succeed(self):
        events, error, attempts = self.fetch(
            [FakeResponse(status=503), FakeResponse([event()])])
        self.assertIsNone(error)
        self.assertEqual((len(events), attempts), (1, 2))

    def test_persistent_failure_reports_an_error_rather_than_no_events(self):
        """The distinction the whole file turns on: this must not come back as
        an empty calendar, because Agent 6 reads the error as news_unverified
        and penalises the signal for it."""
        events, error, attempts = self.fetch([RuntimeError("connection refused")])
        self.assertEqual(events, [])
        self.assertIsNotNone(error)
        self.assertIn("connection refused", error)
        self.assertEqual(attempts, a1.FETCH_ATTEMPTS)

    def test_the_error_names_how_many_attempts_were_made(self):
        _, error, _ = self.fetch([RuntimeError("nope")])
        self.assertIn(str(a1.FETCH_ATTEMPTS), error)


class FakeFeed:
    def __init__(self, entries=None, bozo=False, exception=None):
        self.entries = entries or []
        self.bozo = bozo
        self.bozo_exception = exception


class TestHeadlines(unittest.TestCase):
    def fetch(self, per_url, max_per_feed=8):
        """`per_url` maps each configured feed to the FakeFeed it returns.

        Keyed by URL rather than by call order: retries call the same URL more
        than once, so a positional script would hand the second feed's result
        to the first feed's retry.
        """
        urls = list(a1.NEWS_FEEDS.values())
        if not isinstance(per_url, dict):
            per_url = {url: per_url[min(i, len(per_url) - 1)]
                       for i, url in enumerate(urls)}
        calls = {"n": 0}

        def fake_parse(url):
            calls["n"] += 1
            return per_url[url]

        original = a1.feedparser.parse
        a1.feedparser.parse = fake_parse
        try:
            return a1.fetch_news_headlines(max_per_feed, sleep=lambda s: None) + (calls["n"],)
        finally:
            a1.feedparser.parse = original

    def entry(self, title="ECB holds rates"):
        return {"title": title, "published": "Thu, 13 Aug 2026 10:00:00 GMT",
                "link": "https://example.invalid/story"}

    def test_headlines_are_collected_from_every_feed(self):
        headlines, errors, _ = self.fetch([FakeFeed([self.entry()])])
        self.assertEqual(len(headlines), len(a1.NEWS_FEEDS))
        self.assertEqual(errors, [])

    def test_max_per_feed_is_respected(self):
        feed = FakeFeed([self.entry(f"story {i}") for i in range(20)])
        headlines, _, _ = self.fetch([feed], max_per_feed=3)
        self.assertEqual(len(headlines), 3 * len(a1.NEWS_FEEDS))

    def test_a_broken_feed_is_reported_and_the_others_still_run(self):
        headlines, errors, _ = self.fetch(
            [FakeFeed(bozo=True, exception="malformed XML"), FakeFeed([self.entry()])])
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(headlines), 1)

    def test_an_empty_feed_is_retried(self):
        _, _, attempts = self.fetch([FakeFeed([])])
        self.assertGreaterEqual(attempts, a1.FETCH_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
