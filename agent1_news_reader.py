"""
Agent 1 - News Reader
=======================
Pulls today's high/medium-impact forex economic calendar events (ForexFactory
public calendar feed) plus recent forex news headlines (RSS), and writes a
structured JSON summary that Agent 3 (Signal Synthesizer) can consume.

Usage:
    python agent1_news_reader.py
    python agent1_news_reader.py --impact high medium --out news_summary.json

Output:
    Prints JSON to stdout, and writes to news_summary.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import feedparser
import requests

# ForexFactory publishes a public JSON calendar feed used widely by trading tools.
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Free forex/econ news RSS feeds (no API key required)
NEWS_FEEDS = {
    "Investing.com - Forex News": "https://www.investing.com/rss/news_1.rss",
    "FXStreet - News": "https://www.fxstreet.com/rss/news",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (forex-signal-agent/1.0)"}


def fetch_calendar_events(impact_filter):
    """Fetch today's economic calendar events filtered by impact level."""
    try:
        resp = requests.get(FF_CALENDAR_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        return [], f"calendar fetch failed: {e}"

    today = datetime.now(timezone.utc).date()
    todays_events = []
    for ev in events:
        try:
            ev_date = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).date()
        except Exception:
            continue
        if ev_date != today:
            continue
        impact = str(ev.get("impact", "")).lower()
        if impact_filter and impact not in impact_filter:
            continue
        todays_events.append({
            "time": ev.get("date"),
            "currency": ev.get("country"),
            "title": ev.get("title"),
            "impact": ev.get("impact"),
            "forecast": ev.get("forecast"),
            "previous": ev.get("previous"),
            "actual": ev.get("actual"),
        })

    return todays_events, None


def fetch_news_headlines(max_per_feed=8):
    """Fetch recent headlines from configured RSS feeds."""
    headlines = []
    errors = []
    for source, url in NEWS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            if feed.bozo and not feed.entries:
                errors.append(f"{source}: {feed.bozo_exception}")
                continue
            for entry in feed.entries[:max_per_feed]:
                headlines.append({
                    "source": source,
                    "title": entry.get("title"),
                    "published": entry.get("published", entry.get("updated")),
                    "link": entry.get("link"),
                })
        except Exception as e:
            errors.append(f"{source}: {e}")

    return headlines, errors


def main():
    parser = argparse.ArgumentParser(description="Agent 1 - News Reader")
    parser.add_argument("--impact", nargs="*", default=["high", "medium"],
                         help="Calendar impact levels to include (high, medium, low)")
    parser.add_argument("--headlines-per-feed", type=int, default=8,
                         help="Max news headlines to pull per RSS feed")
    parser.add_argument("--out", default="news_summary.json", help="Output JSON file path")
    args = parser.parse_args()

    impact_filter = {i.lower() for i in args.impact}

    events, cal_error = fetch_calendar_events(impact_filter)
    headlines, news_errors = fetch_news_headlines(args.headlines_per_feed)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calendar_events_today": events,
        "calendar_error": cal_error,
        "news_headlines": headlines,
        "news_errors": news_errors,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if cal_error:
        print(f"[warn] {cal_error}", file=sys.stderr)
    for err in news_errors:
        print(f"[warn] {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
