"""
Agent 3 - Signal Synthesizer (Rule-Based, No AI API required)
================================================================
Reads Agent 1's news_summary.json and Agent 2's technical_analysis.json,
combines them with simple rules, and picks the single most probable
currency pair signal for the day. Writes signal.json for Agent 4 (LINE
Notifier) to consume.

Usage:
    python agent3_signal_synthesizer.py
    python agent3_signal_synthesizer.py --news news_summary.json --technical technical_analysis.json

Logic (rule-based):
    1. Start from each pair's technical score/bias from Agent 2.
    2. Check Agent 1's high-impact calendar events for the currencies
       involved in each pair. If a high-impact event for that currency
       hasn't been released yet today ("actual" is null), treat the
       pair as risky (confidence downgraded / flagged) because price
       can whipsaw around the release.
    3. Pick the pair with the strongest |score| among the non-risky
       (or least-risky) candidates as the final signal.
"""

import argparse
import json
import sys
from datetime import datetime, timezone

# Currency -> which side of the pair it is doesn't matter for this rule;
# we just check if either currency in the pair has a pending high-impact event.
def currencies_in_pair(pair: str):
    pair = pair.upper()
    return pair[:3], pair[3:6]


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[error] could not read {path}: {e}", file=sys.stderr)
        return None


def pending_high_impact_currencies(news_data):
    """Return set of currencies with a high-impact event today that hasn't
    released yet (actual is null/empty)."""
    pending = set()
    if not news_data:
        return pending
    for ev in news_data.get("calendar_events_today", []):
        impact = str(ev.get("impact", "")).lower()
        actual = ev.get("actual")
        if impact == "high" and not actual:
            currency = ev.get("currency")
            if currency:
                pending.add(currency.upper())
    return pending


def synthesize(news_data, tech_data):
    pending_currencies = pending_high_impact_currencies(news_data)

    candidates = []
    for result in tech_data.get("results", []):
        if "error" in result:
            continue
        pair = result["pair"]
        base, quote = currencies_in_pair(pair)
        risky = bool(pending_currencies & {base, quote})

        candidates.append({
            "pair": pair,
            "bias": result["bias"],
            "score": result["score"],
            "abs_score": abs(result["score"]),
            "risky_pending_news": risky,
            "reasons": result["reasons"],
            "last_close": result["last_close"],
            "support": result["support"],
            "resistance": result["resistance"],
        })

    if not candidates:
        return {
            "signal": None,
            "note": "No valid technical results to synthesize a signal from.",
        }

    # Prefer non-risky candidates; among those, highest |score|.
    # If all candidates are risky, fall back to highest |score| overall but flag it.
    non_risky = [c for c in candidates if not c["risky_pending_news"]]
    pool = non_risky if non_risky else candidates
    pool_sorted = sorted(pool, key=lambda c: c["abs_score"], reverse=True)

    best = pool_sorted[0]

    if best["abs_score"] < 2:
        confidence = "low"
    elif best["abs_score"] == 2:
        confidence = "medium"
    else:
        confidence = "high"

    if best["risky_pending_news"]:
        confidence = "low"

    signal = {
        "pair": best["pair"],
        "direction": best["bias"],
        "confidence": confidence,
        "score": best["score"],
        "entry_reference": best["last_close"],
        "support": best["support"],
        "resistance": best["resistance"],
        "reasons": best["reasons"],
        "pending_high_impact_news": best["risky_pending_news"],
    }

    return {
        "signal": signal,
        "all_candidates": sorted(candidates, key=lambda c: c["abs_score"], reverse=True),
    }


def main():
    parser = argparse.ArgumentParser(description="Agent 3 - Signal Synthesizer (rule-based)")
    parser.add_argument("--news", default="news_summary.json", help="Path to Agent 1 output")
    parser.add_argument("--technical", default="technical_analysis.json", help="Path to Agent 2 output")
    parser.add_argument("--out", default="signal.json", help="Output JSON file path")
    args = parser.parse_args()

    news_data = load_json(args.news)
    tech_data = load_json(args.technical)

    if tech_data is None:
        print("[error] technical analysis data is required to synthesize a signal", file=sys.stderr)
        sys.exit(1)

    result = synthesize(news_data, tech_data)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
