"""
Agent 3 - Signal Synthesizer (Rule-Based, No AI API required)
================================================================
Reads Agent 1's news_summary.json and Agent 2's technical_analysis.json,
combines them with simple rules, and picks the strongest N currency pair
signals. Writes signal.json for Agent 4 (LINE Notifier) to consume.

Usage:
    python agent3_signal_synthesizer.py
    python agent3_signal_synthesizer.py --top 3 --max-open 3
    python agent3_signal_synthesizer.py --news news_summary.json --technical technical_analysis.json

Logic (rule-based):
    1. Start from each pair's technical score/bias from Agent 2.
    2. Check Agent 1's high-impact calendar events for the currencies
       involved in each pair. If a high-impact event for that currency
       hasn't been released yet today ("actual" is null), treat the
       pair as risky (confidence downgraded / flagged) because price
       can whipsaw around the release.
    3. Rank all pairs by |score| and keep the top N non-risky candidates
       (falling back to the single best overall if nothing qualifies).
    4. Turn each bias into a trade card: Action (Buy/Sell/Wait),
       Possibility %, Take Profit 1-3, Stop Loss, Status, Time frame -
       styled like a typical "forex signal" app card.

Because the pipeline now runs several times a day (not once at market open),
open_trade.json is *merged*, not overwritten: trades Agent 5 is still
watching are kept, and only pairs that aren't already open - and aren't in
the post-close cooldown window - get added. Each signal is flagged "new"
so Agent 4 only notifies about signals you haven't been told about yet.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

DEFAULT_TOP_N = 3
DEFAULT_MAX_OPEN = 3
DEFAULT_COOLDOWN_HOURS = 6
DEFAULT_MIN_SCORE = 2
# Stop-loss is capped at this many ATRs from entry, and a signal whose
# reward/risk falls below DEFAULT_MIN_RR is not published at all.
DEFAULT_ATR_MULT = 1.5
DEFAULT_MIN_RR = 1.5


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


def decimals_for_pair(pair: str) -> int:
    """Metals move in bigger increments than most FX pairs — round accordingly."""
    if pair.upper() in ("XAUUSD", "XAGUSD"):
        return 2
    if pair.upper().endswith("JPY"):
        return 3
    return 5


def possibility_percent(abs_score: int, risky: bool) -> int:
    """Turn the rule-based score into a rough confidence percentage (30-95%)."""
    base = 50 + abs_score * 10  # score 0->50, 1->60, 2->70, 3->80
    if risky:
        base -= 15
    return max(30, min(base, 95))


def build_trade_levels(action: str, entry: float, support: float, resistance: float, ndigits: int,
                       atr: float = None, atr_mult: float = DEFAULT_ATR_MULT):
    """Compute Take Profit 1/2/3 and Stop Loss from entry + support/resistance.

    Buy: target = resistance, stop = support (TP1 closest to entry, TP3 furthest).
    Sell: target = support, stop = resistance.
    Falls back to a small percentage-based buffer if support/resistance don't
    bracket the entry price sensibly (can happen with thin/odd data).

    Taking the stop straight from the far edge of the range is what produced
    trades risking 11 to make 1: when price is already sitting near one edge,
    the target is a few pips away while the stop is most of the range wide.
    So when ATR is available the stop is pulled in to at most `atr_mult` ATRs
    from entry. That bounds risk but can't manufacture reward — a target
    that's simply too close still yields a poor ratio, which is what the
    `min_rr` filter in synthesize() is for.
    """
    if action not in ("Buy", "Sell"):
        return None

    sane = support < entry < resistance
    if action == "Buy":
        target = resistance if sane else entry * 1.015
        stop = support if sane else entry * 0.9925
    else:  # Sell
        target = support if sane else entry * 0.985
        stop = resistance if sane else entry * 1.0075

    # atr_mult <= 0 means "no cap" — without this guard it would collapse the
    # stop onto the entry price, i.e. a trade that stops out instantly.
    if atr and atr > 0 and atr_mult > 0:
        max_risk = atr * atr_mult
        if abs(entry - stop) > max_risk:
            stop = entry - max_risk if action == "Buy" else entry + max_risk

    if action == "Buy":
        tp1 = entry + (target - entry) / 3
        tp2 = entry + (target - entry) * 2 / 3
    else:
        tp1 = entry - (entry - target) / 3
        tp2 = entry - (entry - target) * 2 / 3
    tp3 = target
    sl = stop

    return {
        "take_profit_1": round(tp1, ndigits),
        "take_profit_2": round(tp2, ndigits),
        "take_profit_3": round(tp3, ndigits),
        "stop_loss": round(sl, ndigits),
    }


def risk_reward(entry: float, tp3: float, sl: float):
    """Reward (entry->TP3) divided by risk (entry->SL). None if risk is zero."""
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    return abs(tp3 - entry) / risk


def build_signal(candidate, time_frame, atr_mult=DEFAULT_ATR_MULT):
    """Turn one ranked candidate into a full trade card."""
    if candidate["abs_score"] < 2:
        confidence = "low"
    elif candidate["abs_score"] == 2:
        confidence = "medium"
    else:
        confidence = "high"

    if candidate["risky_pending_news"]:
        confidence = "low"

    action = {"bullish": "Buy", "bearish": "Sell", "neutral": "Wait"}[candidate["bias"]]
    ndigits = decimals_for_pair(candidate["pair"])
    levels = build_trade_levels(
        action, candidate["last_close"], candidate["support"], candidate["resistance"], ndigits,
        atr=candidate.get("atr14"), atr_mult=atr_mult,
    )
    now_utc = datetime.now(timezone.utc)

    signal = {
        "pair": candidate["pair"],
        "action": action,
        "direction": candidate["bias"],
        "confidence": confidence,
        "possibility_percent": possibility_percent(candidate["abs_score"], candidate["risky_pending_news"]),
        "score": candidate["score"],
        "status": "Active" if action != "Wait" else "No Trade",
        "opening_time": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "last_update": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "open_price": round(candidate["last_close"], ndigits),
        "time_frame": time_frame,
        # How stale the candle behind open_price is, so the LINE card can say
        # "priced off a 15m bar that closed 4 min ago" instead of hiding it.
        "last_bar_time": candidate.get("last_bar_time"),
        "data_age_minutes": candidate.get("data_age_minutes"),
        "profit_loss": "Waiting",
        "trade_result": "Waiting",
        "support": round(candidate["support"], ndigits),
        "resistance": round(candidate["resistance"], ndigits),
        "reasons": candidate["reasons"],
        "pending_high_impact_news": candidate["risky_pending_news"],
        "comment": "High-impact news pending — expect volatility" if candidate["risky_pending_news"] else "-",
    }
    if levels:
        signal.update(levels)
        rr = risk_reward(signal["open_price"], levels["take_profit_3"], levels["stop_loss"])
        signal["risk_reward"] = round(rr, 2) if rr is not None else None
    return signal


def synthesize(news_data, tech_data, top_n=DEFAULT_TOP_N, min_score=DEFAULT_MIN_SCORE,
               min_rr=DEFAULT_MIN_RR, atr_mult=DEFAULT_ATR_MULT):
    pending_currencies = pending_high_impact_currencies(news_data)
    time_frame = tech_data.get("interval", "?")

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
            "atr14": result.get("atr14"),
            "last_bar_time": result.get("last_bar_time"),
            "data_age_minutes": result.get("data_age_minutes"),
        })

    if not candidates:
        return {
            "signals": [],
            "signal": None,
            "note": "No valid technical results to synthesize a signal from.",
        }

    # Prefer non-risky candidates; among those, highest |score|.
    # If all candidates are risky, fall back to highest |score| overall but flag it.
    non_risky = [c for c in candidates if not c["risky_pending_news"]]
    pool = non_risky if non_risky else candidates
    pool_sorted = sorted(pool, key=lambda c: c["abs_score"], reverse=True)

    # Everything that clears the bar, not just the single best — the whole
    # point of top-N is that you get more than one option per run.
    tradable = [c for c in pool_sorted if c["bias"] != "neutral" and c["abs_score"] >= min_score]

    # A strong-looking bias is still a bad trade if the target sits a few pips
    # away while the stop sits half a range away, so reward/risk decides what
    # actually gets published — not just the indicator score.
    signals, rejected = [], []
    for candidate in tradable:
        signal = build_signal(candidate, time_frame, atr_mult=atr_mult)
        rr = signal.get("risk_reward")
        if min_rr > 0 and (rr is None or rr < min_rr):
            rejected.append({
                "pair": signal["pair"],
                "action": signal["action"],
                "risk_reward": rr,
                "reason": f"reward/risk {rr} below minimum {min_rr}",
            })
            continue
        signals.append(signal)
        if len(signals) >= top_n:
            break

    if not signals and not tradable:
        # Nothing cleared the score bar — surface the strongest read anyway so
        # the run still reports what the market looks like.
        signals = [build_signal(pool_sorted[0], time_frame, atr_mult=atr_mult)]

    return {
        "signals": signals,
        # Kept so anything still reading the old single-signal shape works.
        "signal": signals[0] if signals else None,
        "rejected_low_rr": rejected,
        "all_candidates": sorted(candidates, key=lambda c: c["abs_score"], reverse=True),
    }


def load_open_trades(path):
    """Read open_trade.json, tolerating both the legacy single-dict format
    and the current list-of-trades format. Closed / non-trade entries are
    dropped — Agent 5 has already logged those to trades_log.jsonl."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    return [
        t for t in data
        if isinstance(t, dict) and t.get("action") in ("Buy", "Sell") and not t.get("closed")
    ]


def pairs_in_cooldown(trades_log_path, cooldown_hours):
    """Pairs whose trade closed within the last `cooldown_hours`.

    Without this, a pair that just stopped out gets re-entered on the very
    next pipeline run (4h later the indicators usually still look the same),
    which turns one bad read into a string of them."""
    if cooldown_hours <= 0:
        return set()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)
    cooling = set()
    try:
        with open(trades_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    closed_at = datetime.fromisoformat(str(rec["closed_at"]).replace("Z", "+00:00"))
                except Exception:
                    continue
                if closed_at.tzinfo is None:
                    closed_at = closed_at.replace(tzinfo=timezone.utc)
                if closed_at >= cutoff and rec.get("pair"):
                    cooling.add(rec["pair"])
    except FileNotFoundError:
        return set()
    return cooling


def make_open_trade(signal, opened_at):
    return {
        "pair": signal["pair"],
        "action": signal["action"],
        "entry_price": signal["open_price"],
        "take_profit_1": signal["take_profit_1"],
        "take_profit_2": signal["take_profit_2"],
        "take_profit_3": signal["take_profit_3"],
        "stop_loss": signal["stop_loss"],
        "current_sl": signal["stop_loss"],
        "opened_at": opened_at,
        "closed": False,
        "alerts_sent": {"tp1": False, "tp2": False, "tp3": False, "sl": False},
    }


def main():
    parser = argparse.ArgumentParser(description="Agent 3 - Signal Synthesizer (rule-based)")
    parser.add_argument("--news", default="news_summary.json", help="Path to Agent 1 output")
    parser.add_argument("--technical", default="technical_analysis.json", help="Path to Agent 2 output")
    parser.add_argument("--out", default="signal.json", help="Output JSON file path")
    parser.add_argument("--open-trade-out", default="open_trade.json",
                         help="Open-trade state file for Agent 5 (merged, not overwritten)")
    parser.add_argument("--trades-log", default="trades_log.jsonl",
                         help="Closed-trade log, used for the re-entry cooldown check")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                         help="How many signals to publish per run")
    parser.add_argument("--max-open", type=int, default=DEFAULT_MAX_OPEN,
                         help="Cap on simultaneously open trades across all runs")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE,
                         help="Minimum |technical score| for a pair to be publishable")
    parser.add_argument("--min-rr", type=float, default=DEFAULT_MIN_RR,
                         help="Drop signals whose reward/risk is below this (0 disables the filter)")
    parser.add_argument("--atr-mult", type=float, default=DEFAULT_ATR_MULT,
                         help="Cap the stop-loss at this many ATRs from entry")
    parser.add_argument("--cooldown-hours", type=float, default=DEFAULT_COOLDOWN_HOURS,
                         help="Don't re-enter a pair this soon after its last trade closed (0 disables)")
    args = parser.parse_args()

    news_data = load_json(args.news)
    tech_data = load_json(args.technical)

    if tech_data is None:
        print("[error] technical analysis data is required to synthesize a signal", file=sys.stderr)
        sys.exit(1)

    result = synthesize(news_data, tech_data, top_n=args.top, min_score=args.min_score,
                        min_rr=args.min_rr, atr_mult=args.atr_mult)
    generated_at = datetime.now(timezone.utc).isoformat()

    # Decide which signals are actually new *before* writing signal.json, so
    # Agent 4 can notify about those and stay quiet about the rest.
    open_trades = load_open_trades(args.open_trade_out)
    open_pairs = {t["pair"] for t in open_trades}
    cooling = pairs_in_cooldown(args.trades_log, args.cooldown_hours)

    new_trades = []
    for signal in result.get("signals", []):
        if signal["action"] not in ("Buy", "Sell"):
            signal["new"] = False
            signal["skip_reason"] = "no trade (neutral bias)"
        elif signal["pair"] in open_pairs:
            signal["new"] = False
            signal["skip_reason"] = "already open — Agent 5 is watching it"
        elif signal["pair"] in cooling:
            signal["new"] = False
            signal["skip_reason"] = f"cooldown ({args.cooldown_hours}h since last close)"
        elif len(open_trades) + len(new_trades) >= args.max_open:
            signal["new"] = False
            signal["skip_reason"] = f"max open trades reached ({args.max_open})"
        else:
            signal["new"] = True
            new_trades.append(make_open_trade(signal, generated_at))

    output = {
        "generated_at": generated_at,
        **result,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))

    with open(args.open_trade_out, "w", encoding="utf-8") as f:
        json.dump(open_trades + new_trades, f, ensure_ascii=False, indent=2)

    for r in result.get("rejected_low_rr", []):
        print(f"[skip] {r['pair']} {r['action']}: {r['reason']}", file=sys.stderr)

    print(
        f"[ok] {len(result.get('signals', []))} signal(s) published, "
        f"{len(result.get('rejected_low_rr', []))} rejected on reward/risk, "
        f"{len(new_trades)} new trade(s) opened, "
        f"{len(open_trades) + len(new_trades)} now being watched.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
