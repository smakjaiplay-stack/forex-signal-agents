"""
Agent 6 - Signal QC Reviewer (Rule-Based, No AI API required)
==============================================================
Sits between Agent 3 (Signal Synthesizer) and Agent 4 (LINE Notifier) and
sanity-checks every signal before it can reach your phone. The point isn't to
second-guess the indicators — it's to catch the cases where Agent 3 produced a
technically-valid-looking card that nobody should actually trade: a signal
fired minutes before a high-impact release, a 55% coin flip dressed up as a
Buy, a target that isn't worth the stop, a signal pointing the opposite way to
a trade you already have on, or a whole run built on data that went stale
while the job sat in GitHub's queue.

Pipeline:
    Agent 3 -> signal.json -> [Agent 6] -> signal_reviewed.json -> Agent 4

Validation rules (all rule-based, no external API):
    1. News conflict  - high-impact event for either currency in the pair
                        landing within the next N minutes (default 120) and
                        still unreleased, while the action is Buy/Sell
                        -> "high_risk", possibility % reduced.
    2. Confidence     - possibility % (after rule 1) below the threshold
                        (default 60) -> "low_confidence", action forced to
                        "Wait".
    3. Reward/risk    - (TP1 - entry) / (entry - SL) below the minimum
                        (default 1.5) -> "poor_rr".
    4. Contradiction  - an open trade on the same pair running the opposite
                        direction -> "conflicting_signal".
    5. Data freshness - news_summary.json or technical_analysis.json older
                        than the limit (default 30 min) -> "stale_data", and
                        the whole run is blocked.

Output: signal_reviewed.json — the same shape as signal.json, with every
signal carrying qc_status / qc_flags / qc_notes / original_action /
final_action, plus a run-level "qc" summary block.

Because Agent 3 writes new trades into open_trade.json *before* QC runs, a
signal that ends up blocked (or downgraded to "Wait") would otherwise leave
Agent 5 babysitting a trade you were never told about. So trades this run
opened for a rejected signal get rolled back out of open_trade.json.

Every flagged or blocked signal is appended to qc_log.jsonl, so you can go
back later and ask whether the QC layer was actually right.

Usage:
    python agent6_qc_reviewer.py
    python agent6_qc_reviewer.py --dry-run
    python agent6_qc_reviewer.py --min-possibility 65 --min-rr 2.0
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

DEFAULT_MIN_POSSIBILITY = 60
DEFAULT_MIN_RR = 1.5
DEFAULT_NEWS_WINDOW_MINUTES = 120
DEFAULT_MAX_DATA_AGE_MINUTES = 30
# What a pending high-impact release costs a signal. Agent 3 already docks 15
# points for "high-impact event somewhere today"; this is the extra penalty
# for one that is actually imminent.
NEWS_RISK_PENALTY = 15

FLAG_HIGH_RISK = "high_risk"
FLAG_LOW_CONFIDENCE = "low_confidence"
FLAG_POOR_RR = "poor_rr"
FLAG_CONFLICTING = "conflicting_signal"
FLAG_STALE_DATA = "stale_data"

STATUS_APPROVED = "approved"
STATUS_DOWNGRADED = "downgraded"
STATUS_BLOCKED = "blocked"

TRADE_ACTIONS = ("Buy", "Sell")
OPPOSITE = {"Buy": "Sell", "Sell": "Buy"}


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        print(f"[warn] could not read {path}: {e}", file=sys.stderr)
        return default


def parse_time(value):
    """Parse an ISO-8601 timestamp into an aware datetime. Naive input is read
    as UTC; anything unparseable comes back as None, so a malformed field
    degrades into "unknown" instead of crashing the run."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def currencies_in_pair(pair):
    pair = str(pair or "").upper()
    return pair[:3], pair[3:6]


# --------------------------------------------------------------------------
# Rule 5 - data freshness
# --------------------------------------------------------------------------
def data_age_minutes(data, now):
    """Minutes since `data["generated_at"]`. None if there's no usable stamp."""
    if not isinstance(data, dict):
        return None
    generated = parse_time(data.get("generated_at"))
    if generated is None:
        return None
    return (now - generated).total_seconds() / 60.0


def check_data_freshness(news_data, tech_data, now, max_age_minutes):
    """Run-level staleness check.

    Returns (is_stale, notes). A missing or unreadable timestamp counts as
    stale: QC that quietly passes when it can't verify freshness is worse than
    no QC at all.
    """
    notes = []
    stale = False
    for label, data in (("news_summary", news_data), ("technical_analysis", tech_data)):
        age = data_age_minutes(data, now)
        if age is None:
            stale = True
            notes.append(f"{label}.json has no readable generated_at timestamp")
        elif age > max_age_minutes:
            stale = True
            notes.append(f"{label}.json is {age:.0f} min old (limit {max_age_minutes} min)")
    return stale, notes


# --------------------------------------------------------------------------
# Rule 1 - imminent high-impact news
# --------------------------------------------------------------------------
def imminent_high_impact_events(news_data, now, window_minutes):
    """Currency -> list of high-impact events due within the window.

    Only events still waiting on their number ("actual" empty) count — once
    the figure is out, the volatility it causes is already in the price.
    """
    events = {}
    if not isinstance(news_data, dict):
        return events
    horizon = now + timedelta(minutes=window_minutes)
    for event in news_data.get("calendar_events_today", []):
        if not isinstance(event, dict):
            continue
        if str(event.get("impact", "")).lower() != "high":
            continue
        if event.get("actual"):
            continue
        event_time = parse_time(event.get("time"))
        if event_time is None or not (now <= event_time <= horizon):
            continue
        currency = str(event.get("currency", "")).upper()
        if not currency:
            continue
        events.setdefault(currency, []).append({
            "currency": currency,
            "title": event.get("title"),
            "time": event.get("time"),
            "minutes_away": round((event_time - now).total_seconds() / 60.0, 1),
        })
    return events


# --------------------------------------------------------------------------
# Rule 3 - reward/risk measured off TP1
# --------------------------------------------------------------------------
def reward_risk_tp1(signal):
    """Reward (entry -> TP1) divided by risk (entry -> SL).

    Deliberately measured off TP1, not Agent 3's TP3-based `risk_reward`: TP1
    is the only target most trades actually reach, so it's the honest number
    to gate on.
    """
    entry = signal.get("open_price")
    tp1 = signal.get("take_profit_1")
    stop = signal.get("stop_loss")
    if entry is None or tp1 is None or stop is None:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    return abs(tp1 - entry) / risk


# --------------------------------------------------------------------------
# Rule 4 - contradiction with an already-open trade
# --------------------------------------------------------------------------
def load_open_trades(path):
    """Open trades from Agent 5's state file, tolerating the legacy
    single-dict format."""
    data = load_json(path, default=[])
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [
        t for t in data
        if isinstance(t, dict) and t.get("action") in TRADE_ACTIONS and not t.get("closed")
    ]


def conflicting_open_trades(signal, open_trades):
    """Open trades on the same pair pointing the other way."""
    action = signal.get("action")
    if action not in TRADE_ACTIONS:
        return []
    opposite = OPPOSITE[action]
    return [
        t for t in open_trades
        if t.get("pair") == signal.get("pair") and t.get("action") == opposite
    ]


# --------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------
def review_signal(signal, *, now, imminent_events, open_trades, stale, stale_notes,
                  min_possibility=DEFAULT_MIN_POSSIBILITY, min_rr=DEFAULT_MIN_RR,
                  news_penalty=NEWS_RISK_PENALTY):
    """Run every rule over one signal and return a reviewed copy of it.

    The input dict is not mutated.
    """
    reviewed = dict(signal)
    original_action = signal.get("action")
    original_possibility = signal.get("possibility_percent")
    possibility = original_possibility if isinstance(original_possibility, (int, float)) else 0
    final_action = original_action
    flags = []
    notes = []

    is_trade = original_action in TRADE_ACTIONS

    # Rule 1 - a high-impact release is about to land on this pair.
    base, quote = currencies_in_pair(signal.get("pair"))
    pair_events = imminent_events.get(base, []) + imminent_events.get(quote, [])
    if is_trade and pair_events:
        flags.append(FLAG_HIGH_RISK)
        possibility = max(0, possibility - news_penalty)
        soonest = min(pair_events, key=lambda e: e["minutes_away"])
        titles = ", ".join(f"{e['currency']} {e['title']}" for e in pair_events)
        # The before/after possibility numbers live in their own fields, so the
        # note only has to say what's coming and when.
        notes.append(f"high-impact news in {soonest['minutes_away']:.0f} min ({titles})")

    # Rule 2 - a coin flip is not a trade.
    if is_trade and possibility < min_possibility:
        flags.append(FLAG_LOW_CONFIDENCE)
        final_action = "Wait"
        notes.append(
            f"possibility {possibility}% below the {min_possibility}% threshold — action forced to Wait"
        )

    # Rule 3 - is the target worth the stop?
    rr = reward_risk_tp1(signal)
    if is_trade:
        if rr is None:
            flags.append(FLAG_POOR_RR)
            notes.append("reward/risk to TP1 could not be computed (missing entry/TP1/SL)")
        elif min_rr > 0 and rr < min_rr:
            flags.append(FLAG_POOR_RR)
            notes.append(f"reward/risk to TP1 is {rr:.2f}, below {min_rr}")

    # Rule 4 - already holding the other side of this pair.
    conflicts = conflicting_open_trades(signal, open_trades)
    if conflicts:
        flags.append(FLAG_CONFLICTING)
        held = ", ".join(f"{t.get('action')} @ {t.get('entry_price')}" for t in conflicts)
        notes.append(f"contradicts an open trade on {signal.get('pair')} ({held})")

    # Rule 5 - the whole run is built on stale inputs.
    if stale:
        flags.append(FLAG_STALE_DATA)
        notes.extend(stale_notes)
        qc_status = STATUS_BLOCKED
    elif flags or final_action != original_action:
        qc_status = STATUS_DOWNGRADED
    else:
        qc_status = STATUS_APPROVED

    reviewed["qc_status"] = qc_status
    reviewed["qc_flags"] = flags
    reviewed["qc_notes"] = "; ".join(notes) if notes else "passed all QC checks"
    reviewed["original_action"] = original_action
    reviewed["final_action"] = final_action
    reviewed["qc_risk_reward_tp1"] = round(rr, 2) if rr is not None else None
    reviewed["qc_reviewed_at"] = now.isoformat()

    # Downstream consumers (Agent 4's card renderer, Agent 5's state) read
    # `action` and `possibility_percent`, so the QC verdict has to land on
    # those fields too — with the pre-QC values preserved beside them.
    if possibility != original_possibility:
        reviewed["original_possibility_percent"] = original_possibility
        reviewed["possibility_percent"] = possibility
    if final_action != original_action:
        reviewed["action"] = final_action
        reviewed["status"] = "No Trade"

    if qc_status == STATUS_BLOCKED:
        reviewed["new"] = False
        reviewed["skip_reason"] = f"blocked by QC ({', '.join(flags)})"

    return reviewed


def review(signal_data, news_data, tech_data, open_trades, *, now=None,
           min_possibility=DEFAULT_MIN_POSSIBILITY, min_rr=DEFAULT_MIN_RR,
           news_window_minutes=DEFAULT_NEWS_WINDOW_MINUTES,
           max_data_age_minutes=DEFAULT_MAX_DATA_AGE_MINUTES,
           news_penalty=NEWS_RISK_PENALTY):
    """Review a whole signal.json payload. Returns the signal_reviewed.json dict."""
    now = now or datetime.now(timezone.utc)

    stale, stale_notes = check_data_freshness(news_data, tech_data, now, max_data_age_minutes)
    imminent_events = imminent_high_impact_events(news_data, now, news_window_minutes)

    signals = signal_data.get("signals")
    if signals is None:
        # Legacy single-signal signal.json
        signals = [signal_data["signal"]] if signal_data.get("signal") else []

    reviewed_signals = [
        review_signal(
            s, now=now, imminent_events=imminent_events, open_trades=open_trades,
            stale=stale, stale_notes=stale_notes, min_possibility=min_possibility,
            min_rr=min_rr, news_penalty=news_penalty,
        )
        for s in signals
    ]

    counts = {STATUS_APPROVED: 0, STATUS_DOWNGRADED: 0, STATUS_BLOCKED: 0}
    for s in reviewed_signals:
        counts[s["qc_status"]] += 1

    if reviewed_signals and counts[STATUS_BLOCKED] == len(reviewed_signals):
        run_status = STATUS_BLOCKED
    elif counts[STATUS_DOWNGRADED] or counts[STATUS_BLOCKED]:
        run_status = STATUS_DOWNGRADED
    else:
        run_status = STATUS_APPROVED

    output = dict(signal_data)
    output["signals"] = reviewed_signals
    # Keep the legacy single-signal key in sync, so anything still reading it
    # sees the reviewed card and not the pre-QC one.
    output["signal"] = reviewed_signals[0] if reviewed_signals else None
    output["qc"] = {
        "reviewed_at": now.isoformat(),
        "qc_status": run_status,
        "counts": counts,
        "stale_data": stale,
        "stale_notes": stale_notes,
        "imminent_high_impact_currencies": sorted(imminent_events.keys()),
        "thresholds": {
            "min_possibility_percent": min_possibility,
            "min_risk_reward_tp1": min_rr,
            "news_window_minutes": news_window_minutes,
            "max_data_age_minutes": max_data_age_minutes,
        },
    }
    # Convenience mirror, so a consumer can check a single top-level key.
    output["qc_status"] = run_status
    return output


# --------------------------------------------------------------------------
# Side effects: QC log + open-trade rollback
# --------------------------------------------------------------------------
def log_records(reviewed_output, log_all=False):
    """One record per signal worth remembering, for post-mortem analysis."""
    records = []
    reviewed_at = reviewed_output.get("qc", {}).get("reviewed_at")
    for s in reviewed_output.get("signals", []):
        if not log_all and s.get("qc_status") == STATUS_APPROVED:
            continue
        records.append({
            "reviewed_at": reviewed_at,
            "signal_generated_at": reviewed_output.get("generated_at"),
            "pair": s.get("pair"),
            "qc_status": s.get("qc_status"),
            "qc_flags": s.get("qc_flags", []),
            "qc_notes": s.get("qc_notes"),
            "original_action": s.get("original_action"),
            "final_action": s.get("final_action"),
            "original_possibility_percent": s.get("original_possibility_percent",
                                                  s.get("possibility_percent")),
            "possibility_percent": s.get("possibility_percent"),
            "entry": s.get("open_price"),
            "stop_loss": s.get("stop_loss"),
            "take_profit_1": s.get("take_profit_1"),
            "risk_reward_tp1": s.get("qc_risk_reward_tp1"),
            "risk_reward_tp3": s.get("risk_reward"),
            "was_new": s.get("new"),
        })
    return records


def append_qc_log(path, records):
    if not records:
        return 0
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def rollback_open_trades(open_trades, reviewed_output):
    """Drop the trades this run opened for signals QC then rejected.

    Agent 3 writes open_trade.json before QC gets a look, so without this a
    blocked signal still leaves Agent 5 tracking — and alerting on — a trade
    that was never sent. Only trades stamped with *this* run's generated_at
    are touched; anything already running is left alone.
    """
    generated_at = reviewed_output.get("generated_at")
    if not generated_at:
        return open_trades, []

    rejected_pairs = {
        s.get("pair")
        for s in reviewed_output.get("signals", [])
        if s.get("qc_status") == STATUS_BLOCKED or s.get("final_action") not in TRADE_ACTIONS
    }
    if not rejected_pairs:
        return open_trades, []

    kept, dropped = [], []
    for trade in open_trades:
        if trade.get("pair") in rejected_pairs and trade.get("opened_at") == generated_at:
            dropped.append(trade)
        else:
            kept.append(trade)
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description="Agent 6 - Signal QC Reviewer (rule-based)")
    parser.add_argument("--signal", default="signal.json", help="Path to Agent 3 output")
    parser.add_argument("--news", default="news_summary.json", help="Path to Agent 1 output")
    parser.add_argument("--technical", default="technical_analysis.json", help="Path to Agent 2 output")
    parser.add_argument("--open-trade", default="open_trade.json",
                        help="Agent 5's open-trade state (rejected signals are rolled back out of it)")
    parser.add_argument("--out", default="signal_reviewed.json", help="Reviewed output for Agent 4")
    parser.add_argument("--qc-log", default="qc_log.jsonl", help="Append-only QC audit log")
    parser.add_argument("--min-possibility", type=int, default=DEFAULT_MIN_POSSIBILITY,
                        help="Below this possibility %%, the action is forced to Wait")
    parser.add_argument("--min-rr", type=float, default=DEFAULT_MIN_RR,
                        help="Flag poor_rr below this reward/risk to TP1 (0 disables)")
    parser.add_argument("--news-window-minutes", type=int, default=DEFAULT_NEWS_WINDOW_MINUTES,
                        help="How far ahead a high-impact release counts as imminent")
    parser.add_argument("--max-data-age-minutes", type=int, default=DEFAULT_MAX_DATA_AGE_MINUTES,
                        help="Block the run if the news/technical files are older than this")
    parser.add_argument("--log-all", action="store_true",
                        help="Log approved signals too, not just flagged/blocked ones")
    parser.add_argument("--no-rollback", action="store_true",
                        help="Leave open_trade.json alone even when a signal is rejected")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the verdict without writing any file")
    args = parser.parse_args()

    signal_data = load_json(args.signal)
    if signal_data is None:
        print(f"[error] could not read {args.signal} — nothing to review.", file=sys.stderr)
        sys.exit(1)

    news_data = load_json(args.news)
    tech_data = load_json(args.technical)
    open_trades = load_open_trades(args.open_trade)

    reviewed = review(
        signal_data, news_data, tech_data, open_trades,
        min_possibility=args.min_possibility,
        min_rr=args.min_rr,
        news_window_minutes=args.news_window_minutes,
        max_data_age_minutes=args.max_data_age_minutes,
    )

    records = log_records(reviewed, log_all=args.log_all)
    kept, dropped = ([], []) if args.no_rollback else rollback_open_trades(open_trades, reviewed)

    markers = {STATUS_APPROVED: "[ok]", STATUS_DOWNGRADED: "[warn]", STATUS_BLOCKED: "[block]"}
    for s in reviewed["signals"]:
        print(f"{markers[s['qc_status']]} {s.get('pair')} {s.get('original_action')} -> "
              f"{s.get('final_action')}: {s['qc_status']} {s.get('qc_flags')} — {s.get('qc_notes')}",
              file=sys.stderr)

    if args.dry_run:
        print(json.dumps(reviewed, ensure_ascii=False, indent=2))
        print(f"[dry-run] nothing written ({len(records)} log record(s) and "
              f"{len(dropped)} trade rollback(s) skipped).", file=sys.stderr)
        return

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(reviewed, f, ensure_ascii=False, indent=2)

    logged = append_qc_log(args.qc_log, records)

    if dropped:
        with open(args.open_trade, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        for trade in dropped:
            print(f"[rollback] removed {trade.get('pair')} {trade.get('action')} from "
                  f"{args.open_trade} — QC rejected the signal.", file=sys.stderr)

    counts = reviewed["qc"]["counts"]
    print(
        f"[ok] QC {reviewed['qc_status']}: {counts[STATUS_APPROVED]} approved, "
        f"{counts[STATUS_DOWNGRADED]} downgraded, {counts[STATUS_BLOCKED]} blocked; "
        f"{logged} record(s) appended to {args.qc_log}.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
