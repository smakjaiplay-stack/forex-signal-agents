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
       styled like a typical "forex signal" app card. The stop is ATR-scaled
       and the targets are multiples of that risk (see build_trade_levels for
       what the previous range-edge rule turned out to be doing), so a pair
       with no usable ATR yields no publishable trade at all.

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

import scoring

DEFAULT_TOP_N = 3
DEFAULT_MAX_OPEN = 3
DEFAULT_COOLDOWN_HOURS = 6
# Float, not int: Agent 2's score is continuous now. The value itself lives in
# scoring.py next to the measured score distribution that justifies it, so
# Agent 2, Agent 3 and backtest.py cannot each drift to a different cut.
DEFAULT_MIN_SCORE = scoring.DEFAULT_MIN_SCORE
# The stop sits this many ATRs from entry and the targets are multiples of that
# risk. Both live in scoring.py, with the measurement that justifies them and
# the account of what the previous range-edge rule actually did. Agent 6 reads
# the same constants, which is the point: it used to gate on a 1.5 that meant
# something different from Agent 3's 1.5.
DEFAULT_ATR_MULT = scoring.SL_ATR_MULT
DEFAULT_TP_R_MULTS = scoring.TP_R_MULTS
DEFAULT_MIN_RR = scoring.MIN_RR_TP1


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
    """How finely to round this pair's levels.

    The constraint is not cosmetic. Levels are rounded AFTER being placed at
    exact multiples of the risk, so the rounding error has to stay small
    against that risk or the published ratio drifts off the 1:1 it claims -
    and Agent 6 reads that drift as broken geometry.

    Silver is the case that proves it: at ~28 with a ~0.11 ATR the risk is
    about 0.165, and rounding that to 2dp moved the reward/risk to 1.06,
    outside agent6.RR_TOLERANCE. Gold at ~2400 has a risk near 5.0, where the
    same 2dp is nowhere near the tolerance. Same asset class, different answer,
    which is why this is per-pair rather than per-category.
    """
    pair = pair.upper()
    if pair == "XAUUSD":
        return 2
    if pair == "XAGUSD":
        return 3
    if pair.endswith("JPY"):
        return 3
    return 5


CALIBRATION_PATH = "score_calibration.json"


def load_calibration(path=CALIBRATION_PATH):
    """What validate.py measured, or None when no backtest has been run.

    Returns a dict with:
        points              [(score_midpoint, win_rate_percent), ...] sorted
        breakeven_win_rate  win rate at which the payoff cancels out
        edge_significant    is measured expectancy distinguishable from zero
        expectancy_r / expectancy_t / payoff_ratio   for the QC note

    The file only exists once a backtest has actually been measured, which is
    the point: before that, the system has no basis for claiming any
    particular probability, and says so via possibility_calibrated: false.

    Everything past `points` is forwarded untouched to Agent 6. Agent 3 does
    not gate on any of it - it measures, Agent 6 decides - but the numbers
    have to ride along on the signal, because Agent 6 must not have to re-open
    and re-interpret this file to know what the score means.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    buckets = data.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        return None
    try:
        points = sorted((float(b["score_mid"]), float(b["win_rate"])) for b in buckets)
    except (KeyError, TypeError, ValueError):
        return None
    if not points:
        return None
    return {
        "points": points,
        # Absent in a calibration file written before this field existed. None
        # (not 0) so Agent 6 can tell "no payoff measured" from "breaks even at
        # 0%", and fall back to its own floor instead of waving everything past.
        "breakeven_win_rate": _opt_float(data.get("breakeven_win_rate")),
        "payoff_ratio": _opt_float(data.get("payoff_ratio")),
        "expectancy_r": _opt_float(data.get("expectancy_r")),
        "expectancy_t": _opt_float(data.get("expectancy_t")),
        # Missing counts as false: a calibration that never measured an edge
        # has not demonstrated one.
        "edge_significant": bool(data.get("edge_significant", False)),
        "trades": data.get("trades"),
    }


def _opt_float(x):
    """float(x), or None for anything that is not a real number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def possibility_percent(abs_score: float, risky: bool, calibration=None) -> int:
    """Confidence percentage for the trade card (30-95%).

    THE OLD VERSION WAS A CONSTANT. It computed 50 + |score| * 10, and since
    every published signal had |score| == 2 (see scoring.py), every card ever
    sent to LINE said exactly 70%. It looked like a measured probability and
    was a decoration.

    With a calibration file present this interpolates the win rate actually
    observed at that score level, so the number means something. Without one
    it falls back to a linear map and the signal is stamped
    possibility_calibrated: false - an admission, not a measurement.
    """
    if calibration:
        # Piecewise-linear between measured bucket midpoints, clamped at the ends.
        points = calibration["points"] if isinstance(calibration, dict) else calibration
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if abs_score <= xs[0]:
            base = ys[0]
        elif abs_score >= xs[-1]:
            base = ys[-1]
        else:
            i = next(i for i in range(len(xs) - 1) if xs[i] <= abs_score <= xs[i + 1])
            span = xs[i + 1] - xs[i]
            frac = 0.0 if span <= 0 else (abs_score - xs[i]) / span
            base = ys[i] + frac * (ys[i + 1] - ys[i])
    else:
        # score 0 -> 50%, 3 -> 80%. Same slope as before, now continuous.
        base = 50 + abs_score * 10

    if risky:
        base -= 15
    return int(max(30, min(round(base), 95)))


def build_trade_levels(action: str, entry: float, ndigits: int, atr: float = None,
                       atr_mult: float = DEFAULT_ATR_MULT,
                       tp_r_mults=DEFAULT_TP_R_MULTS):
    """Stop and targets for one trade, or None if they cannot be built honestly.

    The risk is `atr_mult` ATRs, and TP1/TP2/TP3 sit at 1R/2R/3R from entry.
    That makes the reward/risk on the card a stated fact rather than a
    by-product of which branch of a guard happened to fire.

    THE RULE THIS REPLACED, AND WHY
    -------------------------------
    Targets came from the 50-bar range edge and the stop from the opposite
    edge, with a fixed-percentage fallback "if support/resistance don't bracket
    the entry price sensibly (can happen with thin/odd data)". It was not thin
    or odd data. The bounds were rolling extremes of the CLOSE and the entry is
    the latest close, so every setup making new ground - which is what a high
    score selects - failed the guard and took the fallback. Measured over 502
    entries: the guard failed 42% of the time, and once the reward/risk filter
    had finished, 98.6% of all trades taken came out of the fallback. The
    published strategy was a flat 1.5%-of-price target wearing a technical
    rule's clothes. Full numbers in scoring.py's TRADE GEOMETRY section.

    So there is no fallback here. Without an ATR there is no volatility
    estimate, without a volatility estimate there is no honest stop distance,
    and a signal with no honest stop distance does not get published: this
    returns None and synthesize() records it in `rejected_no_levels`.
    """
    if action not in ("Buy", "Sell"):
        return None
    # atr_mult <= 0 used to mean "no cap" back when the stop came from
    # structure. There is no structure fallback to uncap onto any more, so it
    # would mean a stop sitting on the entry price - a trade that stops out on
    # the spread.
    if not atr or atr <= 0 or atr != atr or atr_mult <= 0:
        return None

    risk = atr * max(atr_mult, scoring.MIN_SL_ATR_MULT)
    sign = 1.0 if action == "Buy" else -1.0
    tp1, tp2, tp3 = (entry + sign * risk * m for m in tp_r_mults)

    return {
        "take_profit_1": round(tp1, ndigits),
        "take_profit_2": round(tp2, ndigits),
        "take_profit_3": round(tp3, ndigits),
        "stop_loss": round(entry - sign * risk, ndigits),
        # Reported so a card, a backtest row and a QC verdict can all be
        # checked against the volatility they were built from.
        "risk_atr_mult": round(risk / atr, 3),
    }


def risk_reward(entry: float, target: float, sl: float):
    """Reward (entry->target) divided by risk (entry->SL). None if risk is zero.

    With R-multiple targets this is a constant by construction - 1.0 to TP1,
    3.0 to TP3 - which is exactly why it is no longer a filter. It is computed
    and published so that a card claiming 3:1 can be checked against its own
    numbers, and so Agent 6 blocks the run if the geometry ever comes apart
    again the way it did before.
    """
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    return abs(target - entry) / risk


def build_signal(candidate, time_frame, atr_mult=DEFAULT_ATR_MULT, calibration=None):
    """Turn one ranked candidate into a full trade card."""
    # Bands, not equality: the score is a float now, so the old
    # `abs_score == 2` test for "medium" would essentially never fire.
    abs_score = candidate["abs_score"]
    if abs_score < 2.0:
        confidence = "low"
    elif abs_score < 2.5:
        confidence = "medium"
    else:
        confidence = "high"

    if candidate["risky_pending_news"]:
        confidence = "low"

    action = {"bullish": "Buy", "bearish": "Sell", "neutral": "Wait"}[candidate["bias"]]
    ndigits = decimals_for_pair(candidate["pair"])
    levels = build_trade_levels(
        action, candidate["last_close"], ndigits,
        atr=candidate.get("atr14"), atr_mult=atr_mult,
    )
    now_utc = datetime.now(timezone.utc)

    signal = {
        "pair": candidate["pair"],
        "action": action,
        "direction": candidate["bias"],
        "confidence": confidence,
        "possibility_percent": possibility_percent(abs_score, candidate["risky_pending_news"],
                                                   calibration=calibration),
        # False means the percentage above is a linear guess, not a measured
        # win rate. Run validate.py to produce score_calibration.json.
        "possibility_calibrated": bool(calibration),
        # What the measurement says about the strategy as a whole, forwarded
        # for Agent 6's gate. possibility_percent alone cannot answer "is this
        # trade worth taking": a 44% win rate is a loss at a 1.1:1 payoff and a
        # fortune at 3:1, so the payoff has to travel with the percentage.
        "calibration": {
            "breakeven_win_rate": calibration.get("breakeven_win_rate"),
            "payoff_ratio": calibration.get("payoff_ratio"),
            "expectancy_r": calibration.get("expectancy_r"),
            "expectancy_t": calibration.get("expectancy_t"),
            "edge_significant": calibration.get("edge_significant", False),
            "trades": calibration.get("trades"),
        } if isinstance(calibration, dict) else None,
        "score": round(candidate["score"], 4),
        # Which components drove this read, carried from Agent 2 so the live
        # signals can be IC-tested the same way the backtest is.
        "components": candidate.get("components") or {},
        "status": "Active" if action != "Wait" else "No Trade",
        "opening_time": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "last_update": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "open_price": round(candidate["last_close"], ndigits),
        "time_frame": time_frame,
        # How stale the candle behind open_price is, so the LINE card can say
        # "priced off a 1h bar that closed 40 min ago" instead of hiding it.
        # This matters more than it used to: the interval moved from 15m to 1h
        # (see scoring.DEFAULT_INTERVAL), so this number can now reach ~60.
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
        entry = signal["open_price"]
        # Both ratios travel with the card. TP1 is the one that gets gated on
        # and the one Agent 4 leads with, because it is the target most trades
        # actually reach; TP3 is the stretch, and quoting only that number is
        # how a card came to read "R:R 28.4".
        rr1 = risk_reward(entry, levels["take_profit_1"], levels["stop_loss"])
        rr3 = risk_reward(entry, levels["take_profit_3"], levels["stop_loss"])
        signal["risk_reward_tp1"] = round(rr1, 2) if rr1 is not None else None
        signal["risk_reward"] = round(rr3, 2) if rr3 is not None else None
        signal["atr14"] = candidate.get("atr14")
    return signal


def synthesize(news_data, tech_data, top_n=DEFAULT_TOP_N, min_score=DEFAULT_MIN_SCORE,
               min_rr=DEFAULT_MIN_RR, atr_mult=DEFAULT_ATR_MULT, calibration=None):
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
            "components": result.get("components"),
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

    # What gets dropped here has changed, and the change is the point. Under the
    # old geometry this was a reward/risk filter, and because reward/risk was an
    # accident of which branch of build_trade_levels fired, the filter was in
    # practice selecting the broken branch: 212 of 212 fixed-percentage setups
    # passed it against 3 of 290 structure-based ones. Reward/risk is a stated
    # constant now, so it cannot select anything.
    #
    # The gate that remains is the one that actually matters: a signal with no
    # usable ATR has no honest stop distance, so it is not published. `min_rr`
    # survives as a tripwire on the constructed ratio - if the geometry ever
    # drifts away from what scoring.py says it should be, that is a bug, and
    # these rejections are where it surfaces.
    signals, rejected = [], []
    for candidate in tradable:
        signal = build_signal(candidate, time_frame, atr_mult=atr_mult, calibration=calibration)
        rr = signal.get("risk_reward_tp1")
        if signal.get("stop_loss") is None:
            rejected.append({
                "pair": signal["pair"],
                "action": signal["action"],
                "risk_reward_tp1": None,
                "reason": f"no usable ATR for {signal['pair']} — cannot size a stop",
            })
            continue
        if min_rr > 0 and (rr is None or rr < min_rr):
            rejected.append({
                "pair": signal["pair"],
                "action": signal["action"],
                "risk_reward_tp1": rr,
                "reason": f"reward/risk to TP1 {rr} below minimum {min_rr} — "
                          f"the levels are not what scoring.TP_R_MULTS specifies",
            })
            continue
        signals.append(signal)
        if len(signals) >= top_n:
            break

    if not signals and not tradable:
        # Nothing cleared the score bar — surface the strongest read anyway so
        # the run still reports what the market looks like.
        signals = [build_signal(pool_sorted[0], time_frame, atr_mult=atr_mult,
                                calibration=calibration)]

    return {
        "signals": signals,
        # Kept so anything still reading the old single-signal shape works.
        "signal": signals[0] if signals else None,
        "rejected": rejected,
        # Kept under its old name so an existing reader (and the CI commit
        # step) does not silently start seeing nothing.
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
        # Carried so Agent 5 can stamp them onto the closed-trade record, which
        # is what lets validate.py run the decile test and the per-component IC
        # against LIVE outcomes. Without these two fields the forward samples
        # can only ever answer "did it make money", never "which indicator was
        # carrying it" - and the second question is the whole point of
        # collecting them.
        "score": signal.get("score"),
        "components": signal.get("components") or {},
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
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                         help="Minimum |technical score| for a pair to be publishable "
                              "(score is continuous in -3..+3 - see scoring.py)")
    parser.add_argument("--calibration", default=CALIBRATION_PATH,
                         help="Score-to-win-rate table from validate.py; without it "
                              "possibility_percent is an uncalibrated guess")
    parser.add_argument("--min-rr", type=float, default=DEFAULT_MIN_RR,
                         help="Tripwire on the constructed reward/risk to TP1 (0 disables). "
                              "Not a trade filter any more — the ratio is fixed by "
                              "scoring.TP_R_MULTS, so this only fires if the geometry breaks")
    parser.add_argument("--atr-mult", type=float, default=DEFAULT_ATR_MULT,
                         help="Place the stop-loss this many ATRs from entry")
    parser.add_argument("--cooldown-hours", type=float, default=DEFAULT_COOLDOWN_HOURS,
                         help="Don't re-enter a pair this soon after its last trade closed (0 disables)")
    args = parser.parse_args()

    news_data = load_json(args.news)
    tech_data = load_json(args.technical)

    if tech_data is None:
        print("[error] technical analysis data is required to synthesize a signal", file=sys.stderr)
        sys.exit(1)

    calibration = load_calibration(args.calibration)
    if calibration is None:
        print(f"[warn] no calibration at {args.calibration} - possibility_percent will be "
              f"a linear guess, not a measured win rate. Run validate.py to build one.",
              file=sys.stderr)
    result = synthesize(news_data, tech_data, top_n=args.top, min_score=args.min_score,
                        min_rr=args.min_rr, atr_mult=args.atr_mult, calibration=calibration)
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

    for r in result.get("rejected", []):
        print(f"[skip] {r['pair']} {r['action']}: {r['reason']}", file=sys.stderr)

    print(
        f"[ok] {len(result.get('signals', []))} signal(s) published, "
        f"{len(result.get('rejected', []))} rejected, "
        f"{len(new_trades)} new trade(s) opened, "
        f"{len(open_trades) + len(new_trades)} now being watched.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
