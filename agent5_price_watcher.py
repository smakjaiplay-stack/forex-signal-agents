"""
Agent 5 - Price Watcher
=========================
Watches every open trade in open_trade.json (written by Agent 3, which can
publish several pairs per run, then pruned by Agent 6's QC layer of anything
it blocked before it reached LINE) against live intraday price data, applies the
same TP1/TP2/TP3 trailing stop-loss rule that backtest.py simulates
(breakeven after TP1, TP1-level after TP2, close at TP3), and sends a LINE
alert plus updates open_trade.json whenever a level is crossed.

This runs far more often than the signal pipeline - every 15 min during
forex trading hours - so a trade actually gets watched between pipeline
runs instead of only being evaluated once when it's already stale.

Each time a trade closes (TP3, SL, or a ~24h timeout - matching the daily
reset the live system already does), the outcome is appended to
trades_log.jsonl so live performance can be tracked over time instead of
being lost every time Agent 3 overwrites open_trade.json the next day.

Usage:
    python agent5_price_watcher.py
    python agent5_price_watcher.py --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from agent2_technical_analyzer import DEFAULT_PAIRS
from agent4_line_notifier import send_broadcast

MAX_HOLD_HOURS = 24

# Yahoo serves 5m bars for the last 60 days, so the window below is a choice
# about how much to ask for, not a hard limit. "2d" was too little: this job
# does not run at weekends (the cron is Mon-Fri plus the Sunday Asia open), so
# a trade opened late on Friday came back on Monday to a window that no longer
# reached the bars it had missed. The trade would then time out at whatever
# price happened to be current, recording an R-multiple that had nothing to do
# with the SL it had actually traded through.
MIN_FETCH_DAYS = 2
MAX_FETCH_DAYS = 30
# A trade opens between bars, so the first bar available after `last_checked`
# is normally a few minutes later and that is not a gap. One watcher cycle of
# slack keeps the warning meaning "bars are actually missing".
COVERAGE_GAP_TOLERANCE_MINUTES = 15


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def append_trade_log(record: dict, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch_window_days(last_checked, now=None):
    """How much 5m history to pull, given how long this trade went unwatched.

    Always at least MIN_FETCH_DAYS, and enough to cover the gap since the last
    check plus a day of margin. A weekend, a failed run or a paused schedule
    all produce the same gap, and all of them used to silently fall outside a
    fixed 2-day window.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    if last_checked is None:
        return MIN_FETCH_DAYS
    gap_days = (now - last_checked).total_seconds() / 86400.0
    return int(max(MIN_FETCH_DAYS, min(MAX_FETCH_DAYS, gap_days + 1)))


def fetch_recent_bars(symbol: str, last_checked=None):
    """Fine-grained (5m) bars covering everything since `last_checked`.

    5m rather than hourly so a watcher running every ~15 min catches the
    intrabar wicks that hit a stop between runs instead of only seeing closes.
    """
    period = f"{fetch_window_days(last_checked)}d"
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="5m", auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def notify(token, text, dry_run):
    print("--- Alert ---")
    print(text)
    print("-------------")
    if dry_run:
        print("[dry-run] not sent.")
        return
    if not token:
        print("[warn] LINE_CHANNEL_ACCESS_TOKEN not set, skipping send.", file=sys.stderr)
        return
    resp = send_broadcast(token, text)
    if resp.status_code != 200:
        print(f"[error] LINE API returned {resp.status_code}: {resp.text}", file=sys.stderr)


def watch(open_trade: dict, token: str, dry_run: bool, trades_log_path: str):
    pair = open_trade["pair"]
    action = open_trade["action"]
    symbol = DEFAULT_PAIRS.get(pair)
    if not symbol:
        print(f"[error] unknown pair '{pair}', cannot fetch price", file=sys.stderr)
        return open_trade

    last_checked = pd.Timestamp(open_trade.get("last_checked_at", open_trade["opened_at"]))
    if last_checked.tzinfo is None:
        last_checked = last_checked.tz_localize("UTC")

    df = fetch_recent_bars(symbol, last_checked=last_checked)
    if df is None or df.empty:
        print(f"[warn] no price data for {pair} ({symbol}) right now, skipping this run", file=sys.stderr)
        return open_trade

    # If the feed still does not reach back to the last check, the bars that
    # decided this trade are gone and any outcome derived from what remains is
    # a guess. Say so on the record rather than logging a confident number.
    missing_minutes = (df.index.min() - last_checked).total_seconds() / 60
    if missing_minutes > COVERAGE_GAP_TOLERANCE_MINUTES:
        missing_hours = missing_minutes / 60
        print(f"[warn] {pair}: price history starts {missing_hours:.1f}h after the last check — "
              f"bars in that gap were never seen", file=sys.stderr)
        open_trade["coverage_gap_hours"] = round(missing_hours, 2)

    entry = open_trade["entry_price"]
    tp1, tp2, tp3 = open_trade["take_profit_1"], open_trade["take_profit_2"], open_trade["take_profit_3"]
    sl0 = open_trade["stop_loss"]
    current_sl = open_trade["current_sl"]
    alerts = open_trade["alerts_sent"]
    risk = abs(entry - sl0)

    new_bars = df[df.index > last_checked]

    def close_trade(outcome, exit_price):
        r_multiple = (exit_price - entry) / risk if action == "Buy" else (entry - exit_price) / risk
        open_trade["closed"] = True
        open_trade["outcome"] = outcome
        open_trade["exit_price"] = round(float(exit_price), 5)
        open_trade["r_multiple"] = round(r_multiple, 3)
        open_trade["closed_at"] = datetime.now(timezone.utc).isoformat()
        append_trade_log({
            "pair": pair, "action": action, "entry_price": entry,
            "outcome": outcome, "exit_price": open_trade["exit_price"],
            "r_multiple": open_trade["r_multiple"],
            "tp1": alerts["tp1"], "tp2": alerts["tp2"], "tp3": alerts["tp3"],
            "opened_at": open_trade["opened_at"], "closed_at": open_trade["closed_at"],
            # The score that triggered the trade and the components behind it.
            # validate.py needs both to measure a live information coefficient;
            # trades logged before this existed simply carry None and are
            # skipped by the component report rather than corrupting it.
            "score": open_trade.get("score"),
            "components": open_trade.get("components") or {},
            # Non-null means some of this trade's price history was never
            # seen, so validate.py's reader can tell a measured outcome from
            # a reconstructed one instead of averaging them together.
            "coverage_gap_hours": open_trade.get("coverage_gap_hours"),
        }, trades_log_path)
        notify(token,
               f"{'🟢' if r_multiple > 0 else '🔴' if r_multiple < 0 else '⚪'} {pair} {action} closed - "
               f"{outcome} @ {open_trade['exit_price']} ({r_multiple:+.2f}R)",
               dry_run)

    for ts, bar in new_bars.iterrows():
        if open_trade["closed"]:
            break
        high, low = float(bar["High"]), float(bar["Low"])

        adverse = low if action == "Buy" else high
        sl_hit = (action == "Buy" and adverse <= current_sl) or (action == "Sell" and adverse >= current_sl)
        if sl_hit:
            close_trade("SL", current_sl)
            break

        favorable = high if action == "Buy" else low
        if not alerts["tp1"] and ((action == "Buy" and favorable >= tp1) or (action == "Sell" and favorable <= tp1)):
            alerts["tp1"] = True
            current_sl = entry
            notify(token, f"🎯 {pair} {action} hit TP1 ({tp1}) - SL moved to breakeven ({entry})", dry_run)
        if alerts["tp1"] and not alerts["tp2"] and ((action == "Buy" and favorable >= tp2) or (action == "Sell" and favorable <= tp2)):
            alerts["tp2"] = True
            current_sl = tp1
            notify(token, f"🎯 {pair} {action} hit TP2 ({tp2}) - SL moved to TP1 ({tp1})", dry_run)
        if alerts["tp2"] and not alerts["tp3"] and ((action == "Buy" and favorable >= tp3) or (action == "Sell" and favorable <= tp3)):
            alerts["tp3"] = True
            close_trade("TP3", tp3)
            break

    if not open_trade["closed"]:
        opened_at = pd.Timestamp(open_trade["opened_at"])
        if opened_at.tzinfo is None:
            opened_at = opened_at.tz_localize("UTC")
        hours_open = (pd.Timestamp.now(tz="UTC") - opened_at).total_seconds() / 3600
        if hours_open >= MAX_HOLD_HOURS:
            close_trade("TIMEOUT", float(df["Close"].iloc[-1]))

    open_trade["current_sl"] = current_sl
    open_trade["alerts_sent"] = alerts
    if not open_trade["closed"]:
        open_trade["last_checked_at"] = datetime.now(timezone.utc).isoformat()
    return open_trade


def load_open_trades(path):
    """Agent 3 writes a list of trades now (it can publish several pairs per
    run). Older state files hold a single dict — accept both."""
    data = load_json(path)
    if data is None:
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    return [t for t in data if isinstance(t, dict)]


def main():
    parser = argparse.ArgumentParser(description="Agent 5 - Price Watcher (TP1/TP2/TP3 trailing SL)")
    parser.add_argument("--open-trade", default="open_trade.json", help="Path to Agent 3's open-trade state")
    parser.add_argument("--trades-log", default="trades_log.jsonl", help="Where closed trades get appended")
    parser.add_argument("--dry-run", action="store_true", help="Print alerts instead of sending to LINE")
    args = parser.parse_args()

    trades = load_open_trades(args.open_trade)
    if not trades:
        print(f"[info] no {args.open_trade} found - nothing to watch.")
        return

    watchable = [t for t in trades if not t.get("closed") and t.get("action") in ("Buy", "Sell")]
    if not watchable:
        print("[info] no open Buy/Sell trade to watch right now.")
        return

    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    still_open = []
    for trade in watchable:
        updated = watch(trade, token, args.dry_run, args.trades_log)
        if updated.get("closed"):
            print(f"[ok] {updated['pair']} closed: {updated['outcome']} ({updated['r_multiple']:+.2f}R)")
        else:
            print(f"[ok] {updated['pair']} still open, current_sl={updated['current_sl']}, "
                  f"alerts={updated['alerts_sent']}")
            still_open.append(updated)

    # Closed trades are already in trades_log.jsonl, so they're dropped here
    # rather than re-scanned on every future run.
    with open(args.open_trade, "w", encoding="utf-8") as f:
        json.dump(still_open, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
