"""
Agent 5 - Price Watcher & Risk Manager
=========================================
Runs on an hourly schedule (separate from the daily Agent 1-4 pipeline).
Watches the currently open trade (open_trade.json, written by Agent 3
when it picks a new daily signal) against live price. When price gets
close to or crosses a Take Profit / Stop Loss level, it:
  1. Sends a LINE alert
  2. Adjusts (trails) the Stop Loss to lock in profit / cap loss
  3. Marks that level as already-alerted so it doesn't spam every hour

State lives in open_trade.json, which this script updates and the
workflow commits back to the repo so state survives between runs.

Usage:
    python agent5_price_watcher.py
    python agent5_price_watcher.py --state open_trade.json --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests
import yfinance as yf

from agent2_technical_analyzer import DEFAULT_PAIRS

LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"

# How close price needs to be to a level to count as "approaching" it,
# expressed as a fraction of the level's price (0.0015 = 0.15%).
PROXIMITY_PCT = 0.0015


def load_state(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[error] could not read {path}: {e}", file=sys.stderr)
        return None


def save_state(path, state):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_current_price(pair: str):
    symbol = DEFAULT_PAIRS.get(pair.upper())
    if not symbol:
        return None
    df = yf.download(symbol, period="1d", interval="5m", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if hasattr(df.columns, "get_level_values"):
        try:
            df.columns = [c[0] for c in df.columns]
        except Exception:
            pass
    return float(df["Close"].dropna().iloc[-1])


def near_or_past(action: str, level_name: str, level_price: float, current_price: float):
    """Return True if price has reached or is within PROXIMITY_PCT of the level,
    in the direction that matters for this action (Buy = moving up, Sell = moving down)."""
    threshold = level_price * PROXIMITY_PCT
    if action == "Buy":
        return current_price >= level_price - threshold
    else:  # Sell
        return current_price <= level_price + threshold


def build_alert_text(pair, action, level_name, level_price, current_price, new_sl, note):
    direction = "🟢 Buy" if action == "Buy" else "🔴 Sell"
    lines = [
        f"🔔 {pair} — {level_name} alert",
        f"{direction}",
        f"Current price: {current_price}",
        f"{level_name}: {level_price}",
        "",
        note,
    ]
    if new_sl is not None:
        lines.append(f"Stop Loss moved to: {new_sl}")
    return "\n".join(lines)


def send_line(token: str, text: str, dry_run: bool):
    print("--- LINE alert ---")
    print(text)
    print("------------------")
    if dry_run:
        print("[dry-run] not sent.")
        return
    if not token:
        print("[warn] LINE_CHANNEL_ACCESS_TOKEN not set, skipping send.", file=sys.stderr)
        return
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    payload = {"messages": [{"type": "text", "text": text}]}
    resp = requests.post(LINE_BROADCAST_URL, headers=headers, json=payload, timeout=15)
    if resp.status_code != 200:
        print(f"[error] LINE API returned {resp.status_code}: {resp.text}", file=sys.stderr)


def watch(state: dict, token: str, dry_run: bool) -> dict:
    action = state.get("action")
    pair = state.get("pair")

    if action not in ("Buy", "Sell"):
        print(f"[info] no active Buy/Sell trade for {pair} (action={action}); nothing to watch.")
        return state

    if state.get("closed"):
        print(f"[info] trade on {pair} already closed this cycle; nothing to watch.")
        return state

    current_price = get_current_price(pair)
    if current_price is None:
        print(f"[warn] could not fetch current price for {pair}", file=sys.stderr)
        return state

    state.setdefault("alerts_sent", {"tp1": False, "tp2": False, "tp3": False, "sl": False})
    alerts = state["alerts_sent"]
    entry = state["entry_price"]
    current_sl = state.get("current_sl", state["stop_loss"])

    # --- Stop Loss check first (using the *current*, possibly trailed, SL) ---
    sl_hit = (action == "Buy" and current_price <= current_sl) or \
             (action == "Sell" and current_price >= current_sl)
    if sl_hit and not alerts["sl"]:
        text = build_alert_text(
            pair, action, "Stop Loss", current_sl, current_price, None,
            "⚠️ Price hit the stop loss level. Consider this trade closed for today."
        )
        send_line(token, text, dry_run)
        alerts["sl"] = True
        state["closed"] = True
        state["last_checked"] = datetime.now(timezone.utc).isoformat()
        return state

    # --- Take Profit levels, in order ---
    tp_levels = [
        ("tp1", "Take Profit 1", state["take_profit_1"]),
        ("tp2", "Take Profit 2", state["take_profit_2"]),
        ("tp3", "Take Profit 3", state["take_profit_3"]),
    ]

    for key, label, level_price in tp_levels:
        if alerts[key]:
            continue
        if near_or_past(action, label, level_price, current_price):
            if key == "tp1":
                new_sl = entry  # move to breakeven
                note = "🎯 Approaching/hit Take Profit 1. Moving Stop Loss to breakeven to protect capital."
            elif key == "tp2":
                new_sl = state["take_profit_1"]  # lock in TP1-level profit
                note = "🎯 Approaching/hit Take Profit 2. Trailing Stop Loss up to Take Profit 1 to lock in profit."
            else:  # tp3
                new_sl = state["take_profit_2"]
                note = "🏁 Approaching/hit Take Profit 3 — final target. Consider closing the remaining position."

            text = build_alert_text(pair, action, label, level_price, current_price, new_sl, note)
            send_line(token, text, dry_run)
            alerts[key] = True
            state["current_sl"] = new_sl
            if key == "tp3":
                state["closed"] = True
            break  # only send one alert per run to avoid double-firing on the same check

    state["last_checked"] = datetime.now(timezone.utc).isoformat()
    state["last_price"] = current_price
    return state


def main():
    parser = argparse.ArgumentParser(description="Agent 5 - Price Watcher & Risk Manager")
    parser.add_argument("--state", default="open_trade.json", help="Path to the open-trade state file")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually send LINE messages")
    args = parser.parse_args()

    state = load_state(args.state)
    if state is None:
        print(f"[info] no open trade state file ({args.state}) found; nothing to watch.")
        sys.exit(0)

    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    updated = watch(state, token, args.dry_run)
    save_state(args.state, updated)
    print(json.dumps(updated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
