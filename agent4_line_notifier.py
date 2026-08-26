"""
Agent 4 - LINE Notifier
=========================
Reads Agent 6's signal_reviewed.json — Agent 3's signals after the QC layer
has vetted them — and sends them as a formatted LINE message using the LINE
Messaging API Broadcast endpoint (sends to everyone who has added your bot as
a friend — fine for personal/solo use).

What QC does to what gets sent:
    approved   — sent as normal
    downgraded — sent, with the QC warning and flags attached to the card
    blocked    — not sent at all

Setup:
    1. Create a Messaging API channel at https://developers.line.biz
    2. Issue a "Channel access token" from the channel's Messaging API tab
    3. Add your bot as a friend by scanning its QR code
    4. Set the token as an environment variable (PowerShell):
           setx LINE_CHANNEL_ACCESS_TOKEN "your-long-token-here"
       (restart PowerShell after running setx once)

Usage:
    python agent4_line_notifier.py
    python agent4_line_notifier.py --signal signal_reviewed.json --dry-run
"""

import argparse
import json
import os
import sys

import requests

LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"


def build_signal_block(signal: dict) -> list:
    """One trade card, as a list of lines."""
    direction_emoji = {"bullish": "🟢 BUY", "bearish": "🔴 SELL", "neutral": "⚪ NEUTRAL"}
    direction_label = direction_emoji.get(signal.get("direction"), str(signal.get("direction")).upper())

    # A signal QC turned into a Wait keeps its bullish/bearish read, but the
    # card must not open with a green BUY banner on a trade you shouldn't take.
    original_action = signal.get("original_action")
    if signal.get("action") == "Wait" and original_action in ("Buy", "Sell"):
        direction_label = f"⚪ NO TRADE (QC held back a {original_action})"

    confidence_emoji = {"high": "🔥", "medium": "⚖️", "low": "⚠️"}
    conf_label = confidence_emoji.get(signal.get("confidence"), "")

    lines = [
        f"📊 {signal['pair']}",
        f"{direction_label}  |  Confidence: {signal.get('confidence')} {conf_label}"
        f"  |  Possibility: {signal.get('possibility_percent')}%",
        f"Status: {signal.get('status')}  |  Time frame: {signal.get('time_frame')}",
        f"Entry: {signal.get('open_price')}",
    ]

    if signal.get("action") in ("Buy", "Sell"):
        lines.append(
            f"TP1: {signal.get('take_profit_1')}  |  TP2: {signal.get('take_profit_2')}"
            f"  |  TP3: {signal.get('take_profit_3')}"
        )
        rr = signal.get("risk_reward")
        lines.append(f"SL: {signal.get('stop_loss')}" + (f"  |  R:R {rr}" if rr else ""))

    lines.append(f"Support: {signal.get('support')} | Resistance: {signal.get('resistance')}")

    # Say how stale the entry price is, so a delayed run is obvious at a glance
    # instead of looking like a live quote.
    age = signal.get("data_age_minutes")
    if age is not None:
        lines.append(f"🕒 Price from {signal.get('last_bar_time')} ({age:.0f} min ago)")

    for reason in signal.get("reasons", []):
        lines.append(f"• {reason}")

    if signal.get("pending_high_impact_news"):
        lines.append("⚠️ High-impact news pending for this pair — expect volatility.")

    lines.extend(build_qc_warning(signal))

    return lines


def build_qc_warning(signal: dict) -> list:
    """Agent 6's verdict, spelled out on the card.

    Only downgraded signals reach here — blocked ones are filtered out before
    the message is built, and approved ones have nothing to warn about. The
    warning goes last so it's the final thing read before acting on the card.
    """
    if signal.get("qc_status") != "downgraded":
        return []

    flags = ", ".join(signal.get("qc_flags", [])) or "unspecified"
    lines = [f"🛑 QC WARNING [{flags}]"]

    original_action = signal.get("original_action")
    if original_action and original_action != signal.get("action"):
        lines.append(
            f"   Downgraded {original_action} → {signal.get('action')} by QC — do not enter this trade."
        )

    original_possibility = signal.get("original_possibility_percent")
    if original_possibility is not None:
        lines.append(
            f"   Possibility cut {original_possibility}% → {signal.get('possibility_percent')}%"
        )

    if signal.get("qc_notes"):
        lines.append(f"   {signal['qc_notes']}")

    return lines


def select_signals(signal_data: dict, send_all: bool) -> list:
    """Signals worth sending this run.

    The pipeline runs several times a day now, so by default only signals
    Agent 3 flagged as new go out — otherwise the same trade would be
    broadcast again on every run until it closes.

    Anything Agent 6 blocked is dropped here regardless of --all: a blocked
    signal failed QC outright (stale data, most often), and the whole point of
    the QC layer is that those never reach your phone."""
    signals = signal_data.get("signals")
    if signals is None:
        # Legacy single-signal signal.json
        signals = [signal_data["signal"]] if signal_data.get("signal") else []
    signals = [s for s in signals if s.get("qc_status") != "blocked"]
    if send_all:
        return signals
    return [s for s in signals if s.get("new")]


def build_message_text(signal_data: dict, signals: list) -> str:
    if not signals:
        return "⚠️ Forex Signal Bot: no valid signal right now.\n" + str(signal_data.get("note", ""))

    header = "📊 Forex Signals" if len(signals) > 1 else "📊 Forex Signal"
    blocks = [f"{header} ({len(signals)})"]
    for signal in signals:
        blocks.append("─────────────")
        blocks.extend(build_signal_block(signal))
    return "\n".join(blocks)


def send_broadcast(token: str, text: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "messages": [
            {"type": "text", "text": text}
        ]
    }
    resp = requests.post(LINE_BROADCAST_URL, headers=headers, json=payload, timeout=15)
    return resp


def main():
    parser = argparse.ArgumentParser(description="Agent 4 - LINE Notifier")
    parser.add_argument("--signal", default="signal_reviewed.json",
                         help="Path to Agent 6 output (QC-reviewed signals)")
    parser.add_argument("--dry-run", action="store_true", help="Print the message instead of sending to LINE")
    parser.add_argument("--all", action="store_true",
                         help="Send every published signal, not just the ones flagged new")
    args = parser.parse_args()

    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token and not args.dry_run:
        print("[error] LINE_CHANNEL_ACCESS_TOKEN environment variable not set.", file=sys.stderr)
        print("        Set it with: setx LINE_CHANNEL_ACCESS_TOKEN \"your-token\"", file=sys.stderr)
        print("        Or run with --dry-run to preview the message without sending.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.signal, "r", encoding="utf-8") as f:
            signal_data = json.load(f)
    except Exception as e:
        print(f"[error] could not read {args.signal}: {e}", file=sys.stderr)
        sys.exit(1)

    signals = select_signals(signal_data, args.all)
    if not signals and signal_data.get("signals"):
        print("[info] no new signals this run — nothing to send.")
        for skipped in signal_data["signals"]:
            print(f"       {skipped['pair']}: {skipped.get('skip_reason', 'not new')}")
        return

    message_text = build_message_text(signal_data, signals)
    print("--- Message to send ---")
    print(message_text)
    print("-----------------------")

    if args.dry_run:
        print("[dry-run] not sent.")
        return

    resp = send_broadcast(token, message_text)
    if resp.status_code == 200:
        print("[ok] broadcast sent successfully.")
    else:
        print(f"[error] LINE API returned {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
