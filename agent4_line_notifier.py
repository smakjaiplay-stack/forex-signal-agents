"""
Agent 4 - LINE Notifier
=========================
Reads Agent 3's signal.json and sends it as a formatted LINE message using
the LINE Messaging API Broadcast endpoint (sends to everyone who has added
your bot as a friend — fine for personal/solo use).

Setup:
    1. Create a Messaging API channel at https://developers.line.biz
    2. Issue a "Channel access token" from the channel's Messaging API tab
    3. Add your bot as a friend by scanning its QR code
    4. Set the token as an environment variable (PowerShell):
           setx LINE_CHANNEL_ACCESS_TOKEN "your-long-token-here"
       (restart PowerShell after running setx once)

Usage:
    python agent4_line_notifier.py
    python agent4_line_notifier.py --signal signal.json
"""

import argparse
import json
import os
import sys

import requests

LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"


def build_message_text(signal_data: dict) -> str:
    signal = signal_data.get("signal")
    if not signal:
        return "⚠️ Forex Signal Bot: no valid signal today.\n" + str(signal_data.get("note", ""))

    direction_emoji = {"bullish": "🟢 BUY", "bearish": "🔴 SELL", "neutral": "⚪ NEUTRAL"}
    direction_label = direction_emoji.get(signal["direction"], signal["direction"].upper())

    confidence_emoji = {"high": "🔥", "medium": "⚖️", "low": "⚠️"}
    conf_label = confidence_emoji.get(signal["confidence"], "")

    lines = [
        f"📊 Forex Signal — {signal['pair']}",
        f"{direction_label}  |  Confidence: {signal['confidence']} {conf_label}",
        f"Ref price: {signal['entry_reference']}",
        f"Support: {signal['support']} | Resistance: {signal['resistance']}",
        "",
        "Reasons:",
    ]
    for reason in signal.get("reasons", []):
        lines.append(f"• {reason}")

    if signal.get("pending_high_impact_news"):
        lines.append("")
        lines.append("⚠️ High-impact news pending today for this pair's currency — expect volatility.")

    return "\n".join(lines)


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
    parser.add_argument("--signal", default="signal.json", help="Path to Agent 3 output")
    parser.add_argument("--dry-run", action="store_true", help="Print the message instead of sending to LINE")
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

    message_text = build_message_text(signal_data)
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
