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
import time

import requests

LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"

# LINE rejects a text message over 5000 characters and accepts at most 5
# message objects per request. Three cards with QC warnings attached had no
# trouble fitting; three cards, three QC warnings and a run of long event
# titles is another matter, and the failure mode was a 400 that dropped the
# whole broadcast rather than one long card.
LINE_MAX_TEXT_CHARS = 5000
LINE_MAX_MESSAGES = 5
# Leave room for the "(1/3)" continuation marker.
CHUNK_CHARS = LINE_MAX_TEXT_CHARS - 100

# Transient failures worth another attempt: LINE's own rate limit and any 5xx.
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


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
        # R:R to TP1 leads, TP3 follows as the stretch.
        #
        # The card used to quote the TP3 ratio alone, and under the old level
        # rule that number reached 28.4 while TP3 was being hit on 1.9% of
        # trades. Leading with a ratio for an outcome that almost never happens
        # is the same class of decoration as the constant 70% possibility this
        # project already removed.
        rr1 = signal.get("risk_reward_tp1")
        rr3 = signal.get("risk_reward")
        sl_line = f"SL: {signal.get('stop_loss')}"
        if rr1:
            sl_line += f"  |  R:R {rr1} to TP1"
            if rr3:
                sl_line += f" ({rr3} to TP3)"
        elif rr3:
            sl_line += f"  |  R:R {rr3} to TP3"
        lines.append(sl_line)

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

    # Published under --allow-unproven-edge: the action was NOT downgraded, so
    # without this line the card reads like any other approved trade with a
    # warning flag attached. Say the quiet part on its own line.
    if "unproven_edge" in signal.get("qc_flags", []) and signal.get("action") in ("Buy", "Sell"):
        lines.append(
            "   ⚠️ UNPROVEN EDGE — the backtest shows no measurable edge for this "
            "strategy. This card is being sent to collect live samples. Size accordingly."
        )

    if "stale_price" in signal.get("qc_flags", []):
        lines.append(
            "   🕒 STALE PRICE — the entry above is older than one bar of this "
            "timeframe. Re-check the live quote before acting."
        )

    if "news_unverified" in signal.get("qc_flags", []):
        lines.append(
            "   📰 NEWS NOT CHECKED — the economic calendar could not be fetched, "
            "so a pending high-impact release cannot be ruled out."
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


def split_for_line(text: str, chunk_chars: int = CHUNK_CHARS,
                   max_messages: int = LINE_MAX_MESSAGES):
    """Split a message on card boundaries so LINE will accept it.

    Splits on the "─────" separator between cards first, so a card is never
    torn in half, and only falls back to a hard cut for a single card that is
    somehow longer than the limit on its own. Anything past `max_messages` is
    dropped with a line saying so - silently sending the first five sixths of
    a run would be worse than saying five sixths is all that fits.
    """
    if len(text) <= chunk_chars:
        return [text]

    chunks, current = [], ""
    for block in text.split("\n─────────────"):
        block = block if not chunks and not current else "\n─────────────" + block
        if current and len(current) + len(block) > chunk_chars:
            chunks.append(current)
            current = block.lstrip("\n")
        else:
            current += block
    if current:
        chunks.append(current)

    # A single oversized card: cut it rather than let LINE reject the request.
    final = []
    for chunk in chunks:
        while len(chunk) > chunk_chars:
            final.append(chunk[:chunk_chars])
            chunk = chunk[chunk_chars:]
        final.append(chunk)

    if len(final) > max_messages:
        dropped = len(final) - max_messages
        final = final[:max_messages]
        final[-1] += f"\n\n… {dropped} more message(s) did not fit and were not sent."

    total = len(final)
    return [f"({i}/{total})\n{c}" for i, c in enumerate(final, 1)]


def send_broadcast(token: str, text: str, attempts: int = RETRY_ATTEMPTS,
                   backoff: float = RETRY_BACKOFF_SECONDS, sleep=time.sleep):
    """Broadcast one message, retrying the failures that are worth retrying.

    A 429 or a 5xx is LINE having a moment; a 400 or a 401 is this code or the
    token being wrong, and retrying those just sends the same broken request
    three times. The response of the last attempt is returned either way, so
    the caller still reports the real status.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {"messages": [{"type": "text", "text": text}]}

    resp = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(LINE_BROADCAST_URL, headers=headers, json=payload, timeout=15)
        except requests.RequestException as e:
            if attempt == attempts:
                raise
            print(f"[retry] LINE request failed ({e}); attempt {attempt}/{attempts}",
                  file=sys.stderr)
            sleep(backoff * attempt)
            continue
        if resp.status_code not in RETRY_STATUS or attempt == attempts:
            return resp
        print(f"[retry] LINE returned {resp.status_code}; attempt {attempt}/{attempts}",
              file=sys.stderr)
        sleep(backoff * attempt)
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
    except FileNotFoundError:
        print(f"[error] {args.signal} not found. This file is Agent 6's output — "
              f"run the pipeline in order (run_all.py), or at least "
              f"`python agent6_qc_reviewer.py` first.", file=sys.stderr)
        sys.exit(1)
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

    parts = split_for_line(message_text)
    failed = 0
    for i, part in enumerate(parts, 1):
        resp = send_broadcast(token, part)
        if resp is not None and resp.status_code == 200:
            print(f"[ok] broadcast {i}/{len(parts)} sent successfully.")
        else:
            failed += 1
            status = "no response" if resp is None else resp.status_code
            body = "" if resp is None else f": {resp.text}"
            print(f"[error] LINE API returned {status} on part {i}/{len(parts)}{body}",
                  file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
