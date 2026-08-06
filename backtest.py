"""
backtest.py - Historical backtest of the exact Buy/Sell/TP-SL logic used by
Agent 2 (technical scoring) + Agent 3 (trade-level construction) + Agent 5
(TP1/TP2/TP3 trailing stop-loss), replayed against real historical price data.

This answers the question the live system never answers on its own: "does
this rule actually make money historically, and how often?"

Mechanics (matches the live system's actual behavior, including its
daily-reset quirk):
  - Once per simulated "day" (matching the 7am daily cron), score every pair
    in DEFAULT_PAIRS using the same EMA20/EMA50/RSI14/MACD rule as Agent 2.
  - Pick the single strongest-scoring Buy/Sell candidate (or skip the day if
    none qualify) - matches Agent 3's "pick the best of the day" behavior.
  - Build TP1/TP2/TP3/SL exactly like Agent 3 (support/resistance from the
    last 50 hourly bars).
  - Walk forward hour-by-hour for up to 24 hours (because the live system
    resets open_trade.json at the next day's run, so no trade is actually
    held longer than ~1 day in practice), applying Agent 5's trailing-SL
    rule (breakeven after TP1, TP1-level after TP2, close at TP3).
  - Record the outcome of every trade opened this way, then report:
    win rate, average R-multiple, profit factor, and an equity curve
    assuming a fixed 1R risk per trade.

Usage:
    python backtest.py
    python backtest.py --pairs EURUSD GBPUSD XAUUSD --period 730d

CAVEAT - high-impact news is NOT filtered here:
    The live system (Agent 1 + Agent 3) downgrades/avoids a pair when a
    high-impact calendar event for one of its currencies hasn't released yet
    today (see agent3_signal_synthesizer.pending_high_impact_currencies).
    This backtest has no historical economic-calendar data source to replay
    that check against (ForexFactory's public feed only covers the current
    week), so every trade here is picked on technical score alone, with no
    news-risk filter applied.
    Consequence: this backtest is optimistic relative to the live system. It
    will show trades on some days the live system would have skipped or
    flagged low-confidence around high-impact releases. Treat these results
    as an upper bound on trade frequency/confidence, not a like-for-like
    replay of live behavior.
"""

import argparse
import json
import sys
import time
from datetime import timezone

import numpy as np
import pandas as pd
import yfinance as yf

from agent2_technical_analyzer import DEFAULT_PAIRS, calc_rsi, calc_macd
from agent3_signal_synthesizer import (
    decimals_for_pair,
    build_trade_levels,
    possibility_percent,
)


def download_history(pair: str, symbol: str, period: str, interval: str, retries: int = 4, delay: float = 4.0):
    """Download with retries + a pause before each attempt - Yahoo Finance
    throttles/blocks rapid sequential requests, which otherwise shows up as
    a false 'possibly delisted' error on perfectly valid tickers.

    Uses yf.Ticker(...).history(...) (per-symbol endpoint) instead of the
    batch yf.download() function - the batch endpoint is far more prone to
    triggering Yahoo's anti-bot detection when called repeatedly for
    different symbols in the same run."""
    for attempt in range(1, retries + 1):
        time.sleep(delay)
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
        except Exception as e:
            df = None
            print(f"  [warn] {pair} attempt {attempt}/{retries} raised: {e}", file=sys.stderr)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[["High", "Low", "Close"]].dropna()
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        if attempt < retries:
            backoff = delay * attempt * 2
            print(f"  [retry] {pair}: empty result, waiting {backoff:.0f}s before retry {attempt + 1}/{retries}...")
            time.sleep(backoff)
    return None


def compute_indicators(df: pd.DataFrame):
    close = df["Close"]
    df = df.copy()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["rsi14"] = calc_rsi(close)
    macd_line, signal_line, _ = calc_macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["support"] = close.rolling(50).min()
    df["resistance"] = close.rolling(50).max()
    return df


def score_row(row) -> int:
    score = 0
    score += 1 if row["ema20"] > row["ema50"] else -1
    rsi = row["rsi14"]
    if not np.isnan(rsi):
        if rsi > 70:
            score -= 1
        elif rsi < 30:
            score += 1
    score += 1 if row["macd"] > row["macd_signal"] else -1
    return score


def bias_for_score(score: int) -> str:
    if score >= 2:
        return "bullish"
    if score <= -2:
        return "bearish"
    return "neutral"


def simulate_trade(pair: str, hist: pd.DataFrame, entry_idx: int, action: str,
                    entry: float, tp1: float, tp2: float, tp3: float, sl0: float,
                    max_hold_bars: int):
    """Walk forward from entry_idx applying Agent 5's trailing-SL rule.

    Checks each bar's High/Low (not just its Close) so intrabar wicks that
    would have triggered a real SL/TP fill aren't missed. When a single
    bar's range spans both the current stop and a take-profit level, the
    stop is assumed to hit first - the true intrabar order is unknown, and
    assuming the stop wins is the conservative (non-optimistic) choice.
    Returns dict with outcome, exit price, R-multiple."""
    current_sl = sl0
    tp1_hit = tp2_hit = tp3_hit = False
    risk = abs(entry - sl0)
    if risk == 0:
        return None

    highs = hist["High"].values
    lows = hist["Low"].values
    closes = hist["Close"].values
    n = len(closes)
    end_idx = min(entry_idx + max_hold_bars, n - 1)

    for i in range(entry_idx + 1, end_idx + 1):
        high, low = highs[i], lows[i]

        adverse = low if action == "Buy" else high
        sl_hit = (action == "Buy" and adverse <= current_sl) or (action == "Sell" and adverse >= current_sl)
        if sl_hit:
            exit_price = current_sl
            r_multiple = (exit_price - entry) / risk if action == "Buy" else (entry - exit_price) / risk
            return {"pair": pair, "outcome": "SL", "exit_price": exit_price, "r_multiple": r_multiple,
                    "tp1": tp1_hit, "tp2": tp2_hit, "tp3": tp3_hit}

        favorable = high if action == "Buy" else low
        if not tp1_hit and ((action == "Buy" and favorable >= tp1) or (action == "Sell" and favorable <= tp1)):
            tp1_hit = True
            current_sl = entry
        if tp1_hit and not tp2_hit and ((action == "Buy" and favorable >= tp2) or (action == "Sell" and favorable <= tp2)):
            tp2_hit = True
            current_sl = tp1
        if tp2_hit and not tp3_hit and ((action == "Buy" and favorable >= tp3) or (action == "Sell" and favorable <= tp3)):
            tp3_hit = True
            exit_price = tp3
            r_multiple = (exit_price - entry) / risk if action == "Buy" else (entry - exit_price) / risk
            return {"pair": pair, "outcome": "TP3", "exit_price": exit_price, "r_multiple": r_multiple,
                    "tp1": tp1_hit, "tp2": tp2_hit, "tp3": tp3_hit}

    # Day ended (next daily reset) without SL or TP3 - mark-to-market at last close
    exit_price = closes[end_idx]
    r_multiple = (exit_price - entry) / risk if action == "Buy" else (entry - exit_price) / risk
    return {"pair": pair, "outcome": "TIMEOUT", "exit_price": exit_price, "r_multiple": r_multiple,
            "tp1": tp1_hit, "tp2": tp2_hit, "tp3": tp3_hit}


def run_backtest(pairs, period, interval, bars_per_day, delay):
    print(f"Downloading {interval} history ({period}) for {len(pairs)} pairs...")
    data = {}
    for pair in pairs:
        symbol = DEFAULT_PAIRS.get(pair.upper())
        if not symbol:
            continue
        hist = download_history(pair, symbol, period, interval, delay=delay)
        if hist is None or len(hist) < 100:
            print(f"  [skip] {pair}: insufficient data")
            continue
        data[pair] = compute_indicators(hist)
        print(f"  [ok] {pair}: {len(hist)} bars")

    if not data:
        print("[error] no data downloaded for any pair - cannot backtest.", file=sys.stderr)
        sys.exit(1)

    # Align all pairs on a shared set of "daily decision" timestamps (every
    # bars_per_day bars, matching the once-a-day cron), starting after a
    # 50-bar warmup so rolling indicators are valid.
    min_len = min(len(df) for df in data.values())
    decision_indices = list(range(60, min_len - bars_per_day, bars_per_day))

    trades = []
    for idx in decision_indices:
        candidates = []
        for pair, df in data.items():
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            if row[["ema20", "ema50", "rsi14", "macd", "macd_signal", "support", "resistance"]].isna().any():
                continue
            score = score_row(row)
            bias = bias_for_score(score)
            action = {"bullish": "Buy", "bearish": "Sell", "neutral": "Wait"}[bias]
            if action == "Wait":
                continue
            candidates.append((abs(score), pair, action, score, row))

        if not candidates:
            continue

        candidates.sort(key=lambda c: c[0], reverse=True)
        _, pair, action, score, row = candidates[0]
        df = data[pair]
        entry = float(row["Close"])
        support = float(row["support"])
        resistance = float(row["resistance"])
        ndigits = decimals_for_pair(pair)
        levels = build_trade_levels(action, entry, support, resistance, ndigits)
        if not levels:
            continue

        result = simulate_trade(
            pair, df, idx, action, entry,
            levels["take_profit_1"], levels["take_profit_2"], levels["take_profit_3"],
            levels["stop_loss"], bars_per_day,
        )
        if result:
            result["timestamp"] = str(df.index[idx])
            result["action"] = action
            result["score"] = score
            trades.append(result)

    return trades


def summarize(trades):
    if not trades:
        print("No trades were generated - nothing to summarize.")
        return

    n = len(trades)
    wins = [t for t in trades if t["r_multiple"] > 0]
    losses = [t for t in trades if t["r_multiple"] < 0]
    breakeven = [t for t in trades if t["r_multiple"] == 0]
    decided = len(wins) + len(losses)
    win_rate = len(wins) / decided * 100 if decided else 0.0
    avg_r = sum(t["r_multiple"] for t in trades) / n
    gross_win = sum(t["r_multiple"] for t in wins)
    gross_loss = -sum(t["r_multiple"] for t in losses) or 1e-9
    profit_factor = gross_win / gross_loss

    equity = [0.0]
    for t in trades:
        equity.append(equity[-1] + t["r_multiple"])
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    by_pair = {}
    for t in trades:
        by_pair.setdefault(t["pair"], []).append(t["r_multiple"])

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Total trades:      {n}")
    print(f"Win rate:          {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L, excl. {len(breakeven)} breakeven)")
    print(f"Avg R-multiple:    {avg_r:+.2f}R per trade")
    print(f"Profit factor:     {profit_factor:.2f}  (>1.0 = net profitable in R terms)")
    print(f"Net R over period: {equity[-1]:+.2f}R")
    print(f"Max drawdown:      {max_dd:.2f}R")
    outcome_counts = {}
    for t in trades:
        outcome_counts[t["outcome"]] = outcome_counts.get(t["outcome"], 0) + 1
    print(f"Outcomes:          {outcome_counts}")

    print("\nBy pair (trades, avg R):")
    for pair, rs in sorted(by_pair.items(), key=lambda kv: -len(kv[1])):
        print(f"  {pair:8s}  n={len(rs):3d}  avgR={sum(rs)/len(rs):+.2f}")
    print("=" * 60)
    print("NOTE: high-impact news is NOT filtered in this backtest (no historical")
    print("calendar data source available - see module docstring). Every trade above")
    print("was picked on technical score alone. The live system additionally avoids/")
    print("downgrades pairs with a pending high-impact release, so it should trade")
    print("less often and with lower confidence around news than shown here - treat")
    print("these numbers as an optimistic upper bound, not a like-for-like replay.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Backtest the Agent 2/3/5 Buy-Sell-TP-SL rule against history")
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS.keys()),
                         help=f"Pairs to include (default: all {len(DEFAULT_PAIRS)} in DEFAULT_PAIRS)")
    parser.add_argument("--period", default="730d", help="History window (yfinance format, e.g. 730d, 60d)")
    parser.add_argument("--interval", default="1h", help="Bar interval (must match what Agent 2 uses live: 1h)")
    parser.add_argument("--bars-per-day", type=int, default=24,
                         help="Bars between daily decision points (24 for hourly bars = daily cron)")
    parser.add_argument("--delay", type=float, default=3.0,
                         help="Seconds to pause before each pair's download, to avoid Yahoo Finance rate limits")
    parser.add_argument("--out", default="backtest_trades.json", help="Where to save the raw trade log")
    args = parser.parse_args()

    trades = run_backtest(args.pairs, args.period, args.interval, args.bars_per_day, args.delay)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)

    summarize(trades)
    print(f"\nFull trade log written to {args.out}")


if __name__ == "__main__":
    main()
