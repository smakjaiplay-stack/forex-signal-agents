"""
Agent 2 - Technical Analyzer
=============================
Pulls recent price data for major forex pairs from Yahoo Finance,
computes technical indicators (RSI, MACD, EMA, support/resistance),
and outputs a structured JSON summary that Agent 3 (Signal Synthesizer)
can consume.

Usage:
    python agent2_technical_analyzer.py
    python agent2_technical_analyzer.py --pairs EURUSD GBPUSD --interval 1h --period 5d

Output:
    Prints JSON to stdout, and writes to technical_analysis.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# Major forex pairs -> Yahoo Finance ticker symbols
DEFAULT_PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCHF": "USDCHF=X",
    # Yahoo no longer serves spot XAUUSD=X/XAGUSD=X - GC=F/SI=F (COMEX futures)
    # are the closest free proxy (tracks spot closely, small basis).
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
}


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def analyze_pair(symbol: str, interval: str, period: str) -> dict:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)

    if df is None or df.empty or len(df) < 50:
        return {"error": f"insufficient data for {symbol} (got {0 if df is None else len(df)} rows)"}

    # yfinance sometimes returns MultiIndex columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    close = df["Close"].dropna()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    rsi = calc_rsi(close)
    macd_line, signal_line, histogram = calc_macd(close)

    last_close = float(close.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None
    last_macd = float(macd_line.iloc[-1])
    last_signal = float(signal_line.iloc[-1])
    last_hist = float(histogram.iloc[-1])

    # Simple recent support/resistance from last 50 bars
    recent = close.tail(50)
    resistance = float(recent.max())
    support = float(recent.min())

    # --- Bias scoring (simple rule-based, -3 to +3) ---
    score = 0
    reasons = []

    if last_ema20 > last_ema50:
        score += 1
        reasons.append("EMA20 above EMA50 (short-term uptrend)")
    else:
        score -= 1
        reasons.append("EMA20 below EMA50 (short-term downtrend)")

    if last_rsi is not None:
        if last_rsi > 70:
            score -= 1
            reasons.append(f"RSI {last_rsi:.1f} overbought")
        elif last_rsi < 30:
            score += 1
            reasons.append(f"RSI {last_rsi:.1f} oversold")
        else:
            reasons.append(f"RSI {last_rsi:.1f} neutral")

    if last_macd > last_signal:
        score += 1
        reasons.append("MACD above signal line (bullish momentum)")
    else:
        score -= 1
        reasons.append("MACD below signal line (bearish momentum)")

    if score >= 2:
        bias = "bullish"
    elif score <= -2:
        bias = "bearish"
    else:
        bias = "neutral"

    return {
        "symbol": symbol,
        "last_close": round(last_close, 5),
        "ema20": round(last_ema20, 5),
        "ema50": round(last_ema50, 5),
        "rsi14": round(last_rsi, 2) if last_rsi is not None else None,
        "macd": round(last_macd, 6),
        "macd_signal": round(last_signal, 6),
        "macd_histogram": round(last_hist, 6),
        "support": round(support, 5),
        "resistance": round(resistance, 5),
        "bias": bias,
        "score": score,
        "reasons": reasons,
    }


def main():
    parser = argparse.ArgumentParser(description="Agent 2 - Technical Analyzer")
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS.keys()),
                         help="Currency pairs to analyze, e.g. EURUSD GBPUSD")
    parser.add_argument("--interval", default="1h", help="Candle interval (e.g. 15m, 1h, 4h, 1d)")
    parser.add_argument("--period", default="5d", help="How much history to pull (e.g. 5d, 1mo)")
    parser.add_argument("--out", default="technical_analysis.json", help="Output JSON file path")
    args = parser.parse_args()

    results = []
    for pair in args.pairs:
        symbol = DEFAULT_PAIRS.get(pair.upper())
        if not symbol:
            print(f"[warn] unknown pair '{pair}', skipping", file=sys.stderr)
            continue
        try:
            result = analyze_pair(symbol, args.interval, args.period)
        except Exception as e:
            result = {"error": str(e)}
        result["pair"] = pair.upper()
        results.append(result)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interval": args.interval,
        "period": args.period,
        "results": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
