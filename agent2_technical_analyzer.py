"""
Agent 2 - Technical Analyzer
=============================
Pulls recent price data for major forex pairs from Yahoo Finance,
computes technical indicators (RSI, MACD, EMA, support/resistance),
and outputs a structured JSON summary that Agent 3 (Signal Synthesizer)
can consume.

Usage:
    python agent2_technical_analyzer.py
    python agent2_technical_analyzer.py --pairs EURUSD GBPUSD

The bar interval is NOT a per-run choice: it defaults to scoring.DEFAULT_INTERVAL
because the score threshold and the win-rate calibration were both measured on
that bar size. Changing it means re-running backtest.py and validate.py.

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

import scoring

# EMA200 is one of the score's components, so a window shorter than this
# makes the higher-timeframe read meaningless rather than merely noisy.
MIN_BARS = 200

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


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing).

    Agent 3 uses this to cap how far a stop-loss can sit from entry. Without
    it, a stop parked at the far edge of the 50-bar range can be many times
    wider than the target, which is how the old levels produced trades
    risking 11 to make 1.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def analyze_pair(symbol: str, interval: str, period: str,
                 min_score: float = scoring.DEFAULT_MIN_SCORE,
                 use_ic_weights: bool = False) -> dict:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)

    if df is None or df.empty or len(df) < MIN_BARS:
        return {"error": f"insufficient data for {symbol} "
                         f"(got {0 if df is None else len(df)} rows, need {MIN_BARS})"}

    # yfinance sometimes returns MultiIndex columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    close = df["Close"].dropna()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    # One timeframe up. See scoring.component_values for why EMA80/EMA200 on
    # the same series stands in for a resampled higher timeframe.
    ema80 = close.ewm(span=80, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    rsi = calc_rsi(close)
    macd_line, signal_line, histogram = calc_macd(close)

    last_close = float(close.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_ema80 = float(ema80.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])
    last_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None
    last_macd = float(macd_line.iloc[-1])
    last_signal = float(signal_line.iloc[-1])
    last_hist = float(histogram.iloc[-1])

    # Simple recent support/resistance from last 50 bars
    recent = close.tail(50)
    resistance = float(recent.max())
    support = float(recent.min())

    # How old is the candle we're pricing off? Yahoo's intraday feed lags,
    # and the last bar is still forming, so the entry price can be minutes
    # behind the market — report it instead of letting it hide.
    atr = None
    plus_di = minus_di = None
    if {"High", "Low"}.issubset(df.columns):
        atr_series = calc_atr(df).dropna()
        if not atr_series.empty:
            atr = float(atr_series.iloc[-1])
        _, plus_series, minus_series = scoring.calc_adx(df)
        plus_series, minus_series = plus_series.dropna(), minus_series.dropna()
        if not plus_series.empty and not minus_series.empty:
            plus_di = float(plus_series.iloc[-1])
            minus_di = float(minus_series.iloc[-1])

    last_ts = pd.Timestamp(close.index[-1])
    last_ts = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")
    data_age_minutes = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 60

    # --- Bias scoring (continuous, -3 to +3) ---
    # The rule itself lives in scoring.py so this and backtest.py cannot
    # drift apart. See that module's docstring for why the old +/-1 vote
    # score was replaced.
    values = scoring.component_values(
        close=last_close, ema20=last_ema20, ema50=last_ema50,
        ema80=last_ema80, ema200=last_ema200, rsi=last_rsi,
        macd_hist=last_hist, atr=atr, plus_di=plus_di, minus_di=minus_di,
        support=support, resistance=resistance,
    )
    score, contributions = scoring.compute_score(values, use_ic_weights=use_ic_weights)
    bias = scoring.bias_for_score(score, min_score)
    reasons = scoring.describe(values, rsi=last_rsi)

    return {
        "symbol": symbol,
        "last_close": round(last_close, 5),
        "ema20": round(last_ema20, 5),
        "ema50": round(last_ema50, 5),
        "ema80": round(last_ema80, 5),
        "ema200": round(last_ema200, 5),
        "rsi14": round(last_rsi, 2) if last_rsi is not None else None,
        "macd": round(last_macd, 6),
        "macd_signal": round(last_signal, 6),
        "macd_histogram": round(last_hist, 6),
        "plus_di": round(plus_di, 2) if plus_di is not None else None,
        "minus_di": round(minus_di, 2) if minus_di is not None else None,
        "support": round(support, 5),
        "resistance": round(resistance, 5),
        "atr14": round(atr, 5) if atr is not None else None,
        "last_bar_time": last_ts.strftime("%Y-%m-%d %H:%M UTC"),
        "data_age_minutes": round(data_age_minutes, 1),
        "bias": bias,
        "score": round(score, 4),
        # Per-component detail, carried through Agent 3 into signal.json so
        # validate.py can measure each component's IC against live outcomes
        # the same way it does against the backtest.
        "components": {k: round(v, 4) for k, v in values.items()},
        "contributions": {k: round(v, 4) for k, v in contributions.items()},
        "reasons": reasons,
    }


def build_parser():
    """Split out of main() so the defaults are testable.

    test_scoring.TestTimeframeIsShared asserts this parser's --interval
    default is scoring.DEFAULT_INTERVAL. That test is the thing that
    actually prevents the live/backtest drift; a comment claiming the two
    agree is what let them diverge in the first place.
    """
    parser = argparse.ArgumentParser(description="Agent 2 - Technical Analyzer")
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS.keys()),
                         help="Currency pairs to analyze, e.g. EURUSD GBPUSD")
    # Both come from scoring.py, which is also where DEFAULT_MIN_SCORE and the
    # calibration table live - the threshold is only valid for the interval it
    # was measured on, so the two cannot be set independently. Overriding
    # --interval here without re-running backtest.py and validate.py means
    # scoring bars against a threshold nobody derived for them.
    parser.add_argument("--interval", default=scoring.DEFAULT_INTERVAL,
                         help=f"Candle interval (default {scoring.DEFAULT_INTERVAL}; must match "
                              f"what backtest.py replayed - see scoring.py)")
    parser.add_argument("--period", default=scoring.DEFAULT_LIVE_PERIOD,
                         help=f"How much history to pull (default {scoring.DEFAULT_LIVE_PERIOD}; "
                              f"needs to clear the {MIN_BARS}-bar EMA200 warmup)")
    parser.add_argument("--out", default="technical_analysis.json", help="Output JSON file path")
    parser.add_argument("--min-score", type=float, default=scoring.DEFAULT_MIN_SCORE,
                         help="Minimum |score| before a pair gets a directional bias")
    parser.add_argument("--use-ic-weights", action="store_true",
                         help="Score with the weights in ic_weights.json instead of equal weights "
                              "(read scoring.py's docstring before turning this on)")
    return parser


def main():
    args = build_parser().parse_args()

    results = []
    for pair in args.pairs:
        symbol = DEFAULT_PAIRS.get(pair.upper())
        if not symbol:
            print(f"[warn] unknown pair '{pair}', skipping", file=sys.stderr)
            continue
        try:
            result = analyze_pair(symbol, args.interval, args.period,
                                  min_score=args.min_score,
                                  use_ic_weights=args.use_ic_weights)
        except Exception as e:
            result = {"error": str(e)}
        result["pair"] = pair.upper()
        results.append(result)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interval": args.interval,
        "period": args.period,
        "min_score": args.min_score,
        "ic_weights_applied": bool(args.use_ic_weights and scoring.load_ic_weights()),
        "results": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
