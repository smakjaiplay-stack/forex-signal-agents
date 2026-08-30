"""
backtest.py - Historical backtest of the exact Buy/Sell/TP-SL logic used by
Agent 2 (technical scoring) + Agent 3 (trade-level construction) + Agent 5
(TP1/TP2/TP3 trailing stop-loss), replayed against real historical price data.

This answers the question the live system never answers on its own: "does
this rule actually make money historically, and how often?"

Mechanics (matches the live system's actual behavior, including its
daily-reset quirk):
  - Once per simulated "day" (see the CADENCE caveat below), score every pair
    in DEFAULT_PAIRS using scoring.py - the same module Agent 2 calls live.
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

CADENCE AND HOLD ARE SEPARATE PARAMETERS - they did not used to be:
    --decision-every says how often a trade can be opened (4 bars, matching
    the "7 */4 * * 1-5" cron), and --max-hold-bars says how long one is held
    (24 bars, matching agent5.MAX_HOLD_HOURS). Both used to come from a single
    --bars-per-day argument, and the calibration checked in before this change
    was generated with --bars-per-day 4 in order to match the 4-hourly cron.
    That silently cut the hold to 4 bars as well, so all 1,312 trades behind
    score_calibration.json were four-hour trades describing a system that
    holds for twenty-four. test_scoring.TestCadenceAndHoldAreSeparate now pins
    the two defaults against agent5, because a caveat in a docstring is
    exactly what failed to hold this invariant the first time.
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import timezone

import numpy as np
import pandas as pd
import yfinance as yf

import scoring
from agent2_technical_analyzer import DEFAULT_PAIRS, calc_rsi, calc_macd, calc_atr
from agent3_signal_synthesizer import (
    DEFAULT_ATR_MULT,
    DEFAULT_MIN_RR,
    DEFAULT_MIN_SCORE,
    decimals_for_pair,
    build_trade_levels,
    possibility_percent,
    risk_reward,
)

# Bump when compute_indicators changes what it produces. The cache used to be
# keyed on (pairs, period, interval) alone while storing the DERIVED columns,
# so editing an indicator - support/resistance moving from closes to lows/highs,
# for instance - left every later run silently replaying the old definition
# from disk and reporting it as a fresh measurement.
INDICATOR_VERSION = 2

# ---------------------------------------------------------------------------
# Exit policies
#
# The original rule ("trail-be") moved the stop to breakeven the moment TP1
# was touched. Measured over 522 trades it turned 129 of them - 24.7% - into
# exactly 0R, while losers still paid the full -1R. TP1 was reached on 78.7%
# of trades but TP3 on only 33.5%, so the trail was harvesting the difference
# and handing back nothing.
#
# These are the alternatives worth measuring against it. Every one is
# evaluated on the same bars, same entries, same levels, so the only variable
# is what happens after the trade is open.
# ---------------------------------------------------------------------------
EXIT_POLICIES = {
    # Current live behaviour, kept as the baseline to beat.
    "trail-be": "SL -> breakeven at TP1, -> TP1 at TP2, close at TP3",
    # No trailing at all: the stop stays where it was placed.
    "fixed": "SL never moves; exit at TP3, original SL, or timeout",
    # Move the stop only once the trade has proved itself twice over.
    "trail-late": "SL -> breakeven at TP2 only, close at TP3",
    # Bank a third at each target instead of tightening the stop.
    "partial": "close 1/3 at TP1, 1/3 at TP2, 1/3 at TP3; SL never moves",
    # Bank a third, then protect the rest - the usual compromise.
    "partial-be": "close 1/3 at TP1 then SL -> breakeven; 1/3 at TP2, 1/3 at TP3",
}
DEFAULT_EXIT_POLICY = "trail-be"


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
    """Every column scoring.score_from_row expects, on historical bars.

    Must stay in step with what Agent 2 computes live - that is the whole
    reason both now read their score out of the shared scoring module rather
    than each implementing the rule.
    """
    close = df["Close"]
    df = df.copy()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    # One timeframe up - see scoring.component_values for why these spans.
    df["ema80"] = close.ewm(span=80, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()
    df["rsi14"] = calc_rsi(close)
    macd_line, signal_line, histogram = calc_macd(close)
    df["atr14"] = calc_atr(df)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram
    _, plus_di, minus_di = scoring.calc_adx(df)
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    # Lows and highs, matching agent2.analyze_pair. Rolling extremes of the
    # CLOSE put the entry price exactly on the range edge on any bar making new
    # ground, which is what sent 42% of entries into Agent 3's fallback branch.
    df["support"] = df["Low"].rolling(50).min()
    df["resistance"] = df["High"].rolling(50).max()
    return df


# The columns that must be present and non-NaN before a row can be scored.
# ema200 needs the longest warmup, which is why the decision window starts
# well past the old 60-bar offset.
REQUIRED_COLS = [
    "ema20", "ema50", "ema80", "ema200", "rsi14", "macd", "macd_signal",
    "macd_histogram", "atr14", "plus_di", "minus_di", "support", "resistance",
]


def simulate_trade(pair: str, hist: pd.DataFrame, entry_idx: int, action: str,
                    entry: float, tp1: float, tp2: float, tp3: float, sl0: float,
                    max_hold_bars: int, policy: str = DEFAULT_EXIT_POLICY):
    """Walk forward from entry_idx applying one of the EXIT_POLICIES.

    Checks each bar's High/Low (not just its Close) so intrabar wicks that
    would have triggered a real SL/TP fill aren't missed. When a single
    bar's range spans both the current stop and a take-profit level, the
    stop is assumed to hit first - the true intrabar order is unknown, and
    assuming the stop wins is the conservative (non-optimistic) choice.

    The reported r_multiple is position-weighted: a policy that closes a
    third at TP1 books a third of that tranche's R, so scaling out and
    holding to TP3 stay directly comparable on one number.
    """
    risk = abs(entry - sl0)
    if risk == 0:
        return None
    if policy not in EXIT_POLICIES:
        raise ValueError(f"unknown exit policy {policy!r}; expected one of {sorted(EXIT_POLICIES)}")

    def r_at(price):
        return (price - entry) / risk if action == "Buy" else (entry - price) / risk

    scales_out = policy.startswith("partial")
    tranche = 1.0 / 3.0 if scales_out else 1.0

    current_sl = sl0
    remaining = 1.0
    realized_r = 0.0
    tp1_hit = tp2_hit = tp3_hit = False

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
            realized_r += remaining * r_at(current_sl)
            return {"pair": pair, "outcome": "SL", "exit_price": float(current_sl),
                    "r_multiple": realized_r, "tp1": tp1_hit, "tp2": tp2_hit, "tp3": tp3_hit,
                    "exit_policy": policy}

        favorable = high if action == "Buy" else low
        reached = lambda level: (action == "Buy" and favorable >= level) or (action == "Sell" and favorable <= level)

        if not tp1_hit and reached(tp1):
            tp1_hit = True
            if scales_out:
                realized_r += tranche * r_at(tp1)
                remaining -= tranche
            if policy in ("trail-be", "partial-be"):
                current_sl = entry
        if tp1_hit and not tp2_hit and reached(tp2):
            tp2_hit = True
            if scales_out:
                realized_r += tranche * r_at(tp2)
                remaining -= tranche
            if policy == "trail-be":
                current_sl = tp1
            elif policy == "trail-late":
                current_sl = entry
        if tp2_hit and not tp3_hit and reached(tp3):
            tp3_hit = True
            realized_r += remaining * r_at(tp3)
            return {"pair": pair, "outcome": "TP3", "exit_price": float(tp3),
                    "r_multiple": realized_r, "tp1": tp1_hit, "tp2": tp2_hit, "tp3": tp3_hit,
                    "exit_policy": policy}

    # Day ended (next daily reset) without SL or TP3 - mark whatever is left
    # to market at the last close.
    exit_price = float(closes[end_idx])
    realized_r += remaining * r_at(exit_price)
    return {"pair": pair, "outcome": "TIMEOUT", "exit_price": exit_price,
            "r_multiple": realized_r, "tp1": tp1_hit, "tp2": tp2_hit, "tp3": tp3_hit,
            "exit_policy": policy}


def load_data(pairs, period, interval, delay, cache_path=None, refresh=False):
    """Download + compute indicators, memoised to disk.

    Yahoo Finance throttles hard and a full 9-pair pull takes minutes, which
    made it impractical to compare exit policies - each comparison meant
    another download of the identical bars. With the cache, the bars are
    fetched once and every subsequent experiment runs offline against exactly
    the same data, which is also what makes the comparison fair.
    """
    wanted_key = (tuple(sorted(p.upper() for p in pairs)), period, interval)
    if cache_path and not refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
        except Exception as e:
            print(f"[warn] cache at {cache_path} unreadable ({e}); re-downloading", file=sys.stderr)
        else:
            if cached.get("key") == wanted_key:
                bars = cached.get("bars")
                if bars is None:
                    # Pre-versioning cache: it holds frames with the indicator
                    # columns already computed. High/Low/Close are still in
                    # there, so the bars are salvageable - recompute rather
                    # than throw away a 20MB download over a column rename.
                    bars = {pair: df[["High", "Low", "Close"]].dropna()
                            for pair, df in cached.get("data", {}).items()}
                    print("[cache] pre-v2 cache: recomputing indicators from its bars")
                print(f"[cache] loaded {len(bars)} pairs of bars from {cache_path}")
                return {pair: compute_indicators(df) for pair, df in bars.items()}
            print("[cache] key mismatch (different pairs/period/interval); re-downloading")

    print(f"Downloading {interval} history ({period}) for {len(pairs)} pairs...")
    bars = {}
    for pair in pairs:
        symbol = DEFAULT_PAIRS.get(pair.upper())
        if not symbol:
            continue
        hist = download_history(pair, symbol, period, interval, delay=delay)
        if hist is None or len(hist) < 100:
            print(f"  [skip] {pair}: insufficient data")
            continue
        bars[pair] = hist
        print(f"  [ok] {pair}: {len(hist)} bars")

    if not bars:
        print("[error] no data downloaded for any pair - cannot backtest.", file=sys.stderr)
        sys.exit(1)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as f:
            # Raw bars only. Indicators are cheap to recompute and expensive to
            # get silently wrong; the bars are the part Yahoo rate-limits.
            pickle.dump({"key": wanted_key, "indicator_version": INDICATOR_VERSION,
                         "bars": bars}, f)
        print(f"[cache] saved to {cache_path}")

    return {pair: compute_indicators(df) for pair, df in bars.items()}


def replay(data, decision_every, max_hold_bars=None, atr_mult=DEFAULT_ATR_MULT,
           min_rr=DEFAULT_MIN_RR, min_score=DEFAULT_MIN_SCORE,
           policy=DEFAULT_EXIT_POLICY, use_ic_weights=False, quiet=False):
    """Run the decision rule over pre-loaded bars. Pure function of `data`.

    `decision_every` is how often a trade may be opened; `max_hold_bars` is how
    long it may be held. Passing one number for both is what made every
    previously-measured trade a four-hour trade - see the module docstring.
    """
    if max_hold_bars is None:
        max_hold_bars = scoring.BARS_PER_DAY
    # Align all pairs on a shared set of decision timestamps. The warmup is 250
    # bars rather than the old 60 because EMA200 is only meaningful after that.
    min_len = min(len(df) for df in data.values())
    decision_indices = list(range(250, min_len - max_hold_bars, decision_every))

    trades = []
    for idx in decision_indices:
        candidates = []
        for pair, df in data.items():
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            if row[REQUIRED_COLS].isna().any():
                continue
            score, values, _ = scoring.score_from_row(row, use_ic_weights=use_ic_weights)
            bias = scoring.bias_for_score(score, min_score)
            action = {"bullish": "Buy", "bearish": "Sell", "neutral": "Wait"}[bias]
            if action == "Wait":
                continue
            candidates.append((abs(score), pair, action, score, values, row))

        if not candidates:
            continue

        # Ties are now vanishingly unlikely - that is the point of a
        # continuous score. The old version was choosing essentially at
        # random among every pair that happened to sit at |score| == 2.
        candidates.sort(key=lambda c: c[0], reverse=True)
        _, pair, action, score, values, row = candidates[0]
        df = data[pair]
        entry = float(row["Close"])
        ndigits = decimals_for_pair(pair)
        atr = float(row["atr14"]) if not pd.isna(row["atr14"]) else None
        # No ATR, no stop, no trade - the same rule Agent 3 applies live, and
        # the reason build_trade_levels no longer has a fallback to invent one.
        levels = build_trade_levels(action, entry, ndigits, atr=atr, atr_mult=atr_mult)
        if not levels:
            continue

        # Constant by construction now (see scoring.py, TRADE GEOMETRY). Kept
        # as the same tripwire Agent 3 runs, so a geometry regression shows up
        # here as trades vanishing rather than as a quietly different strategy.
        rr1 = risk_reward(entry, levels["take_profit_1"], levels["stop_loss"])
        rr = risk_reward(entry, levels["take_profit_3"], levels["stop_loss"])
        if min_rr > 0 and (rr1 is None or rr1 < min_rr):
            continue

        result = simulate_trade(
            pair, df, idx, action, entry,
            levels["take_profit_1"], levels["take_profit_2"], levels["take_profit_3"],
            levels["stop_loss"], max_hold_bars, policy=policy,
        )
        if result:
            result["timestamp"] = str(df.index[idx])
            result["action"] = action
            result["score"] = round(score, 4)
            result["risk_reward"] = round(rr, 2)
            result["risk_reward_tp1"] = round(rr1, 2)
            # Per-component values are what validate.py computes the IC
            # against - without them there is no way to ask which indicator
            # is carrying the edge.
            result["components"] = {k: round(v, 4) for k, v in values.items()}
            trades.append(result)

    if not quiet:
        print(f"[replay] policy={policy} min_score={min_score} -> {len(trades)} trades")
    return trades


def run_backtest(pairs, period, interval, decision_every, delay, max_hold_bars=None,
                 atr_mult=DEFAULT_ATR_MULT, min_rr=DEFAULT_MIN_RR,
                 min_score=DEFAULT_MIN_SCORE, policy=DEFAULT_EXIT_POLICY,
                 cache_path=None, refresh=False, use_ic_weights=False):
    data = load_data(pairs, period, interval, delay, cache_path=cache_path, refresh=refresh)
    return replay(data, decision_every, max_hold_bars=max_hold_bars, atr_mult=atr_mult,
                  min_rr=min_rr, min_score=min_score, policy=policy,
                  use_ic_weights=use_ic_weights)


def expectancy_stats(trades) -> dict:
    """Expectancy with the uncertainty attached to it.

    Reporting a mean R without its t-statistic is what let a +0.014R result
    over 522 trades read as an edge. At that sample size the 95% interval was
    [-0.024, +0.053] - it contained zero, so the honest reading was "no
    measurable edge", not "a small one".
    """
    n = len(trades)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "se": 0.0, "t": 0.0,
                "ci": (0.0, 0.0), "win_rate": 0.0, "profit_factor": 0.0,
                "breakeven": 0, "total": 0.0, "trades_for_significance": None}

    rs = [t["r_multiple"] for t in trades]
    mean = sum(rs) / n
    sd = (sum((r - mean) ** 2 for r in rs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / (n ** 0.5) if sd > 0 else 0.0
    t_stat = mean / se if se > 0 else 0.0

    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    decided = len(wins) + len(losses)
    gross_loss = -sum(losses)

    # How many trades this edge would need before t reaches 2. If the answer
    # is larger than you will ever collect, the backtest cannot settle the
    # question no matter how the parameters are tuned.
    needed = int((2 * sd / mean) ** 2) if mean != 0 and sd > 0 else None

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "t": t_stat,
        "ci": (mean - 1.96 * se, mean + 1.96 * se),
        "win_rate": len(wins) / decided * 100 if decided else 0.0,
        "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else float("inf"),
        "breakeven": sum(1 for r in rs if r == 0),
        "total": sum(rs),
        "trades_for_significance": needed,
    }


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
    stats = expectancy_stats(trades)

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
    print(f"Avg R-multiple:    {avg_r:+.4f}R per trade")
    print(f"  95% CI:          [{stats['ci'][0]:+.4f}, {stats['ci'][1]:+.4f}]R   t = {stats['t']:+.2f}")
    # n is checked before t: with a handful of trades the t-statistic is not
    # merely insignificant, it is meaningless, and reporting "contains zero"
    # for a 3-trade run whose interval is [-1.0, -1.0] would be nonsense.
    if n < 30:
        print(f"  VERDICT:         sample too small to say anything ({n} trades).")
    elif abs(stats["t"]) < 2:
        need = stats["trades_for_significance"]
        print(f"  VERDICT:         NOT significant - the interval contains zero.")
        if need:
            print(f"                   An edge this small needs ~{need:,} trades to prove.")
    else:
        print(f"  VERDICT:         significant at this sample size (|t| >= 2).")
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


def sweep(data, decision_every, max_hold_bars, atr_mult, min_rr, min_score,
          use_ic_weights=False):
    """Every exit policy over identical bars, ranked by expectancy.

    Same entries, same levels, same data: the only thing that varies is what
    happens after the trade opens, so any difference in the numbers below is
    attributable to the exit rule alone.
    """
    rows = []
    for policy in EXIT_POLICIES:
        trades = replay(data, decision_every, max_hold_bars=max_hold_bars,
                        atr_mult=atr_mult, min_rr=min_rr,
                        min_score=min_score, policy=policy,
                        use_ic_weights=use_ic_weights, quiet=True)
        rows.append((policy, trades, expectancy_stats(trades)))

    rows.sort(key=lambda r: -r[2]["mean"])

    print("\n" + "=" * 78)
    print("EXIT POLICY COMPARISON  (identical entries, only the exit rule differs)")
    print("=" * 78)
    header = f"{'policy':<12} {'n':>5} {'exp R':>8} {'t':>6} {'win%':>7} {'PF':>6} {'0R':>6} {'netR':>8}"
    print(header)
    print("-" * 78)
    for policy, trades, s in rows:
        print(f"{policy:<12} {s['n']:>5} {s['mean']:>+8.4f} {s['t']:>+6.2f} "
              f"{s['win_rate']:>6.1f}% {s['profit_factor']:>6.2f} {s['breakeven']:>6} {s['total']:>+8.1f}")
    print("-" * 78)
    for policy in EXIT_POLICIES:
        print(f"  {policy:<12} {EXIT_POLICIES[policy]}")
    print("=" * 78)
    print("t is the expectancy's t-statistic. |t| < 2 means the result is not")
    print("distinguishable from zero at this sample size - a policy topping the")
    print("table on expectancy alone has not been shown to be better.")
    print("=" * 78)
    return rows


def build_parser():
    """Split out of main() so the defaults are testable.

    test_scoring.TestTimeframeIsShared asserts this parser's --interval
    default is scoring.DEFAULT_INTERVAL. That test is the thing that
    actually prevents the live/backtest drift; a comment claiming the two
    agree is what let them diverge in the first place.
    """
    parser = argparse.ArgumentParser(description="Backtest the Agent 2/3/5 Buy-Sell-TP-SL rule against history")
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS.keys()),
                         help=f"Pairs to include (default: all {len(DEFAULT_PAIRS)} in DEFAULT_PAIRS)")
    parser.add_argument("--period", default="730d", help="History window (yfinance format, e.g. 730d, 60d)")
    parser.add_argument("--interval", default=scoring.DEFAULT_INTERVAL,
                         help=f"Bar interval (default {scoring.DEFAULT_INTERVAL}). Shared with "
                              f"Agent 2 via scoring.DEFAULT_INTERVAL rather than asserted in a "
                              f"comment - that assertion is what silently went false before")
    parser.add_argument("--decision-every", type=int, default=scoring.decision_every_bars(),
                         help="Bars between decision points, matching the live cron "
                              "(scoring.LIVE_RUN_INTERVAL_HOURS)")
    parser.add_argument("--max-hold-bars", type=int, default=scoring.BARS_PER_DAY,
                         help="Maximum bars a trade is held, matching agent5.MAX_HOLD_HOURS. "
                              "This and --decision-every used to be one argument, which made "
                              "every measured trade a 4-hour one by accident")
    parser.add_argument("--delay", type=float, default=3.0,
                         help="Seconds to pause before each pair's download, to avoid Yahoo Finance rate limits")
    parser.add_argument("--out", default="backtest_trades.json", help="Where to save the raw trade log")
    parser.add_argument("--atr-mult", type=float, default=DEFAULT_ATR_MULT,
                         help="Place the stop-loss this many ATRs from entry (shared with Agent 3 "
                              "via scoring.SL_ATR_MULT)")
    parser.add_argument("--min-rr", type=float, default=DEFAULT_MIN_RR,
                         help="Tripwire on the constructed reward/risk to TP1 (0 disables). "
                              "Shared with Agent 3 via scoring.MIN_RR_TP1")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                         help="Minimum |score| for a setup to be tradable")
    parser.add_argument("--exit-policy", default=DEFAULT_EXIT_POLICY, choices=sorted(EXIT_POLICIES),
                         help="How an open trade is managed after entry")
    parser.add_argument("--sweep", action="store_true",
                         help="Compare every exit policy on identical bars instead of running just one")
    parser.add_argument("--cache", default=None,
                         help="Path to cache downloaded bars, so repeat experiments skip Yahoo entirely")
    parser.add_argument("--refresh", action="store_true", help="Ignore an existing cache and re-download")
    parser.add_argument("--use-ic-weights", action="store_true",
                         help="Score with the weights in ic_weights.json instead of equal weights")
    return parser


def main():
    args = build_parser().parse_args()

    data = load_data(args.pairs, args.period, args.interval, args.delay,
                     cache_path=args.cache, refresh=args.refresh)

    if args.sweep:
        rows = sweep(data, args.decision_every, args.max_hold_bars, args.atr_mult,
                     args.min_rr, args.min_score, use_ic_weights=args.use_ic_weights)
        # Persist the baseline policy's trades so validate.py has something to
        # read even after a sweep.
        trades = next(t for p, t, _ in rows if p == args.exit_policy)
    else:
        trades = replay(data, args.decision_every, max_hold_bars=args.max_hold_bars,
                        atr_mult=args.atr_mult, min_rr=args.min_rr,
                        min_score=args.min_score, policy=args.exit_policy,
                        use_ic_weights=args.use_ic_weights)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)

    summarize(trades)
    print(f"\nFull trade log written to {args.out}")


if __name__ == "__main__":
    main()
