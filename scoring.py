"""
scoring.py - The one place the technical bias score is defined.
================================================================
Agent 2 (live) and backtest.py (historical) used to each carry their own copy
of the scoring rule. That is how a live system and its backtest drift apart
without anyone noticing, so both now import from here.

WHY THIS REPLACED THE OLD +/-1 VOTE SCORE
-----------------------------------------
The old score was  EMA(+/-1) + MACD(+/-1) + RSI(+/-1 only outside 30/70)  and
Agent 3 only published a signal when |score| >= 2. Because the RSI term almost
never fired, "|score| >= 2" reduced to a single boolean: *do EMA and MACD
agree?*  Every one of the 522 trades in backtest_trades.json came out at
|score| == 2 exactly. Two consequences:

  1. possibility_percent, computed as 50 + |score| * 10, was the constant 70
     on every signal ever sent. It looked like a confidence number and carried
     no information.
  2. With no spread in the score there is nothing to validate. You cannot ask
     "do higher-scored setups win more often" of a variable that only ever
     takes one value, so there was no way to find out which indicator (if any)
     was carrying the edge.

Every component here is therefore CONTINUOUS and signed: positive = bullish,
negative = bearish, and the magnitude means something. That makes the decile
test and the per-component IC analysis in validate.py possible.

SIGN CONVENTIONS AND THE RSI DISAGREEMENT
-----------------------------------------
`rsi_dev` and `rsi_extreme` deliberately point in OPPOSITE directions:

  - rsi_dev     treats RSI as momentum       (RSI 80 -> bullish)
  - rsi_extreme treats RSI as mean-reversion (RSI 80 -> bearish)

The old code used the mean-reversion sign. Whether that was right was never
tested. Rather than silently pick one, both ship as separate components so
validate.py can measure each one's information coefficient against forward
returns and settle it with data.

IC WEIGHTS ARE OPT-IN ON PURPOSE
--------------------------------
validate.py writes ic_weights.json, but this module ignores that file unless
you explicitly pass use_ic_weights=True. The alpha_engine project this idea
came from applied its IC weights automatically, zeroed every component that
measured anti-predictive, and inverted its own scoring: the top-scored picks
went to a 9% win rate while the bottom-scored ones sat at 41%. The cause is
that component ICs are measured while all the OTHER components are active, so
a component that looks anti-predictive may simply be counterbalancing another.
Removing it changes the very dynamics the measurement assumed.

So: negative-IC components are HALVED, never zeroed, and nothing is applied
until you look at the decile report and turn it on yourself.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

# Component order is fixed - validate.py reports in this order and
# ic_weights.json is keyed by these names.
COMPONENTS = (
    "ema_sep",
    "macd_hist",
    "rsi_dev",
    "rsi_extreme",
    "di_spread",
    "htf_align",
    "range_pos",
)

# All-equal starting weights. These are a deliberate non-choice: hand-tuned
# weights are what the IC analysis is meant to replace, so the untuned score
# is the honest baseline to measure against.
DEFAULT_WEIGHTS = {c: 1.0 for c in COMPONENTS}

# Raw component values are squashed with tanh(x / SCALE) so a single wild
# reading cannot dominate the sum. SCALE sets what counts as a "strong" read:
# tanh(1) = 0.76, so a value equal to its scale lands at ~3/4 of full strength.
SCALES = {
    # EMA20-EMA50 separation measured in ATRs. Half an ATR of separation is
    # already a clearly established short-term trend.
    "ema_sep": 0.5,
    # MACD histogram in ATRs. Much smaller numbers than the EMA gap.
    "macd_hist": 0.15,
}

# The score is rescaled to +/-3 so it stays in the same range as the old
# -3..+3 vote score. Note that the ENDS of that range are not reachable in
# practice: it would take all seven components saturating at once, and
# rsi_extreme alone is 0 whenever RSI sits between 30 and 70. Re-measured over
# all 145,969 scored bars of 1h history across the nine pairs, after range_pos
# moved from close-only to low/high bounds:
#
#     p50 0.758   p75 1.218   p90 1.651   p95 1.792   p99 1.905   max 2.044
#
# which is why the threshold below is 1.2 and not 2. A 2.0 cut clears the 99th
# percentile by a distance - a handful of bars in two years, nothing to measure.
SCORE_SCALE = 3.0

# The 75th percentile of that distribution, on the grounds that it yields a
# trade count comparable to the old rule's 522 - i.e. picked for SAMPLE SIZE,
# deliberately not for returns. Tuning this number until the backtest looks
# good is precisely the overfitting the validate.py report exists to catch.
# It survived the range_pos change untouched rather than being re-tuned to fit:
# p75 came back at 1.218, so 1.2 is still the same percentile it always was.
DEFAULT_MIN_SCORE = 1.2

# ---------------------------------------------------------------------------
# TIMEFRAME - one definition, because the threshold below depends on it
# ---------------------------------------------------------------------------
# Agent 2 defaulted to 15m while backtest.py defaulted to 1h, and each file
# asserted in a comment that the other one agreed. Nothing checked, so the
# live system scored 15m bars against a threshold, a calibration table and a
# set of ICs that were every one of them measured on 1h bars.
#
# The interval is not a free parameter you can set per-script: DEFAULT_MIN_SCORE
# is the 75th percentile of the score distribution *on 1h bars*, and
# score_calibration.json maps |score| to a win rate observed *on 1h bars*.
# Score a 15m bar and those numbers describe a different system.
#
# 1h rather than 15m because Yahoo serves 730d of hourly history against only
# 60d of 15m - and the open question about this strategy is sample size, which
# a 60-day window cannot answer. The cost is real and worth stating: an hourly
# candle means the entry price on the LINE card can be up to an hour behind the
# market, where a 15m candle capped that at fifteen minutes.
DEFAULT_INTERVAL = "1h"

# History to pull on a live run. EMA200 is one of the components, so a window
# under 200 bars makes htf_align meaningless (agent2.MIN_BARS enforces this).
# Measured against Yahoo: 1h/10d returns ~238 bars - above the floor, but one
# long holiday weekend from dropping under it. 60d returns ~1,430 for the same
# single request.
DEFAULT_LIVE_PERIOD = "60d"

# One trading day of DEFAULT_INTERVAL bars. This is the backtest's hold window,
# and it mirrors agent5.MAX_HOLD_HOURS = 24 - so it has to be re-derived if the
# interval ever changes, which is why it lives here next to the interval and
# not as a bare 24 in backtest.py's argparse.
BARS_PER_DAY = 24

# How often the live pipeline gets to open a trade: the Actions cron is
# "7 */4 * * 1-5", so every 4 hours.
#
# This is a SEPARATE quantity from the hold window above, and backtest.py used
# to run both off one --bars-per-day argument. The checked-in calibration was
# generated with --bars-per-day 4 to match the 4-hourly cron, which silently
# also cut the maximum hold from 24 bars to 4: every one of the 1,312 trades
# behind score_calibration.json was a four-hour trade, measuring a system that
# holds for twenty-four. One parameter, two meanings, no way to notice.
LIVE_RUN_INTERVAL_HOURS = 4


def decision_every_bars(interval: str = None) -> int:
    """Bars between live decision points, for the backtest to replay."""
    minutes = interval_minutes(interval)
    if not minutes:
        return 1
    return max(1, (LIVE_RUN_INTERVAL_HOURS * 60) // minutes)

# ---------------------------------------------------------------------------
# TRADE GEOMETRY - where the stop and the targets go
# ---------------------------------------------------------------------------
# These used to live in agent3_signal_synthesizer.py, and the stop/target rule
# itself was: target = the 50-bar range edge, stop = the opposite edge, capped
# at ATR * 1.5. It was measured, and it did not work the way anyone assumed.
#
# WHAT THE MEASUREMENT SHOWED (502 entries, 1h bars, 730d, 9 pairs)
# -----------------------------------------------------------------
# support/resistance were the rolling 50-bar min/max OF THE CLOSE, and the
# entry price IS the latest close. A high |score| selects trend setups, and a
# trend setup's latest close is at or beyond its own 50-bar extreme - so the
# `support < entry < resistance` guard failed on 42% of entries, dropping them
# into a fallback branch documented as being for "thin/odd data" that placed
# the target at a flat 1.5% of price.
#
#     TP3 distance, structure branch: median 0.51 ATR   (range edge is CLOSE)
#     TP3 distance, fallback branch:  median 10.16 ATR  (1.5% of price is FAR)
#
# The reward/risk filter then finished the job. At min_rr 1.5 measured on TP3,
# 3 of 290 structure-branch setups survived and 212 of 212 fallback setups did:
#
#     98.6% of every trade this project has ever measured came out of the
#     fallback branch - a hardcoded percentage of price, with no relationship
#     to the pair, its volatility, or its structure.
#
# So backtest_trades.json, score_calibration.json and ic_weights.json were all
# describing an accident, and the "R:R 28.4" on the LINE cards was that accident
# quoted as a feature.
#
# WHAT REPLACED IT
# ----------------
# The stop is ATR-scaled and nothing else, and the targets are multiples of the
# risk. The range edge is no longer part of the geometry at all - at a median
# 0.51 ATR from entry it is not a usable stop or target reference, it is noise.
# Consequences, all deliberate:
#   - reward/risk is now a stated constant (1:1 to TP1, 3:1 to TP3) instead of
#     an accident of which branch fired, so it cannot select trades any more.
#     Agent 6's rule 3 checks the geometry is intact rather than gating on it.
#   - a signal with no ATR is not publishable. There is no honest way to size a
#     stop without a volatility estimate, and inventing one is what the
#     fallback branch did.
#
# WHY THESE NUMBERS AND NOT BETTER ONES
# -------------------------------------
# Max favorable and max adverse excursion over the 24-bar hold, in ATRs:
#
#     distance   0.5x    1.0x    1.5x    2.0x    2.5x    3.0x
#     favorable  87.3%   74.7%   63.9%   53.2%   41.4%   34.9%
#     adverse    87.8%   74.1%   63.1%   52.8%   42.2%   35.9%
#
# The two rows are the same row. At every distance price is as likely to reach
# the target as the stop, which is what "no measurable edge" looks like in the
# raw bars, and it means no choice of geometry manufactures one. SL_ATR_MULT is
# therefore kept at the 1.5 the old code already used - changing it would be
# picking a number the data does not support - and the targets are round
# R-multiples so the card can state the ratio honestly.
SL_ATR_MULT = 1.5

# TP1/TP2/TP3 as multiples of the risk. TP1 = 1R is the one most trades reach
# and the one Agent 6 gates on; TP3 = 3R is the stretch the trailing stop is
# working toward.
TP_R_MULTS = (1.0, 2.0, 3.0)

# Agent 3 built its levels with one number and Agent 6 gated on another that
# happened to share the value 1.5 while measuring a different quantity: Agent 3
# checked entry->TP3 and Agent 6 checked entry->TP1, which is TP3/3 by
# construction, so Agent 6's bar was silently three times the stricter. One
# definition now, measured entry->TP1, and both import it.
MIN_RR_TP1 = TP_R_MULTS[0]

# Above this, the levels are not merely generous - they are broken. The old
# fallback branch published cards reading "R:R 28.4"; anything near that means
# the geometry has come apart again, and Agent 6 blocks rather than warns.
MAX_RR_TP1 = TP_R_MULTS[0] * 3.0

# The stop may never be closer to the entry than this many ATRs. Nothing
# currently narrows it, but a stop that collapses onto the entry price is a
# trade that stops out on the spread, so the floor is enforced rather than
# assumed.
MIN_SL_ATR_MULT = 0.25


def interval_minutes(interval: str = None) -> Optional[int]:
    """Bar length in minutes, or None for anything unrecognised.

    Agent 6 needs this to judge whether the price on a card is stale: on 1h
    bars an entry can legitimately be up to an hour behind the market, and the
    same delay on 15m bars would mean the pipeline is running late. The limit
    has to be derived from the interval rather than typed in, because the
    interval has already moved once (15m -> 1h) and took every hardcoded
    assumption about freshness with it.
    """
    interval = (interval or DEFAULT_INTERVAL).strip().lower()
    match = re.fullmatch(r"(\d+)(m|h|d|wk|mo)", interval)
    if not match:
        return None
    size, unit = int(match.group(1)), match.group(2)
    factors = {"m": 1, "h": 60, "d": 60 * 24, "wk": 60 * 24 * 7, "mo": 60 * 24 * 30}
    return size * factors[unit]


IC_WEIGHTS_PATH = os.environ.get("IC_WEIGHTS_PATH", "ic_weights.json")


def calc_adx(df: pd.DataFrame, period: int = 14):
    """Wilder's ADX with its two directional indicators.

    Returns (adx, plus_di, minus_di). Only the DI spread feeds the score -
    ADX itself is unsigned trend strength, which cannot tell you which way to
    trade, while (+DI - -DI) is both signed and scaled to 0-100.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)

    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1 / period
    atr = true_range.ewm(alpha=alpha, adjust=False).mean()
    # Guard the division: a dead-flat window gives ATR 0 and would otherwise
    # produce inf DIs that poison every downstream mean.
    safe_atr = atr.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / safe_atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / safe_atr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx, plus_di, minus_di


def _finite(x, default=0.0) -> float:
    """Component values must always be real numbers.

    A NaN anywhere in the sum makes the whole score NaN, which silently drops
    the pair from ranking instead of just weakening its read - so an
    unavailable indicator contributes 0 (no opinion), not NaN.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def component_values(*, close: float, ema20: float, ema50: float, ema80: float,
                     ema200: float, rsi: Optional[float], macd_hist: float,
                     atr: Optional[float], plus_di: Optional[float],
                     minus_di: Optional[float], support: float,
                     resistance: float) -> dict:
    """Every component, each squashed into [-1, +1]. Positive = bullish."""
    vals = {}

    # ATR normalises price-unit distances so XAUUSD at 4591 and EURUSD at 1.15
    # produce comparable numbers. Without it, gold's raw EMA gap would swamp
    # every FX pair in the ranking.
    a = _finite(atr, 0.0)
    if a > 0:
        vals["ema_sep"] = math.tanh((ema20 - ema50) / a / SCALES["ema_sep"])
        vals["macd_hist"] = math.tanh(_finite(macd_hist) / a / SCALES["macd_hist"])
    else:
        vals["ema_sep"] = 0.0
        vals["macd_hist"] = 0.0

    rsi_val = _finite(rsi, float("nan"))
    if not math.isfinite(rsi_val):
        vals["rsi_dev"] = 0.0
        vals["rsi_extreme"] = 0.0
    else:
        # Momentum reading: distance from the 50 midline, full strength at 0/100.
        vals["rsi_dev"] = max(-1.0, min(1.0, (rsi_val - 50.0) / 50.0))
        # Mean-reversion reading: silent inside 30-70, then leans against the
        # move, reaching full strength at 0/100. This is the old rule's sign.
        if rsi_val > 70:
            vals["rsi_extreme"] = -min(1.0, (rsi_val - 70.0) / 30.0)
        elif rsi_val < 30:
            vals["rsi_extreme"] = min(1.0, (30.0 - rsi_val) / 30.0)
        else:
            vals["rsi_extreme"] = 0.0

    p, m = _finite(plus_di, float("nan")), _finite(minus_di, float("nan"))
    vals["di_spread"] = 0.0 if not (math.isfinite(p) and math.isfinite(m)) \
        else max(-1.0, min(1.0, (p - m) / 100.0))

    # One timeframe up, without resampling: EMA80/EMA200 on the same series
    # span roughly what EMA20/EMA50 would on 4x the bar size. Resampling would
    # be more literal but introduces bar-alignment bugs at every session edge.
    if a > 0:
        vals["htf_align"] = math.tanh((ema80 - ema200) / a / SCALES["ema_sep"])
    else:
        vals["htf_align"] = 0.0

    # Where price sits in its 50-bar range: -1 at support, +1 at resistance.
    # Signed as momentum (high in range = bullish). Whether that or the
    # contrarian reading is right is for the IC report to say.
    #
    # The range is the 50-bar low/high, not the rolling min/max of the CLOSE it
    # used to be. Against close-only bounds the entry price - itself the latest
    # close - sat exactly on one edge whenever the trend was making new ground,
    # so this component pinned to exactly +/-1 on 14.5% of bars (21,214 of
    # 146,029 measured) and carried no information on precisely the setups a
    # high score selects. Against low/high bounds that drops to 0.8%.
    span = resistance - support
    vals["range_pos"] = 0.0 if span <= 0 else \
        max(-1.0, min(1.0, 2.0 * (close - support) / span - 1.0))

    return {c: _finite(vals.get(c, 0.0)) for c in COMPONENTS}


def load_ic_weights(path: str = None) -> Optional[dict]:
    """Read validate.py's weights, or None when the file is absent/unusable."""
    path = path or IC_WEIGHTS_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    weights = data.get("weights")
    if not isinstance(weights, dict):
        return None
    return {c: float(weights.get(c, DEFAULT_WEIGHTS[c])) for c in COMPONENTS}


def compute_score(values: dict, weights: dict = None, use_ic_weights: bool = False):
    """Weighted, rescaled score in [-SCORE_SCALE, +SCORE_SCALE].

    Dividing by the total weight keeps the range fixed no matter how the
    weights are set, so a --min-score threshold does not quietly get stricter
    every time a component is down-weighted.
    """
    if weights is None:
        weights = (load_ic_weights() or DEFAULT_WEIGHTS) if use_ic_weights else DEFAULT_WEIGHTS

    total_w = sum(abs(weights.get(c, 0.0)) for c in COMPONENTS)
    if total_w <= 0:
        return 0.0, {c: 0.0 for c in COMPONENTS}

    contributions = {c: weights.get(c, 0.0) * values.get(c, 0.0) for c in COMPONENTS}
    score = SCORE_SCALE * sum(contributions.values()) / total_w
    return score, contributions


def bias_for_score(score: float, min_score: float) -> str:
    if score >= min_score:
        return "bullish"
    if score <= -min_score:
        return "bearish"
    return "neutral"


def score_from_row(row, weights=None, use_ic_weights: bool = False):
    """Score one indicator row (a pandas Series or plain dict).

    Shared by Agent 2's live path and backtest.py's historical replay so the
    two cannot diverge.
    """
    def get(key, default=None):
        try:
            v = row[key]
        except (KeyError, IndexError, TypeError):
            return default
        return default if v is None else v

    values = component_values(
        close=_finite(get("Close", get("last_close", 0.0))),
        ema20=_finite(get("ema20", 0.0)),
        ema50=_finite(get("ema50", 0.0)),
        ema80=_finite(get("ema80", 0.0)),
        ema200=_finite(get("ema200", 0.0)),
        rsi=get("rsi14", None),
        macd_hist=_finite(get("macd_histogram", get("macd_hist", 0.0))),
        atr=get("atr14", None),
        plus_di=get("plus_di", None),
        minus_di=get("minus_di", None),
        support=_finite(get("support", 0.0)),
        resistance=_finite(get("resistance", 0.0)),
    )
    score, contributions = compute_score(values, weights=weights, use_ic_weights=use_ic_weights)
    return score, values, contributions


def describe(values: dict, rsi: Optional[float] = None) -> list:
    """Human-readable reasons for the LINE card, strongest component first.

    Only components with a real opinion are mentioned - listing seven lines
    where four of them say "neutral" is how the old reasons list buried the
    one thing that actually drove the signal.
    """
    labels = {
        "ema_sep": ("EMA20 above EMA50", "EMA20 below EMA50"),
        "macd_hist": ("MACD histogram positive", "MACD histogram negative"),
        "rsi_dev": ("RSI above midline", "RSI below midline"),
        "rsi_extreme": ("RSI oversold (mean-reversion)", "RSI overbought (mean-reversion)"),
        "di_spread": ("+DI leads -DI (uptrend)", "-DI leads +DI (downtrend)"),
        "htf_align": ("higher timeframe trending up", "higher timeframe trending down"),
        "range_pos": ("price in upper half of 50-bar range", "price in lower half of 50-bar range"),
    }
    out = []
    for c, v in sorted(values.items(), key=lambda kv: -abs(kv[1])):
        if abs(v) < 0.05 or c not in labels:
            continue
        up, down = labels[c]
        out.append(f"{up if v > 0 else down} ({v:+.2f})")
    rsi_val = _finite(rsi, float("nan"))
    if math.isfinite(rsi_val):
        out.append(f"RSI {rsi_val:.1f}")
    return out or ["no component showed a directional read"]
