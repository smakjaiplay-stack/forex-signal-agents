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
# rsi_extreme alone is 0 whenever RSI sits between 30 and 70. Measured over
# 5,049 scored bars of 1h history the distribution came out:
#
#     p50 0.79   p75 1.27   p90 1.71   p95 1.84   max 2.02
#
# which is why the threshold below is 1.2 and not 2. A 2.0 cut let through
# 4 bars out of 5,049 - three trades in two years, and nothing to measure.
SCORE_SCALE = 3.0

# Chosen at roughly the 75th percentile of that distribution, on the grounds
# that it yields a trade count comparable to the old rule's 522 - i.e. picked
# for SAMPLE SIZE, deliberately not for returns. Tuning this number until the
# backtest looks good is precisely the overfitting the validate.py report
# exists to catch.
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
