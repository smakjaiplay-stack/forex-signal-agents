"""
validate.py - Does the score actually predict anything?
========================================================
Reads a trade log (backtest_trades.json by default) and answers three
questions the pipeline could never answer about itself:

  1. EXPECTANCY, WITH ITS UNCERTAINTY.  A mean R-multiple on its own is not a
     result. The 522-trade log this was written for showed +0.014R, which
     reads like a small edge until you attach the interval: [-0.024, +0.053],
     t = 0.72. It contained zero. Proving an edge that size would have taken
     roughly 4,000 trades.

  2. THE DECILE TEST.  Sort every trade by the score that triggered it, cut
     into buckets, and check whether the high-scoring buckets actually did
     better than the low-scoring ones. A score that does not separate
     outcomes is decoration no matter how sophisticated it looks. This test
     was impossible on the old scoring rule, where all 522 trades came out at
     |score| == 2 exactly - one bucket, nothing to compare.

  3. PER-COMPONENT INFORMATION COEFFICIENT.  For each indicator, the Spearman
     correlation between "how strongly did this component agree with the
     direction we traded" and "what did the trade return". Components with
     positive IC carry the edge; components near zero are noise; components
     with negative IC are actively pointing the wrong way.

WHY THE WEIGHTS THIS WRITES ARE NOT APPLIED AUTOMATICALLY
---------------------------------------------------------
ic_weights.json is written but ignored until you pass --use-ic-weights to
Agent 2 or backtest.py. The project this technique is borrowed from applied
its IC weights automatically and zeroed every anti-predictive component. Its
scoring inverted: top-scored picks fell to a 9% win rate while bottom-scored
picks sat at 41%. Each IC is measured while all the other components are
live, so a component that looks anti-predictive may be counterbalancing
another one, and deleting it changes the very system the measurement
described.

This script therefore HALVES negative-IC components rather than zeroing them,
and leaves the decision to switch them on to a human who has read the report.

Usage:
    python validate.py
    python validate.py --trades backtest_trades.json --buckets 10
    python validate.py --min-bucket 20 --no-write
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from collections import defaultdict

import scoring

# Windows terminals default to cp874/cp1252 here and choke on the box-drawing
# characters in the report tables.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DEFAULT_TRADES = "backtest_trades.json"
IC_WEIGHTS_OUT = "ic_weights.json"
CALIBRATION_OUT = "score_calibration.json"

# A component's IC is not worth reporting on a handful of trades - below this
# the correlation is dominated by whichever way three or four trades went.
MIN_TRADES_FOR_IC = 30
# Weight given to a component whose measured IC is <= 0. Halved, never zeroed
# - see the module docstring for what zeroing did to the system this came from.
NEGATIVE_IC_WEIGHT = 0.5

# Mirrors agent6_qc_reviewer.DEFAULT_MIN_POSSIBILITY. Imported lazily below so
# this script keeps working if Agent 6 is absent or renamed.
try:
    from agent6_qc_reviewer import DEFAULT_MIN_POSSIBILITY as AGENT6_MIN_POSSIBILITY
except ImportError:
    AGENT6_MIN_POSSIBILITY = 60


# ---------------------------------------------------------------------------
# Statistics (stdlib only, so this runs anywhere the pipeline does)
# ---------------------------------------------------------------------------

def _ranks(values):
    """Fractional ranks, averaging ties.

    Tie-averaging matters here: a component that is 0.0 on half the trades
    (an indicator with no opinion) would otherwise have those trades ranked
    in arbitrary input order, manufacturing correlation out of file ordering.
    """
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation. Returns 0.0 when undefined."""
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def corr_t_stat(r, n):
    """t-statistic for a correlation coefficient."""
    if n < 3 or abs(r) >= 1.0:
        return 0.0
    return r * math.sqrt((n - 2) / (1 - r * r))


def mean_stats(values):
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "se": 0.0, "t": 0.0, "ci": (0.0, 0.0)}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "t": mean / se if se else 0.0,
        "ci": (mean - 1.96 * se, mean + 1.96 * se),
    }


def bootstrap_ci(values, iterations=5000, seed=7):
    """Percentile bootstrap for the mean.

    The t-interval assumes a roughly normal sampling distribution. R-multiples
    are anything but - they pile up at exactly -1 and at exactly 0 - so the
    bootstrap is the interval to trust when the two disagree.
    """
    n = len(values)
    if n < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trades(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[error] could not read {path}: {e}", file=sys.stderr)
        sys.exit(1)

    trades = data if isinstance(data, list) else data.get("trades", [])
    trades = [t for t in trades if isinstance(t, dict) and "r_multiple" in t]
    if not trades:
        print(f"[error] no usable trades in {path}", file=sys.stderr)
        sys.exit(1)
    return trades


def direction(trade):
    """+1 for a long, -1 for a short.

    Every component is expressed as bullish-positive, while r_multiple is
    already relative to the direction traded. Multiplying by this puts them in
    the same frame: positive means "this component agreed with the trade".
    """
    return 1.0 if trade.get("action") == "Buy" else -1.0


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def payoff_stats(trades):
    """Win rate, payoff ratio, and the win rate at which the two cancel out.

    Breakeven is 1 / (1 + payoff): at a 1.14:1 payoff you need 46.7% winners
    just to stand still, while at 2.2:1 you need only 31.2%. This is why a
    fixed win-rate floor is not a risk control - the same 60% bar is trivially
    loose for one payoff and unreachable for another, and nothing about the
    number itself tells you which case you are in.
    """
    rs = [float(t["r_multiple"]) for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    decided = len(wins) + len(losses)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff = avg_win / abs(avg_loss) if avg_loss else float("inf")
    return {
        "win_rate": len(wins) / decided * 100 if decided else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff,
        # A payoff of inf (no losses at all) means any win rate breaks even;
        # report 0 rather than a division by infinity.
        "breakeven": 0.0 if payoff == float("inf") else 100.0 / (1.0 + payoff),
    }


def report_expectancy(trades):
    rs = [float(t["r_multiple"]) for t in trades]
    s = mean_stats(rs)
    lo, hi = bootstrap_ci(rs)

    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    flat = [r for r in rs if r == 0]
    decided = len(wins) + len(losses)
    gross_loss = -sum(losses)

    print("=" * 78)
    print("1. EXPECTANCY")
    print("=" * 78)
    print(f"  trades            {s['n']}")
    print(f"  expectancy        {s['mean']:+.4f}R   (sd {s['sd']:.3f})")
    print(f"  t-interval 95%    [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]   t = {s['t']:+.2f}")
    print(f"  bootstrap 95%     [{lo:+.4f}, {hi:+.4f}]")
    print(f"  net               {sum(rs):+.1f}R")
    print(f"  win rate          {len(wins) / decided * 100 if decided else 0:.1f}%  "
          f"({len(wins)}W / {len(losses)}L, {len(flat)} flat at exactly 0R)")
    if wins:
        print(f"  avg win           {sum(wins) / len(wins):+.3f}R")
    if losses:
        print(f"  avg loss          {sum(losses) / len(losses):+.3f}R")
    print(f"  profit factor     {(sum(wins) / gross_loss) if gross_loss > 0 else float('inf'):.3f}")

    # A quarter of trades landing on exactly 0R is not a rounding artifact -
    # it is a stop that was moved to the entry price and then hit.
    if s["n"] and len(flat) / s["n"] > 0.15:
        print()
        print(f"  NOTE: {len(flat) / s['n']:.1%} of trades closed at exactly 0R. That is the")
        print(f"        breakeven trail firing. Compare exit rules with:")
        print(f"          python backtest.py --sweep --cache .cache/bars.pkl")

    print()
    if s["n"] < 30:
        # With a handful of trades the t-statistic is meaningless rather than
        # merely insignificant - saying "contains zero" about a 3-trade run
        # whose interval is [-1.0, -1.0] would be nonsense.
        print(f"  VERDICT: SAMPLE TOO SMALL ({s['n']} trades). No verdict is possible.")
    elif abs(s["t"]) < 2:
        need = int((2 * s["sd"] / s["mean"]) ** 2) if s["mean"] else None
        print("  VERDICT: NO MEASURABLE EDGE. The interval contains zero, so this")
        print("           result is consistent with the strategy having no edge at all.")
        if need:
            print(f"           An edge this size would need ~{need:,} trades to confirm.")
    else:
        print("  VERDICT: edge is statistically distinguishable from zero at this")
        print("           sample size. That is not the same as it surviving live costs -")
        print("           spread and slippage are not modelled here.")
    print()
    return s


def report_deciles(trades, n_buckets, min_bucket):
    """Do higher-scored setups actually do better?"""
    scored = [t for t in trades if t.get("score") is not None]
    print("=" * 78)
    print(f"2. DECILE TEST  ({n_buckets} buckets by |score|)")
    print("=" * 78)

    if len(scored) < n_buckets * min_bucket:
        print(f"  [skip] need >= {n_buckets * min_bucket} scored trades, have {len(scored)}")
        print()
        return None

    distinct = {round(abs(float(t["score"])), 6) for t in scored}
    if len(distinct) < n_buckets:
        print(f"  [warn] |score| takes only {len(distinct)} distinct value(s) across "
              f"{len(scored)} trades.")
        if len(distinct) == 1:
            print("         The score is a constant. There is nothing to separate: it")
            print("         cannot be ranking anything. This is exactly the condition")
            print("         scoring.py was rewritten to fix - re-run backtest.py.")
            print()
            return None
        print("         Buckets will be uneven and the trend below is weak evidence.")

    scored.sort(key=lambda t: abs(float(t["score"])))
    size = len(scored) / n_buckets
    buckets = []
    for i in range(n_buckets):
        chunk = scored[int(i * size):int((i + 1) * size)]
        if not chunk:
            continue
        rs = [float(t["r_multiple"]) for t in chunk]
        ss = [abs(float(t["score"])) for t in chunk]
        wins = sum(1 for r in rs if r > 0)
        decided = sum(1 for r in rs if r != 0)
        buckets.append({
            "bucket": i + 1,
            "n": len(chunk),
            "score_min": min(ss),
            "score_max": max(ss),
            "score_mid": sum(ss) / len(ss),
            "win_rate": wins / decided * 100 if decided else 0.0,
            "avg_r": sum(rs) / len(rs),
            "total_r": sum(rs),
        })

    print(f"  {'#':>3} {'n':>5} {'|score| range':>18} {'win%':>8} {'avg R':>9} {'net R':>8}")
    print("  " + "-" * 60)
    for b in buckets:
        print(f"  {b['bucket']:>3} {b['n']:>5} "
              f"{b['score_min']:>8.3f}-{b['score_max']:<9.3f} "
              f"{b['win_rate']:>7.1f}% {b['avg_r']:>+9.4f} {b['total_r']:>+8.1f}")
    print("  " + "-" * 60)

    rho_r = spearman([b["bucket"] for b in buckets], [b["avg_r"] for b in buckets])
    rho_w = spearman([b["bucket"] for b in buckets], [b["win_rate"] for b in buckets])
    # Trade-level IC is the number that matters; the bucket-level rho is only
    # 10 points and moves a lot on one noisy bucket.
    ic = spearman([abs(float(t["score"])) for t in scored],
                  [float(t["r_multiple"]) for t in scored])
    t_ic = corr_t_stat(ic, len(scored))

    print(f"  bucket rho (avg R)   {rho_r:+.3f}")
    print(f"  bucket rho (win%)    {rho_w:+.3f}")
    print(f"  trade-level IC       {ic:+.4f}   t = {t_ic:+.2f}  (n = {len(scored)})")
    print()

    if abs(t_ic) < 2:
        verdict = "WEAK"
        note = ("The score does not separate outcomes at this sample size. Ranking "
                "setups by it is close to ranking them at random.")
    elif ic > 0 and rho_r > 0.5:
        verdict = "STRONG"
        note = "Higher-scored setups genuinely performed better, monotonically."
    elif ic > 0:
        verdict = "MODERATE"
        note = "Positive but not monotone - the top bucket is doing the work."
    else:
        verdict = "BROKEN (INVERTED)"
        note = ("Higher-scored setups did WORSE. Do not simply flip the sign - a "
                "component is pointing the wrong way; find it in section 3.")
    print(f"  VERDICT: {verdict}")
    print(f"           {note}")
    print()
    return buckets


def report_components(trades, min_trades):
    """Which indicator is actually carrying the edge, if any."""
    print("=" * 78)
    print("3. PER-COMPONENT INFORMATION COEFFICIENT")
    print("=" * 78)

    with_components = [t for t in trades if isinstance(t.get("components"), dict)]
    if len(with_components) < min_trades:
        print(f"  [skip] only {len(with_components)} trades carry component data "
              f"(need {min_trades}).")
        print("         Re-run backtest.py - it records components on every trade now.")
        print()
        return None

    rs = [float(t["r_multiple"]) for t in with_components]
    results = []
    for comp in scoring.COMPONENTS:
        # component * direction = "how strongly did this agree with the trade
        # we actually took", which is the thing that could predict the return.
        agree = [float(t["components"].get(comp, 0.0)) * direction(t) for t in with_components]
        if len({round(a, 6) for a in agree}) < 3:
            results.append((comp, 0.0, 0.0, len(agree), "constant - no opinion"))
            continue
        ic = spearman(agree, rs)
        results.append((comp, ic, corr_t_stat(ic, len(agree)), len(agree), ""))

    results.sort(key=lambda r: -r[1])

    print(f"  {'component':<14} {'IC':>9} {'t':>7} {'n':>6}  reading")
    print("  " + "-" * 66)
    for comp, ic, t_stat, n, note in results:
        if note:
            reading = note
        elif abs(t_stat) < 2:
            reading = "noise (not distinguishable from zero)"
        elif ic > 0:
            reading = "PREDICTIVE"
        else:
            reading = "ANTI-PREDICTIVE - points the wrong way"
        print(f"  {comp:<14} {ic:>+9.4f} {t_stat:>+7.2f} {n:>6}  {reading}")
    print("  " + "-" * 66)

    # Seven components tested at once means roughly one in three runs will
    # throw up a |t| >= 2 by chance alone. Saying so here is cheaper than
    # rebuilding the strategy around a coincidence.
    tested = len(results)
    hits = [r for r in results if abs(r[2]) >= 2 and not r[4]]
    print(f"  {tested} components tested. At |t| >= 2 you expect ~{tested * 0.05:.1f} false")
    print(f"  positive(s) per run by chance; a Bonferroni-corrected bar would be |t| ~2.7.")
    if hits:
        print(f"  {len(hits)} cleared |t| >= 2: {', '.join(r[0] for r in hits)}")
        print("  Treat these as leads to test on fresh data, not as settled findings.")
    print()
    return results


def build_weights(results):
    """Halve the negative-IC components. Never zero them.

    Zeroing is what inverted the scoring in the project this idea came from:
    each IC was measured with every other component active, so a component
    reading anti-predictive may be counterbalancing another rather than
    hurting on its own. Halving reduces its influence without removing the
    counterweight.
    """
    weights = {}
    for comp, ic, t_stat, n, note in results:
        weights[comp] = 1.0 if ic > 0 else NEGATIVE_IC_WEIGHT
    return weights


def main():
    parser = argparse.ArgumentParser(description="Validate that the score predicts anything")
    parser.add_argument("--trades", default=DEFAULT_TRADES, help="Trade log to analyse")
    parser.add_argument("--buckets", type=int, default=10, help="Buckets for the decile test")
    parser.add_argument("--min-bucket", type=int, default=15,
                         help="Minimum trades per bucket before the decile test runs")
    parser.add_argument("--min-trades-ic", type=int, default=MIN_TRADES_FOR_IC,
                         help="Minimum trades before per-component ICs are reported")
    parser.add_argument("--no-write", action="store_true",
                         help="Print the report without writing ic_weights.json / score_calibration.json")
    args = parser.parse_args()

    trades = load_trades(args.trades)
    print()
    print(f"Validating {len(trades)} trades from {args.trades}")
    print()

    stats = report_expectancy(trades)
    buckets = report_deciles(trades, args.buckets, args.min_bucket)
    results = report_components(trades, args.min_trades_ic)

    if args.no_write:
        print("[--no-write] nothing written.")
        return

    if results:
        weights = build_weights(results)
        with open(IC_WEIGHTS_OUT, "w", encoding="utf-8") as f:
            json.dump({
                "source": args.trades,
                "trades": len(trades),
                "note": ("Negative-IC components are halved, not zeroed. NOT applied "
                         "unless you pass --use-ic-weights; read scoring.py first."),
                "ic": {c: round(ic, 5) for c, ic, _, _, _ in results},
                "weights": weights,
            }, f, ensure_ascii=False, indent=2)
        print(f"[write] {IC_WEIGHTS_OUT}  (inactive until --use-ic-weights)")

    if buckets:
        payoff = payoff_stats(trades)
        with open(CALIBRATION_OUT, "w", encoding="utf-8") as f:
            json.dump({
                "source": args.trades,
                "trades": len(trades),
                "note": ("Maps |score| to the win rate AND the average R observed at "
                         "that score. Agent 3 reads this so possibility_percent is a "
                         "measurement rather than a linear guess; Agent 6 reads "
                         "breakeven_win_rate so its confidence floor is derived from "
                         "the measured payoff instead of being a hand-picked number."),
                # A win rate on its own cannot say whether a trade makes money -
                # that depends on what the wins pay relative to what the losses
                # cost. Agent 6's floor used to be a flat 60%, which on this
                # payoff demands 13 points above breakeven from a strategy whose
                # best bucket reaches 52%. The floor belongs here, derived, not
                # there, invented.
                "win_rate": round(payoff["win_rate"], 2),
                "avg_win_r": round(payoff["avg_win"], 4),
                "avg_loss_r": round(payoff["avg_loss"], 4),
                "payoff_ratio": round(payoff["payoff"], 4),
                "breakeven_win_rate": round(payoff["breakeven"], 2),
                # Whether the measured expectancy is distinguishable from zero at
                # all. Agent 6 refuses to publish while this is false, and starts
                # publishing on its own once a future calibration makes it true -
                # so nobody has to remember to flip a switch.
                "expectancy_r": round(stats["mean"], 4),
                "expectancy_t": round(stats["t"], 3),
                "expectancy_ci": [round(stats["ci"][0], 4), round(stats["ci"][1], 4)],
                "edge_significant": bool(stats["n"] >= 30 and abs(stats["t"]) >= 2
                                         and stats["mean"] > 0),
                "buckets": [
                    {"score_mid": round(b["score_mid"], 4),
                     "win_rate": round(b["win_rate"], 2),
                     # Expectancy in the bucket. Reported, but deliberately NOT
                     # used as a per-bucket gate: with a WEAK decile verdict the
                     # sign of any single bucket's avg R is sampling noise, and
                     # selecting the buckets that happened to come out positive
                     # is textbook overfitting.
                     "avg_r": round(b["avg_r"], 4),
                     "n": b["n"]}
                    for b in buckets
                ],
            }, f, ensure_ascii=False, indent=2)
        print(f"[write] {CALIBRATION_OUT}  (Agent 3 picks this up automatically)")

        # What this calibration means for the QC gate, said out loud rather
        # than left for someone to discover when LINE goes quiet.
        print()
        print("=" * 78)
        print("4. WHAT AGENT 6 WILL DO WITH THIS")
        print("=" * 78)
        best = max(b["win_rate"] for b in buckets)
        print(f"  payoff ratio          {payoff['payoff']:.2f} : 1")
        print(f"  breakeven win rate    {payoff['breakeven']:.1f}%   <- Agent 6's derived floor")
        print(f"  best bucket win rate  {best:.1f}%")
        print(f"  expectancy            {stats['mean']:+.4f}R   t = {stats['t']:+.2f}")
        print()
        if not (stats["n"] >= 30 and abs(stats["t"]) >= 2 and stats["mean"] > 0):
            print("  edge_significant: FALSE -> Agent 6 forces every signal to Wait.")
            print()
            print("  This is the gate working, not the gate misfiring. The expectancy")
            print("  interval contains zero, so there is no score level at which a")
            print("  positive-expectancy setup can honestly be claimed. Picking the")
            print("  buckets that happened to print a positive avg R would be fitting")
            print("  the noise - the decile verdict above says the score does not")
            print("  separate outcomes at this sample size.")
            print()
            print("  To collect live samples anyway, run Agent 6 with")
            print("  --allow-unproven-edge. That publishes, stamps every card")
            print("  'UNPROVEN EDGE', and is a decision to gather data - not a fix.")
            print("  The gate re-arms itself the moment a calibration measures a")
            print("  real edge; nothing needs to be switched back.")
        else:
            print(f"  edge_significant: TRUE -> Agent 6 publishes signals whose")
            print(f"  possibility clears the {payoff['breakeven']:.1f}% breakeven floor.")
        if AGENT6_MIN_POSSIBILITY != round(payoff["breakeven"]):
            print()
            print(f"  (agent6.DEFAULT_MIN_POSSIBILITY = {AGENT6_MIN_POSSIBILITY} is now only the fallback for")
            print(f"   runs with no calibration file at all, where possibility_percent")
            print(f"   is still the uncalibrated linear guess.)")


if __name__ == "__main__":
    main()
