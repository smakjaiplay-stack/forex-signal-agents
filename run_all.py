"""
run_all.py - Runs the full forex signal pipeline end-to-end:
    Agent 1 (News Reader) -> Agent 2 (Technical Analyzer)
    -> Agent 3 (Signal Synthesizer) -> Agent 6 (QC Reviewer)
    -> Agent 4 (LINE Notifier)

Agent 6 sits between synthesis and notification on purpose: nothing reaches
LINE until the QC layer has vetted it. Note that Agent 6 currently runs with
--allow-unproven-edge; see the comment on its STEPS entry for what that costs
and the condition under which it should be removed.

Designed to run as a single step in GitHub Actions (or locally).
Exits non-zero if any stage fails, so the Actions run shows as failed.
"""

import subprocess
import sys

STEPS = [
    ["python", "agent1_news_reader.py"],
    ["python", "agent2_technical_analyzer.py"],
    ["python", "agent3_signal_synthesizer.py"],
    # --allow-unproven-edge: publish even though the backtest measures no edge.
    #
    # This is switched on deliberately, and it is not a fix. As of the current
    # score_calibration.json the measured expectancy is -0.0357R with a 95%
    # interval of [-0.0842, +0.0128] over 1,312 trades - the interval contains
    # zero, so QC's rule 2 would otherwise force every signal to Wait, and it
    # would be right to. The flag overrides that in order to accumulate FORWARD
    # samples, which is the only kind this project does not already have: every
    # number above was measured on the same historical bars the rule was built
    # against.
    #
    # Every card goes out stamped "UNPROVEN EDGE" so nobody downstream mistakes
    # these for validated signals. Size positions accordingly, or paper-trade.
    #
    # WHEN TO REMOVE THIS: re-run backtest.py and validate.py once live outcomes
    # have accumulated in trades_log.jsonl. If a calibration ever measures a real
    # edge, edge_significant flips true and QC starts gating on the breakeven win
    # rate by itself - at which point this flag becomes a no-op and should be
    # deleted. If the forward samples instead confirm no edge, the honest move is
    # to delete the flag and let the pipeline go quiet.
    ["python", "agent6_qc_reviewer.py", "--allow-unproven-edge"],
    ["python", "agent4_line_notifier.py"],
]


def run_step(cmd):
    print(f"\n=== Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[error] step failed: {' '.join(cmd)} (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    for step in STEPS:
        run_step(step)
    print(f"\n=== All {len(STEPS)} agents completed successfully ===")


if __name__ == "__main__":
    main()
