"""
run_all.py - Runs the full forex signal pipeline end-to-end:
    Agent 1 (News Reader) -> Agent 2 (Technical Analyzer)
    -> Agent 3 (Signal Synthesizer) -> Agent 4 (LINE Notifier)

Designed to run as a single step in GitHub Actions (or locally).
Exits non-zero if any stage fails, so the Actions run shows as failed.
"""

import subprocess
import sys

STEPS = [
    ["python", "agent1_news_reader.py"],
    ["python", "agent2_technical_analyzer.py"],
    ["python", "agent3_signal_synthesizer.py"],
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
    print("\n=== All 4 agents completed successfully ===")


if __name__ == "__main__":
    main()
