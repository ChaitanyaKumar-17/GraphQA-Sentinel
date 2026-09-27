"""
.github/scripts/check_regression.py

CI gate (M7): compares the just-completed CI eval run's faithfulness
score against the immediately preceding CI run's score, both stored in
eval/ci_regression_log.json - a rolling log kept only in GitHub Actions
cache, separate from eval/regression_log.json (the git-tracked log the
Streamlit dashboard reads for baseline-vs-agentic history). Fails the
build if faithfulness dropped by more than the configured threshold.

On the very first CI run ever (no prior entry to compare against),
passes automatically - there's nothing to regress against yet.
"""

import argparse
import json
import sys
from pathlib import Path

LOG_PATH = Path("eval/ci_regression_log.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="faithfulness")
    parser.add_argument("--max-drop-percent", type=float, default=5.0)
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print(f"No CI regression log found at {LOG_PATH} - nothing to compare, passing.")
        return

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        log = json.load(f)

    if len(log) < 2:
        print(f"Only {len(log)} CI run(s) recorded so far - nothing to compare against yet, passing.")
        return

    current = log[-1]
    previous = log[-2]

    current_score = current.get(args.metric)
    previous_score = previous.get(args.metric)

    if current_score is None or previous_score is None:
        print(f"Missing '{args.metric}' score in one of the last two runs - skipping check.")
        return

    drop_points = (previous_score - current_score) * 100
    print(f"Previous {args.metric}: {previous_score:.4f}  |  Current: {current_score:.4f}  |  Drop: {drop_points:.2f} points")

    if drop_points > args.max_drop_percent:
        print(
            f"\nFAIL: {args.metric} dropped by {drop_points:.2f} points, "
            f"exceeding the allowed {args.max_drop_percent} point threshold."
        )
        sys.exit(1)

    print(f"\nPASS: {args.metric} is within the allowed {args.max_drop_percent} point drop threshold.")


if __name__ == "__main__":
    main()