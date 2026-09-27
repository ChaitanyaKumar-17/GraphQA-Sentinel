"""
eval/store.py

Append-only regression log for eval runs (M6). Each run_eval.py
execution appends one row summarizing that run's scores, tagged with a
timestamp, label (e.g. "baseline"/"agentic"), and git commit hash (if
available), so the Streamlit dashboard (dashboard/app.py) can plot
metric trends over time and catch regressions.

Unlike eval/results/*.json (one detailed file per run, with every
question's full answer and scores - large and regenerable, gitignored),
this log is small and deliberately kept in version control so the
project's eval history survives across machines and commits.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).parent / "regression_log.json"


def get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None  # not a git repo, git not installed, etc. - non-fatal


def load_log(path: Path = LOG_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_run(label: str, n_questions: int, summary: dict, path: Path = LOG_PATH) -> dict:
    overall = summary.get("overall", {})

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": get_git_commit_hash(),
        "label": label,
        "n_questions": n_questions,
        "faithfulness": overall.get("faithfulness"),
        "answer_relevancy": overall.get("answer_relevancy"),
        "context_precision": overall.get("context_precision"),
        "context_recall": overall.get("context_recall"),
    }

    log = load_log(path)
    log.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    return entry