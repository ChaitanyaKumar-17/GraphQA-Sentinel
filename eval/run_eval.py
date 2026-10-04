"""
eval/run_eval.py

Runs the golden set through a pipeline (baseline or agentic) and scores
every answer with four RAGAS metrics.

Built for free-tier quotas: progress is saved after every single step to
eval/results/<label>.json. If a daily token cap is hit the run stops
cleanly, and re-running the same command picks up exactly where it left
off (answers already generated and metrics already scored are kept).

Usage:
    python -m eval.run_eval --label baseline --pipeline baseline --sample 12
    python -m eval.run_eval --label agentic  --pipeline agentic  --sample 12

    --phase generate   only produce answers (uses the app LLM's quota)
    --phase score      only judge saved answers (uses the judge's quota)
    --sample N         stratified sample across categories; 0 = all questions.
                       Samples are nested: a bigger N always includes the
                       questions of a smaller N, so you can extend a finished
                       run later without redoing anything.
    --fresh            discard saved progress for this label
    --retry-failed     clear recorded failures so failed metrics are attempted again
    --no-log           don't append the result to the regression log (smoke tests)

Exit codes: 0 complete, 1 unreliable run (not logged), 2 incomplete (re-run).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"
CATEGORIES = ["factual", "multi_hop", "out_of_scope"]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

MAX_METRIC_FAILURES = 3      # after this many failed attempts a metric is given up on
MIN_SUCCESS_RATE = 0.7       # below this share of scored metric values, the run is unreliable
GENERATION_DELAY_SECONDS = 3
SCORING_DELAY_SECONDS = 6   # pause between judge calls to stay under free-tier TPM cap
NO_CONTEXT_PLACEHOLDER = "No relevant documentation was retrieved."


# --------------------------------------------------------------------------
# Quota detection
# --------------------------------------------------------------------------

def is_daily_quota_error(exc: BaseException) -> bool:
    """True if this error (or anything it wraps) is a Groq per-day limit."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = str(exc)
        if "DAILY_QUOTA_EXHAUSTED" in text or "tokens per day" in text or "requests per day" in text:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# --------------------------------------------------------------------------
# Question sampling
# --------------------------------------------------------------------------

def spread_order(n: int) -> list[int]:
    """Orders 0..n-1 so that any prefix is spread evenly across the range."""
    order, seen, j = [], set(), 1
    while len(order) < n:
        idx = int(((j * 0.6180339887498949) % 1.0) * n)
        if idx not in seen:
            seen.add(idx)
            order.append(idx)
        j += 1
    return order


def select_ids(golden_set: list[dict], sample: int) -> list[int]:
    if not sample or sample >= len(golden_set):
        return list(range(len(golden_set)))
    picked = []
    for pos, category in enumerate(CATEGORIES):
        ids = [i for i, q in enumerate(golden_set) if q["category"] == category]
        count = sample // len(CATEGORIES) + (1 if pos < sample % len(CATEGORIES) else 0)
        picked += [ids[k] for k in spread_order(len(ids))[:count]]
    return sorted(picked)


# --------------------------------------------------------------------------
# Saved state
# --------------------------------------------------------------------------

def load_state(path: Path, label: str, pipeline: str, fresh: bool) -> dict:
    if path.exists() and not fresh:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("pipeline") != pipeline:
            sys.exit(
                f"{path.name} holds a '{state.get('pipeline')}' run, not '{pipeline}'. "
                f"Use a different --label, or pass --fresh to start over."
            )
        return state
    return {
        "label": label,
        "pipeline": pipeline,
        "created": datetime.now(timezone.utc).isoformat(),
        "logged": False,
        "items": [],
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def ensure_items(state: dict, golden_set: list[dict], ids: list[int]) -> None:
    existing = {item["id"] for item in state["items"]}
    for i in ids:
        if i not in existing:
            q = golden_set[i]
            state["items"].append({
                "id": i,
                "question": q["question"],
                "category": q["category"],
                "reference": q["expected_answer"],
                "expected_sources": q["expected_sources"],
                "answer": None,
                "contexts": [],
                "sources": [],
                "meta": {},
                "scores": {},
                "failures": {},
            })
    state["items"].sort(key=lambda it: it["id"])


def missing_metrics(item: dict) -> list[str]:
    return [
        name for name in METRIC_NAMES
        if name not in item["scores"] and item["failures"].get(name, 0) < MAX_METRIC_FAILURES
    ]


def has_pending_work(items: list[dict]) -> bool:
    return any(it["answer"] is None or missing_metrics(it) for it in items)


def metric_coverage(items: list[dict]) -> dict[str, int]:
    """How many questions each metric has a real (non-None) score for.

    Checked per metric, not pooled, because a run where one metric fails
    on every question but the other three are perfect still has 75% of
    all values filled in - a pooled ratio would call that reliable and
    log it, silently missing the one metric CI actually watches.
    """
    return {m: sum(1 for it in items if it["scores"].get(m) is not None) for m in METRIC_NAMES}


# --------------------------------------------------------------------------
# Phase 1: generate answers
# --------------------------------------------------------------------------

def run_pipeline(pipeline: str, question: str):
    """Returns (answer, contexts, sources, meta). Retries per-minute rate limits."""
    for attempt in range(1, 4):
        try:
            if pipeline == "baseline":
                from api.rag_pipeline import generate_answer, retrieve
                chunks = retrieve(question)
                answer = generate_answer(question, chunks)
                contexts = [c["text"] for c in chunks]
                sources = sorted({c["url"] for c in chunks})
                meta = {}
            else:
                from agent.graph import answer_question_agentic
                result = answer_question_agentic(question)
                answer = result["answer"]
                contexts = result["contexts"]
                sources = result["sources"]
                meta = {
                    "retry_count": result["retry_count"],
                    "used_web_fallback": result["used_web_fallback"],
                    "self_check_passed": result["self_check_passed"],
                }
            if not answer or not answer.strip():
                raise RuntimeError("pipeline returned an empty answer")
            return answer, contexts, sources, meta
        except Exception as exc:
            rate_limited = "ratelimit" in type(exc).__name__.lower()
            if is_daily_quota_error(exc) or not rate_limited or attempt == 3:
                raise
            print(f"  [rate limited, waiting {20 * attempt}s]")
            time.sleep(20 * attempt)


def generate_answers(items: list[dict], pipeline: str, save) -> bool:
    """Returns False if it had to stop early because of a daily quota."""
    pending = [it for it in items if it["answer"] is None]
    if not pending:
        return True
    print(f"\nGenerating answers with the {pipeline} pipeline: {len(pending)} to do")
    for n, item in enumerate(pending, start=1):
        print(f"[gen {n}/{len(pending)}] ({item['category']}) {item['question']}")
        try:
            answer, contexts, sources, meta = run_pipeline(pipeline, item["question"])
        except Exception as exc:
            if is_daily_quota_error(exc):
                print("  The app LLM's daily quota is used up. Progress is saved; re-run later to continue.")
                return False
            print(f"  [generation failed, will retry on the next run] {str(exc)[:200]}")
            continue
        item.update(answer=answer, contexts=contexts, sources=sources, meta=meta)
        save()
        time.sleep(GENERATION_DELAY_SECONDS)
    return True


# --------------------------------------------------------------------------
# Phase 2: score answers
# --------------------------------------------------------------------------

def score_answers(items: list[dict], save) -> bool:
    """Returns False if it had to stop early because of a daily quota."""
    todo = [it for it in items if it["answer"] is not None and missing_metrics(it)]
    if not todo:
        return True
    print(f"\nScoring: {len(todo)} answers have metrics left to score")

    from eval.metrics import build_metrics, score_metric
    metrics = build_metrics()

    for n, item in enumerate(todo, start=1):
        print(f"[score {n}/{len(todo)}] ({item['category']}) {item['question']}")
        # An empty context list means nothing usable was retrieved; RAGAS needs
        # at least one string, so pass an explicit statement of that fact.
        contexts = item["contexts"] or [NO_CONTEXT_PLACEHOLDER]
        for name in missing_metrics(item):
            try:
                value = score_metric(metrics, name, item["question"], item["answer"], contexts, item["reference"])
            except Exception as exc:
                if is_daily_quota_error(exc):
                    print("  The judge's daily quota is used up. Progress is saved; re-run later to continue.")
                    save()
                    return False
                item["failures"][name] = item["failures"].get(name, 0) + 1
                print(f"  {name}: FAILED ({item['failures'][name]}/{MAX_METRIC_FAILURES}) {str(exc)[:160]}")
                save()
                continue
            item["scores"][name] = value
            print(f"  {name}: {'undefined' if value is None else f'{value:.3f}'}")
            save()
            time.sleep(SCORING_DELAY_SECONDS)
    return True


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def summarize(items: list[dict]) -> dict:
    buckets: dict[str, dict[str, list[float]]] = {}
    for item in items:
        for name, value in item["scores"].items():
            if value is None:
                continue
            for key in (item["category"], "overall"):
                buckets.setdefault(key, {}).setdefault(name, []).append(value)
    return {
        key: {name: round(sum(vals) / len(vals), 4) for name, vals in by_metric.items()}
        for key, by_metric in buckets.items()
    }


def print_summary(summary: dict, items: list[dict]) -> None:
    counts = {c: sum(1 for it in items if it["category"] == c) for c in CATEGORIES}
    print(f"\n{'Category':<15} {'n':<4} {'Faithfulness':<14} {'AnswerRel':<12} {'CtxPrecision':<14} {'CtxRecall':<10}")
    for key in CATEGORIES + ["overall"]:
        if key not in summary:
            continue
        row = summary[key]
        n = len(items) if key == "overall" else counts[key]
        cells = [f"{row[m]:.3f}" if m in row else "-" for m in METRIC_NAMES]
        print(f"{key:<15} {n:<4} {cells[0]:<14} {cells[1]:<12} {cells[2]:<14} {cells[3]:<10}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="e.g. baseline or agentic; also names the results file")
    parser.add_argument("--pipeline", choices=["baseline", "agentic"], default="baseline")
    parser.add_argument("--sample", "--limit", type=int, default=12, dest="sample",
                        help="stratified sample size (0 = all questions)")
    parser.add_argument("--phase", choices=["all", "generate", "score"], default="all")
    parser.add_argument("--fresh", action="store_true", help="discard saved progress for this label")
    parser.add_argument("--retry-failed", action="store_true",
                        help="clear recorded failures so failed metrics are attempted again")
    parser.add_argument("--no-log", action="store_true", help="don't write to the regression log")
    parser.add_argument("--regression-log-path", type=str, default=None)
    args = parser.parse_args()

    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    ids = select_ids(golden_set, args.sample)

    path = RESULTS_DIR / f"{args.label}.json"
    state = load_state(path, args.label, args.pipeline, args.fresh)
    ensure_items(state, golden_set, ids)
    if args.retry_failed:
        for it in state["items"]:
            it["failures"] = {}
    save = lambda: save_state(path, state)
    save()

    items = [it for it in state["items"] if it["id"] in set(ids)]
    print(f"Label '{args.label}' | pipeline: {args.pipeline} | {len(items)} questions | saved to {path}")

    if args.phase in ("all", "generate"):
        generate_answers(items, args.pipeline, save)
    if args.phase in ("all", "score"):
        score_answers(items, save)

    summary = summarize(items)
    if summary:
        print_summary(summary, items)

    generated = sum(1 for it in items if it["answer"] is not None)
    coverage = metric_coverage(items)
    print(f"\nAnswers generated: {generated}/{len(items)}")
    print("Questions scored per metric: " + " | ".join(f"{m} {n}/{len(items)}" for m, n in coverage.items()))

    if args.phase == "generate":
        return 2 if any(it["answer"] is None for it in items) else 0

    if has_pending_work(items):
        print("\nRun incomplete. Re-run the same command to continue; nothing has been written "
              "to the regression log yet.")
        return 2

    thin = [m for m, n in coverage.items() if n / len(items) < MIN_SUCCESS_RATE]
    if thin:
        print(f"\nFAIL: {', '.join(thin)} could be scored for fewer than {MIN_SUCCESS_RATE:.0%} of the "
              f"questions. Averages over such a small, non-random subset would be misleading (long answers "
              f"are the ones most likely to fail), so this run was not logged. Fix the cause, then re-run "
              f"with --retry-failed.")
        return 1

    if args.no_log:
        print("\nComplete (--no-log: regression log untouched).")
    elif state["logged"]:
        print("\nComplete (already recorded in the regression log earlier).")
    else:
        from eval.store import append_run
        kwargs = {"path": Path(args.regression_log_path)} if args.regression_log_path else {}
        row = append_run(state["label"], len(items), summary, **kwargs)
        state["logged"] = True
        save()
        print(f"\nComplete. Appended to the regression log: {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())