"""
eval/run_eval.py

Runs the golden evaluation set (M3) through the RAG pipeline and scores
each answer with the four RAGAS metrics (M4), producing a labeled results
file and a summary table broken down by question category. Also appends
a summary row to a regression log (M6/M7) for trend tracking.

Usage:
    python -m eval.run_eval --label baseline --limit 5   # quick smoke test
    python -m eval.run_eval --label baseline             # full run
    python -m eval.run_eval --label ci --limit 10 --regression-log-path eval/ci_regression_log.json
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from api.rag_pipeline import generate_answer, retrieve
from eval.metrics import build_metrics, score_sample
from eval.store import append_run

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_REQUEST_DELAY_SECONDS = 1
MAX_RETRIES = 5


def score_with_retry(metrics, question, answer, contexts, reference):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return score_sample(metrics, question, answer, contexts, reference)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"    [scoring failed after {MAX_RETRIES} attempts] {e}")
                return None
            wait = 3 * attempt
            print(f"    [scoring error, retrying in {wait}s] {e}")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="e.g. 'baseline', 'agentic', or 'ci'")
    parser.add_argument("--limit", type=int, default=None, help="only process a stratified sample of N questions")
    parser.add_argument("--regression-log-path", type=str, default=None, help="override the default eval/regression_log.json path")
    args = parser.parse_args()

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    if args.limit:
        step = max(1, len(golden_set) // args.limit)
        golden_set = golden_set[::step][: args.limit]

    metrics = build_metrics()
    per_item_results = []
    category_scores = defaultdict(lambda: defaultdict(list))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"{args.label}_{timestamp}.json"

    summary = {}

    for i, item in enumerate(golden_set, start=1):
        question = item["question"]
        category = item["category"]
        reference = item["expected_answer"]

        print(f"[{i}/{len(golden_set)}] ({category}) {question}")

        chunks = retrieve(question)
        contexts = [c["text"] for c in chunks]
        answer = generate_answer(question, chunks)

        scores = score_with_retry(metrics, question, answer, contexts, reference)

        per_item_results.append({
            "question": question,
            "category": category,
            "answer": answer,
            "retrieved_sources": sorted({c["url"] for c in chunks}),
            "expected_sources": item["expected_sources"],
            "scores": scores,
        })

        if scores:
            for metric_name, value in scores.items():
                category_scores[category][metric_name].append(value)
                category_scores["overall"][metric_name].append(value)

        summary = {
            cat: {
                metric_name: round(sum(values) / len(values), 4)
                for metric_name, values in metrics_dict.items()
                if values
            }
            for cat, metrics_dict in category_scores.items()
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "label": args.label,
                "timestamp": timestamp,
                "n_questions": len(golden_set),
                "completed": i,
                "summary": summary,
                "per_item": per_item_results,
            }, f, ensure_ascii=False, indent=2)

        time.sleep(JUDGE_REQUEST_DELAY_SECONDS)

    print(f"\nSaved results to {output_path}\n")
    print(f"{'Category':<15} {'Faithfulness':<14} {'AnswerRel':<12} {'CtxPrecision':<14} {'CtxRecall':<10}")
    for category, scores_dict in summary.items():
        print(
            f"{category:<15} "
            f"{scores_dict.get('faithfulness', float('nan')):<14.3f} "
            f"{scores_dict.get('answer_relevancy', float('nan')):<12.3f} "
            f"{scores_dict.get('context_precision', float('nan')):<14.3f} "
            f"{scores_dict.get('context_recall', float('nan')):<10.3f}"
        )

    if summary:
        log_path = Path(args.regression_log_path) if args.regression_log_path else None
        row = append_run(args.label, len(golden_set), summary, path=log_path) if log_path else append_run(args.label, len(golden_set), summary)
        print(f"\nAppended to regression log: {row}")


if __name__ == "__main__":
    main()