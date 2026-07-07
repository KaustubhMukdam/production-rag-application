"""
Eval runner — Phase 4.

Changes from Phase 3:
  - _score() returns dict[str, float] with keys:
      faithfulness, answer_relevancy, context_precision
  - Results include the question so AnswerRelevancyScorer and
    ContextPrecisionScorer receive the right inputs.
  - --metric flag: run a single scorer for fast local iteration.
  - Aligned table printed at the end; CI exits 1 if ANY metric fails.
"""
import argparse
import json
import random
import sys

from app.pipeline import Pipeline
from app.ingestion.loader import load_documents
from app.generation.generator import create_generator
from eval.faithfulness import FaithfulnessScorer, AnswerRelevancyScorer, ContextPrecisionScorer
from eval.thresholds import FAITHFULNESS_MIN, ANSWER_RELEVANCY_MIN, CONTEXT_PRECISION_MIN

_THRESHOLDS = {
    "faithfulness": FAITHFULNESS_MIN,
    "answer_relevancy": ANSWER_RELEVANCY_MIN,
    "context_precision": CONTEXT_PRECISION_MIN,
}


def _load_questions(sample: int | None, seed: int):
    with open("eval/eval_questions.json") as f:
        questions = json.load(f)
    if sample:
        rng = random.Random(seed)
        questions = rng.sample(questions, min(sample, len(questions)))
    return questions


def _run_pipeline(questions: list, provider: str):
    print("  Loading and indexing documents...")
    docs = load_documents()
    pipeline = Pipeline()
    pipeline.generator = create_generator(provider)
    pipeline.index_documents(docs)

    results = []
    total = len(questions)
    for i, q in enumerate(questions):
        result = pipeline.query(q["question"])
        results.append({
            "question": result["question"],
            "answer": result["answer"],
            "contexts": [c["text"] for c in result["retrieved_chunks"]],
        })
        print(f"  [{i+1}/{total}] {q['id']}")

    return results


def _score(results: list, metric: str | None = None) -> dict[str, float]:
    """Score results on all three eval metrics (or a single metric).

    Args:
        results: List of {question, answer, contexts} dicts.
        metric:  If set, only that metric is scored; others return 0.0.

    Returns:
        dict with keys faithfulness, answer_relevancy, context_precision.
    """
    if not results:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}

    scores: dict[str, float] = {}

    if metric is None or metric == "faithfulness":
        print("  Scoring faithfulness...")
        scorer = FaithfulnessScorer()
        vals = [scorer.score(r["answer"], r["contexts"]) for r in results]
        scores["faithfulness"] = round(sum(vals) / len(vals), 3)
    else:
        scores["faithfulness"] = 0.0

    if metric is None or metric == "answer_relevancy":
        print("  Scoring answer relevancy...")
        scorer = AnswerRelevancyScorer()
        vals = [scorer.score(r["question"], r["answer"]) for r in results]
        scores["answer_relevancy"] = round(sum(vals) / len(vals), 3)
    else:
        scores["answer_relevancy"] = 0.0

    if metric is None or metric == "context_precision":
        print("  Scoring context precision...")
        scorer = ContextPrecisionScorer()
        vals = [scorer.score(r["question"], r["contexts"]) for r in results]
        scores["context_precision"] = round(sum(vals) / len(vals), 3)
    else:
        scores["context_precision"] = 0.0

    return scores


def _print_table(provider: str, scores: dict[str, float], metric: str | None) -> bool:
    """Print an aligned results table. Returns True if all checked metrics pass."""
    col_w = 22
    print(f"\n  Provider: {provider}")
    print("  " + "─" * 52)
    print(f"  {'Metric':<{col_w}} {'Score':>7}   {'Threshold':>9}   {'Result'}")
    print("  " + "─" * 52)

    all_pass = True
    for metric_name, score in scores.items():
        if metric is not None and metric_name != metric:
            continue
        threshold = _THRESHOLDS[metric_name]
        passed = score >= threshold
        if not passed:
            all_pass = False
        result_str = "PASS" if passed else "FAIL"
        print(
            f"  {metric_name:<{col_w}} {score:>7.3f}   ≥ {threshold:.3f}       {result_str}"
        )

    print("  " + "─" * 52)
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="CI-gated RAG evaluation")
    parser.add_argument("--provider", choices=["groq", "ollama", "both"], default="both")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric",
        choices=["faithfulness", "answer_relevancy", "context_precision"],
        default=None,
        help="Run only this metric (default: all three)",
    )
    args = parser.parse_args()

    questions = _load_questions(args.sample, args.seed)
    print(f"Loaded {len(questions)} eval question(s)")

    providers = ["groq", "ollama"] if args.provider == "both" else [args.provider]

    print(f"\n{'=' * 56}")
    overall_pass = True
    for provider in providers:
        print(f"\n--- {provider} pipeline ---")
        results = _run_pipeline(questions, provider)
        scores = _score(results, metric=args.metric)
        passed = _print_table(provider, scores, args.metric)
        if not passed:
            overall_pass = False

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
