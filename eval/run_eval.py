import argparse
import json
import random
import time

from app.pipeline import Pipeline
from app.ingestion.loader import load_documents
from app.generation.generator import create_generator
from app.generation.groq import GroqError
from eval.faithfulness import FaithfulnessScorer
from eval.thresholds import FAITHFULNESS_MIN


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
    if provider == "groq":
        from app.generation.groq import GroqGenerator
        pipeline.generator = GroqGenerator(max_retries=5, fallback=None)
    else:
        pipeline.generator = create_generator(provider)
    pipeline.index_documents(docs)

    results = []
    total = len(questions)
    for i, q in enumerate(questions):
        try:
            result = pipeline.query(q["question"])
            results.append({
                "question": result["question"],
                "answer": result["answer"],
                "contexts": [c["text"] for c in result["retrieved_chunks"]],
            })
            print(f"  [{i+1}/{total}] {q['id']}")
        except GroqError as e:
            print(f"  [{i+1}/{total}] {q['id']} SKIPPED ({e})")
        if provider == "groq":
            time.sleep(3)

    return results


def _score(results: list):
    scorer = FaithfulnessScorer()
    print("  Scoring faithfulness...")
    scores = []
    for r in results:
        scores.append(scorer.score(r["answer"], r["contexts"]))
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["groq", "ollama", "both"], default="both")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = _load_questions(args.sample, args.seed)
    print(f"Loaded {len(questions)} eval questions")

    providers = ["groq", "ollama"] if args.provider == "both" else [args.provider]

    all_scores = {}
    for provider in providers:
        print(f"\n--- {provider} ---")
        results = _run_pipeline(questions, provider)
        score = _score(results)
        all_scores[provider] = score
        print(f"  Faithfulness: {score:.3f}")

    print(f"\n{'='*40}")
    print(f"{'Metric':<20} {'Groq':<10} {'Ollama':<10}")
    print(f"{'='*40}")
    groq_val = all_scores.get("groq", "—")
    ollama_val = all_scores.get("ollama", "—")
    print(f"{'faithfulness':<20} {str(groq_val):<10} {str(ollama_val):<10}")

    groq_faith = all_scores.get("groq", 0)
    if groq_faith >= FAITHFULNESS_MIN:
        print(f"\nPASS: Groq faithfulness {groq_faith:.3f} >= {FAITHFULNESS_MIN}")
        exit(0)
    else:
        print(f"\nFAIL: Groq faithfulness {groq_faith:.3f} < {FAITHFULNESS_MIN}")
        exit(1)


if __name__ == "__main__":
    main()
