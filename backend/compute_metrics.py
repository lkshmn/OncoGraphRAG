"""
compute_metrics.py
Computes RAGAS evaluation metrics from run_evaluation.py results.

Metrics: Faithfulness, Answer Relevancy, Semantic Similarity, Context Recall

Install: pip install ragas sentence-transformers
"""

import json
from pathlib import Path

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    SemanticSimilarity,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
RESULTS_FILE = Path("d:/Projects/OncoGraphRAG/data/evaluation/results/eval_results_20260307_192431.jsonl")
MAX_EVAL = 1  # Set to None to evaluate all
OUTPUT_FILE = RESULTS_FILE.parent / "metrics_results.json"

# LM Studio as judge LLM
JUDGE_LLM = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="local-model",
    temperature=0,
)

# Local embedding model (no API needed)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_results(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(records: list[dict]) -> EvaluationDataset:
    samples = []
    for r in records:
        answer = r.get("answer", "") or ""
        retrieved = r.get("retrieved_context", "") or ""
        gt_context = r.get("ground_truth_context", "") or ""

        # If retrieved_context is empty, fall back to using the answer itself
        # (the answer was generated from SPARQL results, so it contains the context)
        if not retrieved.strip():
            retrieved = answer

        sample = SingleTurnSample(
            user_input=r["question"],
            response=answer,
            reference=r.get("ground_truth_answer", ""),
            retrieved_contexts=[retrieved] if retrieved.strip() else ["No context retrieved."],
            reference_contexts=[gt_context] if gt_context.strip() else [],
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def main():
    print(f"Loading results from {RESULTS_FILE.name}...")
    records = load_results(RESULTS_FILE)
    print(f"Loaded {len(records)} records")

    # Filter out errors
    valid = [r for r in records if not r.get("error")]
    skipped = len(records) - len(valid)
    if skipped:
        print(f"Skipping {skipped} records with errors")
    print(f"Evaluating {len(valid)} records\n")

    if MAX_EVAL:
        valid = valid[:MAX_EVAL]
        print(f"Testing with first {MAX_EVAL} record(s)")

    dataset = build_dataset(valid)

    wrapped_llm = LangchainLLMWrapper(JUDGE_LLM)
    wrapped_embeddings = LangchainEmbeddingsWrapper(EMBEDDINGS)

    metrics = [
        Faithfulness(llm=wrapped_llm),
        ResponseRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings),
        SemanticSimilarity(embeddings=wrapped_embeddings),
        ContextRecall(llm=wrapped_llm),
    ]

    print("Running RAGAS evaluation (this will take a while)...\n")
    result = evaluate(dataset=dataset, metrics=metrics)

    # Print overall scores
    print("\n" + "=" * 50)
    print("OVERALL SCORES")
    print("=" * 50)
    for metric_name, score in result._repr_dict.items():
        print(f"  {metric_name}: {score:.4f}")

    # Save per-question + overall results
    df = result.to_pandas()
    per_question = []
    for i, row in df.iterrows():
        entry = {
            "question_id": valid[i]["question_id"],
            "question": valid[i]["question"],
            "question_type": valid[i]["question_type"],
            "source_id": valid[i]["source_id"],
        }
        # Add metric scores from result.scores (list of dicts)
        for k, v in result.scores[i].items():
            entry[k] = v if not isinstance(v, float) or not (v != v) else None
        per_question.append(entry)

    output = {
        "overall": {k: round(v, 4) for k, v in result._repr_dict.items()},
        "total_evaluated": len(valid),
        "total_skipped": skipped,
        "per_question": per_question,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()