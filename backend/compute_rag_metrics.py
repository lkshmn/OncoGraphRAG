"""
compute_metrics.py
Computes evaluation metrics:
- Semantic Similarity: computed locally via embeddings
- Faithfulness, Answer Relevancy, Context Recall: prompts generated for Gemini web interface

Workflow:
1. Run this script -> computes Semantic Similarity + generates prompt files
2. Paste each prompt file into Gemini web -> copy JSON output into corresponding response file
3. Run this script again -> merges all scores into final results
"""

import json
from pathlib import Path

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

# --- Configuration ---
RESULTS_FILE = Path("d:/Projects/OncoGraphRAG/data/evaluation/results/baseline_results_20260308_123041.jsonl")
MAX_EVAL = None  # Set to None to evaluate all
OUTPUT_FILE = RESULTS_FILE.parent / "metrics_results.json"
PROMPTS_DIR = RESULTS_FILE.parent / "gemini_prompts"
RESPONSES_DIR = RESULTS_FILE.parent / "gemini_responses"
BATCH_SIZE = 10  # Questions per Gemini prompt

EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_results(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_samples(records: list[dict]) -> list[SingleTurnSample]:
    samples = []
    for r in records:
        answer = r.get("answer", "") or ""
        retrieved = r.get("retrieved_context", "") or ""
        gt_context = r.get("ground_truth_context", "") or ""
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
    return samples


def compute_semantic_similarity(valid: list[dict]) -> dict[int, float]:
    """Compute semantic similarity locally using embeddings."""
    print("Computing Semantic Similarity locally...")
    samples = build_samples(valid)
    dataset = EvaluationDataset(samples=samples)
    wrapped_embeddings = LangchainEmbeddingsWrapper(EMBEDDINGS)
    run_config = RunConfig(max_workers=1, timeout=300)
    result = evaluate(
        dataset=dataset,
        metrics=[SemanticSimilarity(embeddings=wrapped_embeddings)],
        run_config=run_config,
    )
    scores = {}
    for i, rec in enumerate(valid):
        scores[rec["question_id"]] = result.scores[i].get("semantic_similarity")
    print(f"  Done. Computed {len(scores)} scores.\n")
    return scores


def generate_gemini_prompts(valid: list[dict]):
    """Generate batch prompt files for Gemini web interface."""
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    total_batches = (len(valid) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        batch = valid[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
        prompt_file = PROMPTS_DIR / f"batch_{batch_num + 1:02d}.txt"
        response_file = RESPONSES_DIR / f"batch_{batch_num + 1:02d}.json"

        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(
                "You are an evaluation judge for a RAG chatbot. "
                "Score each question below on three metrics using a fine-grained scale from 0.0 to 1.0. "
                "Use intermediate values (e.g., 0.2, 0.4, 0.6, 0.8) - do NOT just use 0 or 1.\n\n"
                "1. **faithfulness**: What fraction of claims in the Answer are supported by the Retrieved Context?\n"
                "   - 1.0 = all claims supported\n"
                "   - 0.8 = most claims supported, minor unsupported detail\n"
                "   - 0.6 = core answer supported but some claims lack evidence\n"
                "   - 0.4 = partially supported, significant unsupported claims\n"
                "   - 0.2 = mostly unsupported, only minor overlap\n"
                "   - 0.0 = completely fabricated or no context available\n\n"
                "2. **answer_relevancy**: How well does the Answer address the Question?\n"
                "   - 1.0 = directly and completely answers the question\n"
                "   - 0.8 = answers the question but includes extra irrelevant info\n"
                "   - 0.6 = partially answers, missing some key details\n"
                "   - 0.4 = tangentially related but doesn't clearly answer\n"
                "   - 0.2 = mostly irrelevant, only vaguely related\n"
                "   - 0.0 = does not address the question at all or says \"I don't know\"\n\n"
                "3. **context_recall**: What fraction of the Ground Truth Context is covered by the Retrieved Context?\n"
                "   - 1.0 = all ground truth entities, relations, and metadata are present\n"
                "   - 0.8 = most ground truth info present, minor detail missing\n"
                "   - 0.6 = key relation present but some metadata (phase/efficacy/score) missing\n"
                "   - 0.4 = only some ground truth info present\n"
                "   - 0.2 = very little overlap with ground truth\n"
                "   - 0.0 = no overlap or context is empty/\"I don't know\"\n\n"
                "IMPORTANT: Use the full range of scores. Most answers will fall between 0.2 and 0.8.\n"
                "Return ONLY a valid JSON array. No explanation, no markdown fences.\n\n"
                "Format:\n"
                "[\n"
                '  {"question_id": 1, "faithfulness": 0.8, "answer_relevancy": 0.9, "context_recall": 0.7},\n'
                "  ...\n"
                "]\n\n"
                "---\n\n"
            )

            for r in batch:
                retrieved = r.get("retrieved_context") or r.get("answer", "")
                f.write(f"### Question ID: {r['question_id']}\n")
                f.write(f"**Question:** {r['question']}\n")
                f.write(f"**Answer:** {r['answer']}\n")
                f.write(f"**Retrieved Context:** {retrieved}\n")
                f.write(f"**Ground Truth Answer:** {r['ground_truth_answer']}\n")
                f.write(f"**Ground Truth Context:** {r['ground_truth_context']}\n")
                f.write("\n---\n\n")

        # Create empty response file if it doesn't exist
        if not response_file.exists():
            response_file.write_text("", encoding="utf-8")

    print(f"Generated {total_batches} prompt files in {PROMPTS_DIR}")
    print(f"Empty response files created in {RESPONSES_DIR}\n")
    print("Next steps:")
    print("  1. Open each batch_XX.txt and paste into Gemini web")
    print("  2. Copy the JSON output into the matching batch_XX.json")
    print("  3. Run this script again to merge results\n")


def merge_results(valid: list[dict], sim_scores: dict[int, float]):
    """Merge Gemini responses + local semantic similarity into final output."""
    response_files = sorted(RESPONSES_DIR.glob("batch_*.json"))
    gemini_scores = {}

    for rf in response_files:
        content = rf.read_text(encoding="utf-8").strip()
        if not content:
            continue
        try:
            batch_scores = json.loads(content)
            for entry in batch_scores:
                gemini_scores[entry["question_id"]] = entry
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not parse {rf.name} - {e}")

    if not gemini_scores:
        print("No Gemini responses found yet. Paste outputs into gemini_responses/batch_XX.json files.")
        return

    per_question = []
    for r in valid:
        qid = r["question_id"]
        entry = {
            "question_id": qid,
            "question": r["question"],
            "question_type": r["question_type"],
            "source_id": r["source_id"],
            "semantic_similarity": sim_scores.get(qid),
        }
        if qid in gemini_scores:
            entry["faithfulness"] = gemini_scores[qid].get("faithfulness")
            entry["answer_relevancy"] = gemini_scores[qid].get("answer_relevancy")
            entry["context_recall"] = gemini_scores[qid].get("context_recall")
        per_question.append(entry)

    # Compute overall averages
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "semantic_similarity"]
    overall = {}
    for m in metric_names:
        vals = [q[m] for q in per_question if q.get(m) is not None]
        overall[m] = round(sum(vals) / len(vals), 4) if vals else 0.0

    output = {
        "overall": overall,
        "total_evaluated": len(per_question),
        "gemini_responses_found": len(gemini_scores),
        "per_question": per_question,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print("=" * 50)
    print("OVERALL SCORES")
    print("=" * 50)
    for m, s in overall.items():
        print(f"  {m}: {s:.4f}")
    print(f"\nResults saved to {OUTPUT_FILE}")


def main():
    print(f"Loading results from {RESULTS_FILE.name}...")
    records = load_results(RESULTS_FILE)
    valid = [r for r in records if not r.get("error")]
    if MAX_EVAL:
        valid = valid[:MAX_EVAL]
    print(f"Evaluating {len(valid)} records\n")

    # Step 1: Compute semantic similarity locally
    sim_scores = compute_semantic_similarity(valid)

    # Step 2: Generate prompts for Gemini (or merge if responses exist)
    has_responses = RESPONSES_DIR.exists() and any(
        f.read_text(encoding="utf-8").strip() for f in RESPONSES_DIR.glob("batch_*.json")
    )

    if has_responses:
        print("Gemini responses found - merging results...\n")
        merge_results(valid, sim_scores)
    else:
        generate_gemini_prompts(valid)


if __name__ == "__main__":
    main()