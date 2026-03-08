"""
compute_accuracy.py
Computes Accuracy, Precision, Recall, and F1 for RAG and baseline results.

Uses Gemini web interface to classify each answer as correct/incorrect/no_answer,
then maps to TP/FP/FN to compute metrics.

Workflow:
1. Run this script -> generates prompt files for Gemini
2. Paste each prompt into Gemini web -> copy JSON output into response file
3. Run this script again -> computes metrics from classifications
"""

import json
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
RAG_RESULTS = Path("d:/Projects/OncoGraphRAG/data/evaluation/results/eval_results_20260307_192431.jsonl")
BASELINE_RESULTS = Path("d:/Projects/OncoGraphRAG/data/evaluation/results/baseline_results_20260308_123041.jsonl")
OUTPUT_FILE = Path("d:/Projects/OncoGraphRAG/data/evaluation/results/accuracy_report.json")
BATCH_SIZE = 20  # Questions per Gemini prompt


def load_results(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_dirs(results_path: Path) -> tuple[Path, Path]:
    """Get prompt/response dirs named after the results file."""
    stem = results_path.stem  # e.g. "eval_results_20260307_192431"
    parent = results_path.parent
    return parent / f"{stem}_acc_prompts", parent / f"{stem}_acc_responses"


def generate_gemini_prompts(valid: list[dict], results_path: Path):
    """Generate batch prompt files for Gemini to classify answers."""
    prompts_dir, responses_dir = get_dirs(results_path)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    total_batches = (len(valid) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        batch = valid[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
        prompt_file = prompts_dir / f"batch_{batch_num + 1:02d}.txt"
        response_file = responses_dir / f"batch_{batch_num + 1:02d}.json"

        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(
                "You are an evaluation judge. For each question below, classify the Answer "
                "as one of exactly three categories:\n\n"
                '- "correct" — The answer is factually correct and matches the ground truth in substance. '
                "Minor wording differences are fine. For MCQ, the correct option must be selected. "
                "For True/False, the correct value must be stated.\n"
                '- "incorrect" — The answer is provided but is factually wrong, picks the wrong option, '
                "or gives substantially different information from the ground truth.\n"
                '- "no_answer" — The system refused to answer, said "I don\'t know", '
                '"I\'m sorry", or failed to provide a meaningful response.\n\n'
                "Return ONLY a valid JSON array. No explanation, no markdown fences.\n\n"
                "Format:\n"
                "[\n"
                '  {"question_id": 1, "classification": "correct"},\n'
                '  {"question_id": 2, "classification": "incorrect"},\n'
                '  {"question_id": 3, "classification": "no_answer"},\n'
                "  ...\n"
                "]\n\n"
                "---\n\n"
            )

            for r in batch:
                f.write(f"### Question ID: {r['question_id']}\n")
                f.write(f"**Type:** {r.get('question_type', '')}\n")
                f.write(f"**Question:** {r['question']}\n")
                if r.get("options"):
                    f.write(f"**Options:** {', '.join(r['options'])}\n")
                f.write(f"**Answer:** {r.get('answer', '')}\n")
                f.write(f"**Ground Truth Answer:** {r.get('ground_truth_answer', '')}\n")
                f.write("\n---\n\n")

        if not response_file.exists():
            response_file.write_text("", encoding="utf-8")

    print(f"  Generated {total_batches} prompt files in {prompts_dir.name}/")
    print(f"  Empty response files in {responses_dir.name}/")
    print(f"\n  Next steps:")
    print(f"    1. Open each batch_XX.txt and paste into Gemini web")
    print(f"    2. Copy the JSON output into the matching batch_XX.json")
    print(f"    3. Run this script again to compute metrics\n")


def load_classifications(results_path: Path) -> dict[int, str]:
    """Load Gemini classification responses."""
    _, responses_dir = get_dirs(results_path)
    classifications = {}

    if not responses_dir.exists():
        return classifications

    for rf in sorted(responses_dir.glob("batch_*.json")):
        content = rf.read_text(encoding="utf-8").strip()
        if not content:
            continue
        try:
            batch = json.loads(content)
            for entry in batch:
                classifications[entry["question_id"]] = entry["classification"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not parse {rf.name} — {e}")

    return classifications


def compute_metrics(valid: list[dict], classifications: dict[int, str]) -> dict:
    """Compute accuracy, precision, recall, F1 from classifications."""
    # Map classifications to TP/FP/FN
    tp = fp = fn = 0
    per_question = []
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for r in valid:
        qid = r["question_id"]
        qtype = r.get("question_type", "")
        cls = classifications.get(qid, "no_answer")

        if cls == "correct":
            tp += 1
            per_type[qtype]["tp"] += 1
            confusion = "TP"
        elif cls == "incorrect":
            fp += 1
            per_type[qtype]["fp"] += 1
            confusion = "FP"
        else:  # no_answer
            fn += 1
            per_type[qtype]["fn"] += 1
            confusion = "FN"

        per_question.append({
            "question_id": qid,
            "question": r["question"],
            "question_type": qtype,
            "source_id": r.get("source_id", ""),
            "classification": cls,
            "confusion": confusion,
        })

    total = tp + fp + fn
    accuracy = tp / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-type metrics
    type_metrics = {}
    for qtype, counts in sorted(per_type.items()):
        t_tp, t_fp, t_fn = counts["tp"], counts["fp"], counts["fn"]
        t_total = t_tp + t_fp + t_fn
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
        t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0.0
        type_metrics[qtype] = {
            "accuracy": round(t_tp / t_total, 4) if t_total > 0 else 0.0,
            "precision": round(t_prec, 4),
            "recall": round(t_rec, 4),
            "f1": round(t_f1, 4),
            "tp": t_tp, "fp": t_fp, "fn": t_fn,
            "total": t_total,
        }

    return {
        "overall": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "total": total,
        },
        "per_type": type_metrics,
        "per_question": per_question,
    }


def print_metrics(label: str, metrics: dict):
    """Print metrics for one result set."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    o = metrics["overall"]
    print(f"\n  Overall ({o['total']} questions):")
    print(f"    TP={o['tp']}  FP={o['fp']}  FN={o['fn']}")
    print(f"    Accuracy:  {o['accuracy']:.4f}")
    print(f"    Precision: {o['precision']:.4f}")
    print(f"    Recall:    {o['recall']:.4f}")
    print(f"    F1:        {o['f1']:.4f}")

    print(f"\n  Per Question Type:")
    for qtype, m in metrics["per_type"].items():
        print(f"    {qtype} ({m['total']}): "
              f"Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")


def print_comparison(rag: dict, baseline: dict):
    """Print side-by-side comparison."""
    print(f"\n{'=' * 60}")
    print("  COMPARISON: RAG vs BASELINE")
    print(f"{'=' * 60}")

    print(f"\n  {'Metric':<25} {'RAG':>10} {'Baseline':>10} {'Delta':>10}")
    print(f"  {'-' * 55}")

    for m in ["accuracy", "precision", "recall", "f1"]:
        r_val = rag["overall"][m]
        b_val = baseline["overall"][m]
        print(f"  {m.capitalize():<25} {r_val:>10.4f} {b_val:>10.4f} {r_val - b_val:>+10.4f}")

    # Per type comparison
    all_types = sorted(set(list(rag["per_type"].keys()) + list(baseline["per_type"].keys())))
    for qtype in all_types:
        print(f"\n  {qtype}:")
        r_m = rag["per_type"].get(qtype, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0})
        b_m = baseline["per_type"].get(qtype, {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0})
        for m in ["accuracy", "precision", "recall", "f1"]:
            r_val = r_m[m]
            b_val = b_m[m]
            print(f"    {m.capitalize():<23} {r_val:>10.4f} {b_val:>10.4f} {r_val - b_val:>+10.4f}")


def process_file(results_path: Path) -> dict | None:
    """Process a single results file: generate prompts or compute metrics."""
    if not results_path.exists():
        print(f"\n  File not found: {results_path.name}")
        return None

    print(f"\nProcessing: {results_path.name}")

    records = load_results(results_path)
    valid = [r for r in records if not r.get("error")]
    print(f"  Total records: {len(records)} | Valid: {len(valid)}")

    classifications = load_classifications(results_path)

    if classifications:
        print(f"  Found {len(classifications)} classifications from Gemini")
        metrics = compute_metrics(valid, classifications)
        print_metrics(results_path.stem, metrics)
        return {"file": results_path.name, **metrics}
    else:
        print(f"  No classifications found — generating Gemini prompts...")
        generate_gemini_prompts(valid, results_path)
        return None


def main():
    rag_result = process_file(RAG_RESULTS)
    baseline_result = process_file(BASELINE_RESULTS)

    if rag_result and baseline_result:
        print_comparison(rag_result, baseline_result)

    output = {}
    if rag_result:
        output["rag"] = rag_result
    if baseline_result:
        output["baseline"] = baseline_result

    if output:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nFull report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()