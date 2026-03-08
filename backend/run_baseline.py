"""
run_baseline.py
Queries the local LLM directly (no RAG, no knowledge graph) with the same
120 ground truth questions. Serves as a baseline to compare against RAG results.

Requires: LM Studio running on localhost:1234
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Configuration ---
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")

GROUND_TRUTH_DIR = Path("d:/Projects/OncoGraphRAG/data/evaluation/ground_truth")
OUTPUT_DIR = Path("d:/Projects/OncoGraphRAG/data/evaluation/results")
MAX_QUESTIONS = None  # Set to None to run all questions


def load_ground_truth_files() -> list[dict]:
    """Load all ground truth JSON files and flatten into a list of question records."""
    all_questions = []
    gt_files = sorted(GROUND_TRUTH_DIR.glob("*_a_*.txt"))

    for gt_file in gt_files:
        try:
            content = gt_file.read_text(encoding="utf-8").strip()
            if not content:
                print(f"  SKIP (empty): {gt_file.name}")
                continue
            questions = json.loads(content)
            for q in questions:
                q["_gt_file"] = gt_file.name
            all_questions.extend(questions)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  SKIP (parse error): {gt_file.name} — {e}")

    return all_questions


def query_llm_direct(llm: ChatOpenAI, question: str, options: list[str] | None) -> dict:
    """Send a question directly to the LLM without any RAG context."""
    prompt = (
        "You are an oncology expert. Answer the following question based on your general medical knowledge.\n"
        "Be concise and direct.\n\n"
        f"Question: {question}\n"
    )
    if options:
        prompt += "Options:\n" + "\n".join(f"  {opt}" for opt in options) + "\n"
    prompt += "\nAnswer:"

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        return {"answer": answer, "error": None}
    except Exception as e:
        return {"answer": "", "error": str(e)}


def run_baseline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"baseline_results_{timestamp}.jsonl"

    print("Loading ground truth files...")
    questions = load_ground_truth_files()
    print(f"Loaded {len(questions)} questions from {len(set(q['_gt_file'] for q in questions))} files\n")

    if not questions:
        print("No questions found. Make sure ground truth files are populated.")
        return

    if MAX_QUESTIONS:
        questions = questions[:MAX_QUESTIONS]
        print(f"Testing with first {MAX_QUESTIONS} question(s)\n")

    print("Initializing LLM (no RAG)...")
    llm = ChatOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
        model=LMSTUDIO_MODEL,
        temperature=0,
    )
    print("LLM ready.\n")

    results = []
    errors = 0

    for i, q in enumerate(questions, 1):
        question = q["question"]
        print(f"[{i}/{len(questions)}] {question[:80]}...")

        result = query_llm_direct(llm, question, q.get("options"))

        record = {
            "question_id": i,
            "source_id": q.get("source_id", ""),
            "question_type": q.get("type", ""),
            "question": question,
            "options": q.get("options"),
            # LLM output (no context)
            "answer": result["answer"],
            "retrieved_context": "",
            "generated_sparql": "",
            # Ground truth
            "ground_truth_answer": q.get("ground_truth_answer", ""),
            "ground_truth_context": q.get("ground_truth_context", ""),
            "key_entities": q.get("key_entities", []),
            # Metadata
            "gt_file": q.get("_gt_file", ""),
            "error": result["error"],
        }

        results.append(record)

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        status = "ERROR" if result["error"] else "OK"
        if result["error"]:
            errors += 1
        print(f"  → {status}\n")

        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"Baseline evaluation complete!")
    print(f"  Total: {len(questions)} | Success: {len(questions) - errors} | Errors: {errors}")
    print(f"  Results: {output_file}")


if __name__ == "__main__":
    run_baseline()