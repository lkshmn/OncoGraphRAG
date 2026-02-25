import json
from pathlib import Path
from bulk_relations import extract_relations

INPUT = Path(r"d:\Projects\OncoGraphRAG\data\pubmed_cancer_findings.json")
OUTPUT = Path(r"d:\Projects\OncoGraphRAG\data\pubmed_oncology_relations_10.json")

def pick_text(rec):
    return rec.get("abstract") or rec.get("text") or rec.get("AbstractText") or ""

def pick_id(rec, idx):
    return rec.get("source_id") or rec.get("pmid") or rec.get("id") or f"pubmed_{idx}"

def load_first_10(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    abstracts = []
    for i, rec in enumerate(data):
        txt = pick_text(rec)
        if not txt.strip():
            continue
        abstracts.append({"id": pick_id(rec, i), "text": txt})
        if len(abstracts) >= 10:
            break
    return abstracts

def main():
    abstracts = load_first_10(INPUT)
    if not abstracts:
        print("No abstracts found.")
        return
    print(f"Running LM Studio extraction on {len(abstracts)} abstracts...")
    relations = extract_relations(abstracts, model="lmstudio")
    print(f"Extracted {len(relations)} relations.")
    OUTPUT.write_text(json.dumps(relations, indent=2), encoding="utf-8")
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    main()