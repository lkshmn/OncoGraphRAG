import json
import random
from pathlib import Path
from collections import defaultdict

# Config
INPUT_FILES = [
    Path("d:/Projects/OncoGraphRAG/data/abstracts/Breast_Cancer.jsonl"),
    # Add other files if needed
]
RELATIONS_DIR = Path("d:/Projects/OncoGraphRAG/data/extracted relations")
OUTPUT_DIR = Path("d:/Projects/OncoGraphRAG/data/evaluation/prompts")
SAMPLES_PER_FILE = 1


def load_relations_by_pub_id(cancer_type_stem: str) -> dict[str, list[dict]]:
    """
    Loads extracted relations from the matching JSONL file and indexes them by pub_id.
    cancer_type_stem example: 'Breast_Cancer' -> matches 'Breast_Cancer_relations_*.jsonl'
    """
    relations_by_id = defaultdict(list)
    matches = list(RELATIONS_DIR.glob(f"{cancer_type_stem}_relations_*.jsonl"))
    if not matches:
        print(f"Warning: No relations file found for {cancer_type_stem}")
        return relations_by_id

    for rel_file in matches:
        with open(rel_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rel = json.loads(line)
                    pub_id = rel.get('pub_id', '')
                    relations_by_id[pub_id].append(rel)
                except json.JSONDecodeError:
                    continue

    return relations_by_id


def format_relations(relations: list[dict]) -> str:
    """Formats a list of relation dicts into a readable text block for the prompt."""
    lines = []
    for i, rel in enumerate(relations, 1):
        lines.append(
            f"  {i}. [{rel.get('entity1_type', '?')}] \"{rel.get('entity1_name', '?')}\" "
            f"--{rel.get('relation_type', '?')}--> "
            f"[{rel.get('entity2_type', '?')}] \"{rel.get('entity2_name', '?')}\" "
            f"| Efficacy: {rel.get('efficacy', 'Not Stated')} "
            f"| Phase: {rel.get('trial_phase', 'Unknown')} "
            f"| Reliability: {rel.get('reliability_score', 50)}"
        )
    return "\n".join(lines)


def prepare_prompt():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in INPUT_FILES:
        if not file_path.exists():
            continue

        cancer_type = file_path.stem.lower()
        # Load relations indexed by PMID for this cancer type
        relations_by_id = load_relations_by_pub_id(file_path.stem)

        if not relations_by_id:
            print(f"Skipping {cancer_type}: no extracted relations found.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Filter to abstracts that have extracted relations
        abstracts_with_relations = []
        for line in lines:
            data = json.loads(line.strip())
            raw_id = data.get('id', '')
            if raw_id in relations_by_id:
                abstracts_with_relations.append(data)

        if not abstracts_with_relations:
            print(f"No abstracts with matching relations for {cancer_type}")
            continue

        selected = random.sample(
            abstracts_with_relations,
            min(len(abstracts_with_relations), SAMPLES_PER_FILE)
        )

        for data in selected:
            raw_id = data.get('id', 'unknown')
            safe_id = raw_id.replace('PMID:', '')
            rels = relations_by_id.get(raw_id, [])

            snippet = (
                f"ID: {raw_id}\n"
                f"Title: {data.get('title')}\n"
                f"Phase: {data.get('trial_phase_metadata')}\n"
                f"Abstract: {data.get('text')}"
            )

            formatted_rels = format_relations(rels)

            prompt_header = f"""
I am building an evaluation dataset for an Oncology RAG chatbot that answers questions using a Knowledge Graph.

**IMPORTANT CONTEXT — HOW THE SYSTEM WORKS:**
The chatbot does NOT read raw abstracts. It can ONLY answer questions by querying a Knowledge Graph that stores extracted relations in this structure:
  [entity1_type] "entity1_name" --relation_type--> [entity2_type] "entity2_name"
  + metadata: efficacy, trial_phase, reliability_score, source PMID

The chatbot retrieves these relations via SPARQL and uses them to generate answers. It CANNOT answer questions about details that are NOT captured in the extracted relations (e.g., specific dosages, sample sizes, confidence intervals, adverse event percentages, study methodology).

**EXTRACTED RELATIONS FROM THIS ABSTRACT (this is ALL the chatbot knows):**
{formatted_rels}

**TASK:**
Generate at least 5 simple questions that the chatbot CAN answer using ONLY the extracted relations above.

**CRITICAL REQUIREMENT 1: Answerability from Relations**
Every question MUST be answerable using the extracted relations provided above. Do NOT ask about:
- Specific dosages, sample sizes, or detailed statistics not in the relations
- Study methodology or design details
- Adverse event rates or specific percentages
- Any facts only found in the abstract text but NOT in the relations
The ground_truth_answer must be derivable from the relation data (entity names, relation types, efficacy, trial phase, reliability score).

**CRITICAL REQUIREMENT 2: Simplicity**
Keep questions short, direct, and simple. Ask about the treatment, disease, relation type, efficacy outcome, or trial phase as captured in the relations.

**CRITICAL REQUIREMENT 3: Self-Contained Questions**
Every question MUST be standalone. Do NOT use "in this study" or "the patients."
Explicitly mention the specific drug/disease names from the relations.

**Question Types:**
- **Simple 1-Hop**: Basic fact from a single relation (e.g., "What does trastuzumab treat?", "What phase was the trial for erlotinib in lung cancer?").
- **Simple MCQ**: Brief question with 4 options (A, B, C, D), one correct.
- **Simple True/False**: A statement that is clearly supported or contradicted by the relations.
- **Simple Open-Ended**: Short question about the main finding/efficacy for a treatment-disease pair.

**Sample Format:**
Return valid JSON list:
[
  {{
    "source_id": "PMID:12345",
    "type": "Simple 1-Hop",
    "question": "What does trastuzumab treat?",
    "options": null,
    "ground_truth_answer": "HER2-positive early breast cancer.",
    "ground_truth_context": "[treatment] trastuzumab --treat--> [disease] HER2-positive early breast cancer | Efficacy: Positive: improved disease-free survival | Phase: Phase III | Reliability: 95",
    "key_entities": ["trastuzumab", "HER2-positive early breast cancer"]
  }}
]

**New field — ground_truth_context**: For each question, include the specific relation(s) from the list above that support the answer. This will be used to calculate Context Recall.

**Constraint**: The ground_truth_answer MUST be derivable from the extracted relations, NOT from the raw abstract.

**DATA SNIPPET (for reference only — questions must be answerable from RELATIONS above):**
"""

            full_content = prompt_header + "\n\n" + snippet

            output_file = OUTPUT_DIR / f"{cancer_type}_q_{safe_id}.txt"
            with open(output_file, 'w', encoding='utf-8') as out_f:
                out_f.write(full_content)

            print(f"Prompt generated at: {output_file} ({len(rels)} relations included)")


if __name__ == "__main__":
    prepare_prompt()