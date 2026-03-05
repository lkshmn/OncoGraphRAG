import json
import random
from pathlib import Path

# Config
INPUT_FILES = [
    Path("d:/Projects/OncoGraphRAG/data/abstracts/Breast_Cancer.jsonl"),
    # Add other files if needed
]
OUTPUT_FILE = Path("d:/Projects/OncoGraphRAG/data/evaluation/qa_generation_prompt.txt")
ecample_count = 20  # How many abstracts to sample

def prepare_prompt():
    samples = []
    
    # 1. Collect random samples from your data
    for file_path in INPUT_FILES:
        if not file_path.exists(): continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Shuffle and pick random lines to get diverse trials
            selected_lines = random.sample(lines, min(len(lines), ecample_count))
            
            for line in selected_lines:
                data = json.loads(line)
                # Format: ID | Title | Phase | Text Snippet (first 500 chars)
                snippet = f"ID: {data.get('id')}\nTitle: {data.get('title')}\nPhase: {data.get('trial_phase_metadata')}\nAbstract: {data.get('text')[:600]}..."
                samples.append(snippet)

    # 2. Create the Mega-Prompt for Gemini/GPT
    prompt_header = """
    I am building an evaluation dataset for a RAG system getting information from Clinical Trial Abstracts.
    
    TASK:
    Based on the provided abstract snippets, generate 3 SPECIFIC questions for EACH abstract.
    
    REQUIREMENTS:
    1. **Question Types**:
       - Fact Retrieval: "What was the efficacy of [Drug] in [Disease]?"
       - Comparison/Outcome: "Did [Drug] improves survival in [Disease] patients?"
    
    2. **Format**:
       Return the output as a JSON list of objects:
       [
         {
           "source_id": "PMID:12345",
           "question": "Does Drug X improve overall survival in Phase 3?",
           "ground_truth_answer": "Yes, Drug X showed a significant improvement in overall survival (p<0.05).",
           "key_entities": ["Drug X", "Overall Survival", "Phase 3"]
         },
         ...
       ]

    DATA SNIPPETS:
    """
    
    full_content = prompt_header + "\n\n" + "\n---\n".join(samples)
    
    # 3. Save to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    print(f"Prompt generated at: {OUTPUT_FILE}")
    print("Copy the content of this file and paste it into Gemini Pro / ChatGPT.")

if __name__ == "__main__":
    prepare_prompt()