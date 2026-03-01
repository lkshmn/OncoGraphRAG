import json
import os
from pathlib import Path
from bulk_relations import extract_relations

# Define the list of files to process
INPUT_FILES = [
    Path(r"d:\Projects\OncoGraphRAG\data\abstracts\Lung_Cancer.jsonl"),
    Path(r"d:\Projects\OncoGraphRAG\data\abstracts\Colorectal_Cancer.jsonl") 
]

OUTPUT_DIR = Path(r"d:\Projects\OncoGraphRAG\data\extracted relations")

def load_first_n_abstracts(file_path, n=500):
    """
    Reads the first n lines from a JSONL file and formats them for extraction.
    Returns a generator or iterator to be memory efficient if needed, 
    but here we return a list to keep it simple with existing logic.
    """
    abstracts = []
    
    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        return []

    print(f"Reading from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            
            try:
                record = json.loads(line.strip())
                
                entry = {
                    "id": record.get("id", "unknown"),
                    "text": record.get("text", ""),
                    "trial_phase_metadata": record.get("trial_phase_metadata", "Unknown")
                }
                
                if entry["text"]:
                    abstracts.append(entry)
                    
            except json.JSONDecodeError:
                print(f"Skipping line {i}: Invalid JSON.")
                continue

    return abstracts

def process_file(input_file, limit):
    """
    Processes a single input file: loads abstracts, extracts relations 
    one by one, and saves them incrementally.
    """
    input_name = input_file.stem 
    output_filename = f"{input_name}_relations_{limit}.jsonl"
    output_file_path = OUTPUT_DIR / output_filename
    
    # 1. Load Data
    abstracts = load_first_n_abstracts(input_file, n=limit)
    
    if not abstracts:
        print(f"No valid abstracts found in {input_file.name}.")
        return
    
    total_abstracts = len(abstracts)
    print(f"Loaded {total_abstracts} abstracts from {input_file.name}. Starting extraction...")
    
    # Ensure output directory exists (create if needed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear file if it exists (fresh start for this run)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        pass 
    
    total_relations = 0
    
    # 2. Process & Save Incrementally
    for i, abstract in enumerate(abstracts):
        pub_id = abstract.get('id', 'unknown')
        print(f"Processing {i+1}/{total_abstracts} (ID: {pub_id}) ... ", end="", flush=True)

        # Pass verbose=False to suppress internal "1/1" messages
        batch_relations = extract_relations([abstract], model="lmstudio", verbose=False)
        
        if batch_relations:
            with open(output_file_path, 'a', encoding='utf-8') as f:
                for relation in batch_relations:
                    f.write(json.dumps(relation, ensure_ascii=False) + '\n')
            
            count = len(batch_relations)
            total_relations += count
            print(f"Saved {count} relations.")
        else:
             print("No relations found.")

    print(f"Finished {input_file.name}. Total relations extracted: {total_relations}")
    print("-" * 30)

def main():
    # Settings
    LIMIT = 500 # Number of abstracts to process per file
    
    for input_file in INPUT_FILES:
        print(f"--- Processing Queue: {input_file.name} ---")
        process_file(input_file, LIMIT)

if __name__ == "__main__":
    main()