import json
import os
from pathlib import Path
from bulk_relations import add_bulk_relations_to_kg, refine_relations, create_sparql_queries_for_bulk_import

# Path to the file containing relations extracted by relation_extraction.py
INPUT_RELATIONS = Path(r"d:\Projects\OncoGraphRAG\data\extracted relations\Breast_Cancer_relations_500.jsonl")

# Define your GraphDB repository ID here
REPO_ID = "oncograph" 

def load_relations_from_jsonl(file_path):
    """Reads relations from a JSONL file."""
    relations = []
    if not file_path.exists():
        print(f"Error: Input file {file_path} not found.")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                relations.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                
    return relations

def main():
    if not INPUT_RELATIONS.exists():
        print(f"Error: Input file {INPUT_RELATIONS} not found.")
        print("Please run 'relation_extraction.py' first to generate the relations.")
        return

    print(f"Loading relations from {INPUT_RELATIONS}...")
    relations = load_relations_from_jsonl(INPUT_RELATIONS)

    if not relations:
        print("No relations found in the file.")
        return

    print(f"Loaded {len(relations)} raw relations.")
    
    # Check if new metadata keys exist in the first relation (sanity check)
    first_rel = relations[0]
    if 'confidence_score' in first_rel:
        print(f"Sample Metadata found -> Score: {first_rel.get('confidence_score')}, "
              f"Phase: {first_rel.get('trial_phase')}, Efficacy: {first_rel.get('efficacy')}")
    else:
        print("Warning: Confidence scores or trial metadata missing from input. Scores will default to 50.")

    # Step 1: Test Refinement locally
    print("\n--- Testing Refinement Logic ---")
    try:
        refined = refine_relations(relations)
        print(f"Refinement successful. Processing {len(refined)} unique/resolved relations.")
    except Exception as e:
        print(f"Error during refinement: {e}")
        return

    # Step 2: Test SPARQL Query Generation locally
    print("\n--- Testing SPARQL Query Generation (Metadata Enabled) ---")
    try:
        # Note: refine_relations might return fewer items than input
        queries = create_sparql_queries_for_bulk_import(refined, batch_size=5)
        print(f"Generated {len(queries)} SPARQL batches.")
        if queries:
            print("Preview of first query batch (first 500 chars):")
            print(queries[0][:500] + "...")
    except Exception as e:
        print(f"Error during SPARQL generation: {e}")
        return

    # Step 3: Run the full upload function
    print(f"\n--- Attempting Upload to GraphDB (Repo: {REPO_ID}) ---")
    confirm = input(f"Are you sure you want to upload to {REPO_ID}? (y/n): ")
    if confirm.lower() == 'y':
        try:
            # Re-using the logic from bulk_relations to handle final upload
            add_bulk_relations_to_kg(relations, REPO_ID)
            print("Full upload process completed.")
        except Exception as e:
            print(f"Error during upload process: {e}")
            print("Ensure GraphDB is running and the repository ID matches.")
    else:
        print("Upload skipped.")

if __name__ == "__main__":
    main()