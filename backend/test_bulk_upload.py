import json
import os
from pathlib import Path
from bulk_relations import add_bulk_relations_to_kg, refine_relations, create_sparql_queries_for_bulk_import

# Path to the file containing relations extracted by test_lmstudio_pubmed10.py
INPUT_RELATIONS = Path(r"d:\Projects\OncoGraphRAG\data\pubmed_oncology_relations_10.json")

# Define your GraphDB repository ID here
REPO_ID = "oncograph" 

def main():
    if not INPUT_RELATIONS.exists():
        print(f"Error: Input file {INPUT_RELATIONS} not found.")
        print("Please run 'test_lmstudio_pubmed10.py' first to generate the relations.")
        return

    print(f"Loading relations from {INPUT_RELATIONS}...")
    try:
        relations = json.loads(INPUT_RELATIONS.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    if not relations:
        print("No relations found in the file.")
        return

    print(f"Loaded {len(relations)} raw relations.")

    # Step 1: Test Refinement locally (to see if it works without error)
    print("\n--- Testing Refinement Logic ---")
    try:
        refined = refine_relations(relations)
        print(f"Refinement successful. Reduced {len(relations)} raw relations to {len(refined)} unique/resolved relations.")
    except Exception as e:
        print(f"Error during refinement: {e}")
        return

    # Step 2: Test SPARQL Query Generation locally
    print("\n--- Testing SPARQL Query Generation ---")
    try:
        queries = create_sparql_queries_for_bulk_import(refined, batch_size=5) # Small batch for testing
        print(f"Generated {len(queries)} SPARQL batches.")
        if queries:
            print("Preview of first query batch (first 500 chars):")
            print(queries[0][:500] + "...")
    except Exception as e:
        print(f"Error during SPARQL generation: {e}")
        return

    # Step 3: Run the full upload function (Integration Test)
    # Note: This requires GraphDB to be running and the repository to exist.
    print(f"\n--- Attempting Upload to GraphDB (Repo: {REPO_ID}) ---")
    confirm = input(f"Are you sure you want to upload to {REPO_ID}? (y/n): ")
    if confirm.lower() == 'y':
        try:
            add_bulk_relations_to_kg(relations, REPO_ID)
            print("Full upload process completed.")
        except Exception as e:
            print(f"Error during upload process: {e}")
            print("Ensure GraphDB is running and the repository ID matches.")
    else:
        print("Upload skipped.")

if __name__ == "__main__":
    main()