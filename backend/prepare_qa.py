import json
import random
from pathlib import Path

# Config
INPUT_FILES = [
    Path("d:/Projects/OncoGraphRAG/data/abstracts/Breast_Cancer.jsonl"),
    # Add other files if needed
]
# Change from a single file to an output directory
OUTPUT_DIR = Path("d:/Projects/OncoGraphRAG/data/evaluation/prompts")
SAMPLES_PER_FILE = 1  

def prepare_prompt():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect random samples from your data
    for file_path in INPUT_FILES:
        if not file_path.exists(): continue
        
        # Get the prefix for the filename, e.g., 'breast_cancer'
        cancer_type = file_path.stem.lower()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Shuffle and pick random lines to get diverse trials
            selected_lines = random.sample(lines, min(len(lines), SAMPLES_PER_FILE))
            
            for line in selected_lines:
                data = json.loads(line)
                
                # Extract the ID number for the filename (assuming format 'PMID:12345')
                raw_id = data.get('id', 'unknown')
                safe_id = raw_id.replace('PMID:', '')
                
                # Format: ID | Title | Phase | Full Abstract
                snippet = f"ID: {raw_id}\nTitle: {data.get('title')}\nPhase: {data.get('trial_phase_metadata')}\nAbstract: {data.get('text')}"

                # 2. Create the Mega-Prompt for Gemini/GPT
                prompt_header = """
    I am building a comprehensive evaluation dataset for an Oncology RAG system based on Clinical Trial Abstracts.

    TASK:
    Based on the provided abstract snippet, generate a diverse set of SIMPLE questions.
    Generate at least 5 questions corresponding to the different types below.

    CRITICAL REQUIREMENT 1: **Simplicity**
    Keep the questions extremely simple, short, and direct. Ask about high-level, basic facts such as the name of the drug, the type of cancer, the trial phase, or the overall primary outcome. Avoid overly complex medical jargon, long multi-part clauses, or asking for highly specific secondary endpoints.

    CRITICAL REQUIREMENT 2: **Self-Contained Questions**
    Every question MUST be standalone. Do NOT use relative references like "in this study", "this trial", or "the patients". 
    Instead, explicitly mention the specific diseases, drugs, or trial names so the question can be understood WITHOUT knowing the source ID.
    (e.g., Instead of "What was the most common side effect?", use "What was the most common side effect of Pembrolizumab for melanoma?")

    1. **Question Types**:
       - **Simple 1-Hop**: Extremely basic questions answerable by a single obvious fact (e.g., "What phase is the Pembrolizumab trial for melanoma?").
       - **Simple MCQ**: A brief question with 4 short options (A, B, C, D) and one correct answer.
       - **Simple True/False**: A straightforward statement that is clearly either supported or contradicted by the text.
       - **Simple Open-Ended**: Short questions about the main finding regarding the specific treatment.

    2. **Format**:
       Return the output as a valid JSON list of objects.
       
       Structure:
       [
         {
           "source_id": "PMID:12345",
           "type": "Simple 1-Hop",
           "question": "What is the main drug tested for advanced melanoma?",
           "options": null,
           "ground_truth_answer": "Pembrolizumab.",
           "key_entities": ["Pembrolizumab", "melanoma"]
         }
       ]

    3. **Constraint**: Ensure the "ground_truth_answer" is strictly derived from the provided abstract snippet.

    DATA SNIPPET:
    """
                
                full_content = prompt_header + "\n\n" + snippet
                
                # 3. Save to individual file
                output_file = OUTPUT_DIR / f"{cancer_type}_q_{safe_id}.txt"
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(full_content)
                
                print(f"Prompt generated at: {output_file}")

if __name__ == "__main__":
    prepare_prompt()