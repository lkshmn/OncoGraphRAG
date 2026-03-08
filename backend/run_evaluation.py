"""
run_evaluation.py
Queries the OncoGraphRAG chatbot with ground truth questions and saves
results for evaluation (Faithfulness, Answer Relevancy, Semantic Similarity, Context Recall).

Requires: GraphDB running on localhost:7200, LM Studio running on localhost:1234
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

# --- Configuration ---
GRAPHDB_BASE_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200")
REPO_ID = os.getenv("REPO_ID", "oncograph")
ONTOLOGY_BASE_URI = "http://www.semanticweb.org/lkshmn/ontologies/2026/1/OncoGraphRAG#"
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")

GROUND_TRUTH_DIR = Path("d:/Projects/OncoGraphRAG/data/evaluation/ground_truth")
OUTPUT_DIR = Path("d:/Projects/OncoGraphRAG/data/evaluation/results")
MAX_QUESTIONS = 1  # Set to None to run all questions


class EvalCallbackHandler(BaseCallbackHandler):
    """Captures SPARQL query, SPARQL results, and contexts for evaluation."""

    def __init__(self):
        self.sparql_query = ""
        self.sparql_results = ""
        self._llm_call_count = 0

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self._llm_call_count += 1
        # The 2nd LLM call's prompt contains SPARQL results as context
        if self._llm_call_count == 2:
            self.sparql_results = prompts[0]

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        text = response.generations[0][0].text if response.generations else str(response)
        if self._llm_call_count == 1:
            self.sparql_query = text.strip()

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        if not isinstance(outputs, dict):
            return
        for key in ("context", "sparql_query_result", "results"):
            if key in outputs:
                self.sparql_results = str(outputs[key])


# --- SPARQL prompt (same as streamlit_app.py) ---
GRAPH_QA_PROMPT_TEMPLATE = """
Task: Generate a SPARQL query to answer the user's question.
You are querying an Oncology Knowledge Graph.

**PREFIXES (REQUIRED):**
PREFIX ont: <{ontology_uri}>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

**DATA STRUCTURE:**
Relations are stored as reified nodes of type `ont:RELATION` with:
- `ont:relation_subject` — the source entity
- `ont:relation_predicate` — one of: `ont:cause`, `ont:treat`, `ont:present`, `ont:diagnose`, `ont:aggravate`, `ont:prevent`, `ont:improve`, `ont:affect`
- `ont:relation_object` — the target entity

Entity types: DISEASE, SYMPTOM, TREATMENT, RISK_FACTOR, TEST, GENE, BIOMARKER, COMPLICATION, PROGNOSIS, COMORBIDITY, PROGRESSION, BODY_PART

Any entity type can be subject or object. Do NOT assume treatments are always subjects or diseases are always objects.

**RELATION DEFINITIONS (use these to choose the correct predicate):**
- `ont:treat` — Subject is used to manage or cure Object
- `ont:cause` — Subject results in or is the reason for Object
- `ont:present` — Subject exhibits or shows signs of Object
- `ont:diagnose` — Subject identifies the presence of Object
- `ont:aggravate` — Subject worsens or intensifies Object
- `ont:prevent` — Subject stops or hinders the occurrence of Object
- `ont:improve` — Subject enhances or makes Object better
- `ont:affect` — Subject influences or has an impact on Object

**INSTRUCTIONS:**
1. **Choosing the predicate**: Map the user's question to the correct relation predicate. "treats" → `ont:treat`, "causes" → `ont:cause`, "diagnoses" → `ont:diagnose`, "improves" → `ont:improve`, "aggravates/worsens" → `ont:aggravate`, "prevents" → `ont:prevent`, "presents/symptoms" → `ont:present`, "affects" → `ont:affect`.
2. **When the relation type is unknown**: If the user asks a general question like "What do we know about X?" without specifying a relation type, do NOT filter by `ont:relation_predicate`. Instead, select `?predicateLabel` and retrieve all relations involving the entity.
3. **Filtering**: ALWAYS use case-insensitive regex: `FILTER(REGEX(STR(?label), "search.*term", "i"))`. Replace spaces with `.*`.
4. **Labels**: Always join `rdfs:label` to get readable names.
5. **Citations/PMIDs**: Publications are linked via `prov:wasDerivedFrom`. The PMID is in the URI itself (e.g., `ont:pmid12345678`). Publications do NOT have `rdfs:label`. Select the citation URI directly as `?citation`.
6. **Metadata columns**: Use EXACTLY these variable names:
   - `?reliabilityScore` — integer (0-100) from `ont:reliabilityScore`
   - `?trialPhase` — string from `ont:trialPhase`
   - `?efficacyEvidence` — text from `ont:efficacyEvidence`
   These are DIFFERENT fields. Do NOT confuse them.

**EXAMPLE 1: Specific relation type — "What does X treat?"**
*Question:* "What are the various treatments for breast cancer?"
*Query:*
SELECT ?subjectLabel ?reliabilityScore ?trialPhase ?efficacyEvidence ?citation WHERE {{
    ?rel a ont:RELATION .
    ?rel ont:relation_predicate ont:treat .
    ?rel ont:relation_subject ?subject .
    ?rel ont:relation_object ?object .

    ?object rdfs:label ?objectLabel .
    FILTER(REGEX(STR(?objectLabel), "breast.*cancer", "i")) .

    OPTIONAL {{ ?subject rdfs:label ?subjectLabel }} .
    OPTIONAL {{ ?rel ont:reliabilityScore ?reliabilityScore }} .
    OPTIONAL {{ ?rel ont:trialPhase ?trialPhase }} .
    OPTIONAL {{ ?rel ont:efficacyEvidence ?efficacyEvidence }} .
    OPTIONAL {{ ?rel prov:wasDerivedFrom ?citation }} .
}} ORDER BY DESC(?reliabilityScore) LIMIT 10

**EXAMPLE 2: Specific relation type — "What does X improve?"**
*Question:* "What improves sentinel node biopsy?"
*Query:*
SELECT ?subjectLabel ?reliabilityScore ?trialPhase ?efficacyEvidence ?citation WHERE {{
    ?rel a ont:RELATION .
    ?rel ont:relation_predicate ont:improve .
    ?rel ont:relation_subject ?subject .
    ?rel ont:relation_object ?object .

    ?object rdfs:label ?objectLabel .
    FILTER(REGEX(STR(?objectLabel), "sentinel.*node.*biopsy", "i")) .

    OPTIONAL {{ ?subject rdfs:label ?subjectLabel }} .
    OPTIONAL {{ ?rel ont:reliabilityScore ?reliabilityScore }} .
    OPTIONAL {{ ?rel ont:trialPhase ?trialPhase }} .
    OPTIONAL {{ ?rel ont:efficacyEvidence ?efficacyEvidence }} .
    OPTIONAL {{ ?rel prov:wasDerivedFrom ?citation }} .
}} ORDER BY DESC(?reliabilityScore) LIMIT 10

**EXAMPLE 3: Specific relation type — "What aggravates X?"**
*Question:* "What risk factors aggravate sentinel node biopsy?"
*Query:*
SELECT ?subjectLabel ?reliabilityScore ?trialPhase ?efficacyEvidence ?citation WHERE {{
    ?rel a ont:RELATION .
    ?rel ont:relation_predicate ont:aggravate .
    ?rel ont:relation_subject ?subject .
    ?rel ont:relation_object ?object .

    ?object rdfs:label ?objectLabel .
    FILTER(REGEX(STR(?objectLabel), "sentinel.*node.*biopsy", "i")) .

    OPTIONAL {{ ?subject rdfs:label ?subjectLabel }} .
    OPTIONAL {{ ?rel ont:reliabilityScore ?reliabilityScore }} .
    OPTIONAL {{ ?rel ont:trialPhase ?trialPhase }} .
    OPTIONAL {{ ?rel ont:efficacyEvidence ?efficacyEvidence }} .
    OPTIONAL {{ ?rel prov:wasDerivedFrom ?citation }} .
}} ORDER BY DESC(?reliabilityScore) LIMIT 10

**EXAMPLE 4: General query — all relations for an entity**
*Question:* "What do we know about preoperative lymphoscintigraphy?"
*Query:*
SELECT ?subjectLabel ?predicateLabel ?objectLabel ?reliabilityScore ?trialPhase ?efficacyEvidence ?citation WHERE {{
    ?rel a ont:RELATION .
    ?rel ont:relation_predicate ?predicate .
    ?rel ont:relation_subject ?subject .
    ?rel ont:relation_object ?object .

    {{
        ?subject rdfs:label ?subjectLabel .
        FILTER(REGEX(STR(?subjectLabel), "preoperative.*lymphoscintigraphy", "i")) .
        OPTIONAL {{ ?object rdfs:label ?objectLabel }} .
    }} UNION {{
        ?object rdfs:label ?objectLabel .
        FILTER(REGEX(STR(?objectLabel), "preoperative.*lymphoscintigraphy", "i")) .
        OPTIONAL {{ ?subject rdfs:label ?subjectLabel }} .
    }}

    OPTIONAL {{ ?predicate rdfs:label ?predicateLabel }} .
    OPTIONAL {{ ?rel ont:reliabilityScore ?reliabilityScore }} .
    OPTIONAL {{ ?rel ont:trialPhase ?trialPhase }} .
    OPTIONAL {{ ?rel ont:efficacyEvidence ?efficacyEvidence }} .
    OPTIONAL {{ ?rel prov:wasDerivedFrom ?citation }} .
}} ORDER BY DESC(?reliabilityScore) LIMIT 10

**EXAMPLE 5: Specific relation — "What does X diagnose?"**
*Question:* "What does preoperative lymphoscintigraphy diagnose?"
*Query:*
SELECT ?objectLabel ?reliabilityScore ?trialPhase ?efficacyEvidence ?citation WHERE {{
    ?rel a ont:RELATION .
    ?rel ont:relation_predicate ont:diagnose .
    ?rel ont:relation_subject ?subject .
    ?rel ont:relation_object ?object .

    ?subject rdfs:label ?subjectLabel .
    FILTER(REGEX(STR(?subjectLabel), "preoperative.*lymphoscintigraphy", "i")) .

    OPTIONAL {{ ?object rdfs:label ?objectLabel }} .
    OPTIONAL {{ ?rel ont:reliabilityScore ?reliabilityScore }} .
    OPTIONAL {{ ?rel ont:trialPhase ?trialPhase }} .
    OPTIONAL {{ ?rel ont:efficacyEvidence ?efficacyEvidence }} .
    OPTIONAL {{ ?rel prov:wasDerivedFrom ?citation }} .
}} ORDER BY DESC(?reliabilityScore) LIMIT 10

The ontology schema is:
{schema}

Question: {prompt}

CRITICAL:
- Output ONLY the raw SPARQL query starting with PREFIX.
- Do NOT use Markdown backticks.
- Choose the correct `ont:relation_predicate` based on the user's question. Do NOT default to `ont:treat` for every question.
- For general questions about an entity, use UNION to search both subject and object positions.
- For citations, select `?citation` directly. Do NOT use rdfs:label on citations.
- Use EXACTLY these variable names: `?reliabilityScore`, `?trialPhase`, `?efficacyEvidence`. Do NOT rename or shorten them.

SPARQL Query:
"""


def setup_chain():
    """Initialize GraphDB connection and QA chain (same config as streamlit app)."""
    graph = OntotextGraphDBGraph(
        query_endpoint=f"{GRAPHDB_BASE_URL}/repositories/{REPO_ID}",
        query_ontology="""
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            CONSTRUCT {
                ?s rdf:type ?type .
                ?s rdfs:label ?label .
            } 
            WHERE {
                ?s rdf:type ?type .
                FILTER (?type IN (owl:Class, owl:ObjectProperty, owl:DatatypeProperty))
                OPTIONAL { ?s rdfs:label ?label }
            }
        """,
    )

    llm = ChatOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
        model=LMSTUDIO_MODEL,
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["schema", "prompt"],
        partial_variables={"ontology_uri": ONTOLOGY_BASE_URI},
        template=GRAPH_QA_PROMPT_TEMPLATE,
    )

    chain = OntotextGraphDBQAChain.from_llm(
        llm,
        graph=graph,
        sparql_generation_prompt=prompt,
        verbose=True,
        allow_dangerous_requests=True,
    )

    return chain


def query_chatbot(chain, question: str) -> dict:
    """Send a question to the chatbot and capture answer + retrieved context."""
    enhanced_input = (
        f"{question} "
        "Important: Answer ONLY based on the graph query results. "
        "For each finding, report the EXACT values returned in the query results: "
        "the reliabilityScore column value (integer 0-100), the trialPhase column value, "
        "and the efficacyEvidence column value. Do NOT calculate, override, or invent these values — "
        "use the exact data from the SPARQL results. "
        "List supporting PMIDs as [PMID](url)."
    )

    handler = EvalCallbackHandler()

    try:
        response = chain.invoke(
            {chain.input_key: enhanced_input},
            config={"callbacks": [handler]},
        )

        if isinstance(response, dict):
            answer = response.get(chain.output_key) or response.get("result") or str(response)
        else:
            answer = str(response)

        return {
            "answer": answer,
            "generated_sparql": handler.sparql_query,
            "retrieved_context": handler.sparql_results,
            "error": None,
        }
    except Exception as e:
        return {
            "answer": "",
            "generated_sparql": handler.sparql_query,
            "retrieved_context": handler.sparql_results,
            "error": str(e),
        }


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


def run_evaluation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"eval_results_{timestamp}.jsonl"

    print("Loading ground truth files...")
    questions = load_ground_truth_files()
    print(f"Loaded {len(questions)} questions from {len(set(q['_gt_file'] for q in questions))} files\n")

    if not questions:
        print("No questions found. Make sure ground truth files are populated.")
        return
    
    # After loading questions, add:
    if MAX_QUESTIONS:
        questions = questions[:MAX_QUESTIONS]
        print(f"Testing with first {MAX_QUESTIONS} question(s)\n")
    
    print("Initializing QA chain...")
    chain = setup_chain()
    print("Chain ready.\n")

    results = []
    errors = 0

    for i, q in enumerate(questions, 1):
        question = q["question"]
        print(f"[{i}/{len(questions)}] {question[:80]}...")

        result = query_chatbot(chain, question)

        record = {
            "question_id": i,
            "source_id": q.get("source_id", ""),
            "question_type": q.get("type", ""),
            "question": question,
            "options": q.get("options"),
            # Chatbot output
            "answer": result["answer"],
            "retrieved_context": result["retrieved_context"],
            "generated_sparql": result["generated_sparql"],
            # Ground truth
            "ground_truth_answer": q.get("ground_truth_answer", ""),
            "ground_truth_context": q.get("ground_truth_context", ""),
            "key_entities": q.get("key_entities", []),
            # Metadata
            "gt_file": q.get("_gt_file", ""),
            "error": result["error"],
        }

        results.append(record)

        # Write incrementally so partial results are saved if interrupted
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        status = "ERROR" if result["error"] else "OK"
        if result["error"]:
            errors += 1
        print(f"  → {status}\n")

        # Small delay to avoid overwhelming LM Studio
        time.sleep(1)



    print(f"\n{'='*50}")
    print(f"Evaluation complete!")
    print(f"  Total: {len(questions)} | Success: {len(questions) - errors} | Errors: {errors}")
    print(f"  Results: {output_file}")


if __name__ == "__main__":
    run_evaluation()