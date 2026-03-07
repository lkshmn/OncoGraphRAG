import os
import io
import sys
from datetime import datetime
from typing import Any
import streamlit as st
from dotenv import load_dotenv
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

st.set_page_config(page_title="Chat with OncoGraphRAG", page_icon="🧬")
load_dotenv()

# Configuration
GRAPHDB_BASE_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200")
REPO_ID = os.getenv("REPO_ID", "oncograph")

# Ontology Namespace
ONTOLOGY_BASE_URI = "http://www.semanticweb.org/lkshmn/ontologies/2026/1/OncoGraphRAG#"

# LM Studio Configuration
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")

# Logging setup
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

if "log_file" not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.log_file = os.path.join(LOG_DIR, f"chat_log_{timestamp}.txt")


class QueryLogHandler(BaseCallbackHandler):
    """Captures LLM prompts, LLM responses, and chain I/O for logging."""

    def __init__(self):
        self.logs: list[str] = []
        self._llm_call_count = 0

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self._llm_call_count += 1
        label = "SPARQL Generation" if self._llm_call_count == 1 else "Answer Generation"
        self.logs.append(f"\n--- LLM Call #{self._llm_call_count} ({label}) ---")
        self.logs.append(f"[PROMPT TO LLM]:\n{prompts[0]}\n")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        text = response.generations[0][0].text if response.generations else str(response)
        label = "Generated SPARQL" if self._llm_call_count == 1 else "LLM Answer"
        self.logs.append(f"[{label}]:\n{text}\n")

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs: Any) -> None:
        if serialized is None:
            return
        name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        if name == "OntotextGraphDBQAChain":
            self.logs.append(f"\n> Entering {name} chain...\n")

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        if not isinstance(outputs, dict):
            return
        for key in ("context", "sparql_query_result", "results"):
            if key in outputs:
                self.logs.append(f"[SPARQL QUERY RESULTS ({key})]:\n{outputs[key]}\n")

    def get_log(self) -> str:
        return "\n".join(self.logs)


# initialize GraphDB connection
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

# Initialize Local LLM
llm = ChatOpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key=LMSTUDIO_API_KEY,
    model=LMSTUDIO_MODEL,
    temperature=0,
)

# SPARQL QueryGeneration Prompt
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

PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"],
    partial_variables={"ontology_uri": ONTOLOGY_BASE_URI},
    template=GRAPH_QA_PROMPT_TEMPLATE,
)

# Create QA Chain
chain = OntotextGraphDBQAChain.from_llm(
    llm,
    graph=graph,
    sparql_generation_prompt=PROMPT,
    verbose=True,
    allow_dangerous_requests=True,
)


def write_to_log(content: str):
    with open(st.session_state.log_file, "a", encoding="utf-8") as f:
        f.write(content)


def get_response(user_input: str) -> str:
    try:
        enhanced_input = (
            f"{user_input} "
            "Important: Answer ONLY based on the graph query results. "
            "For each finding, report the EXACT values returned in the query results: "
            "the reliabilityScore column value (integer 0-100), the trialPhase column value, "
            "and the efficacyEvidence column value. Do NOT calculate, override, or invent these values — "
            "use the exact data from the SPARQL results. "
            "List supporting PMIDs as [PMID](url)."
        )

        # Log user query and enhanced prompt
        log_header = (
            f"=========================================\n"
            f"TIME: {datetime.now()}\n"
            f"USER QUERY: {user_input}\n"
            f"ENHANCED PROMPT: {enhanced_input}\n"
            f"=========================================\n\n"
        )
        write_to_log(log_header)

        # Use callback handler to capture prompts, SPARQL results, and final LLM prompt
        log_handler = QueryLogHandler()

        # Also capture verbose stdout output
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            response = chain.invoke(
                {chain.input_key: enhanced_input},
                config={"callbacks": [log_handler]},
            )
        finally:
            sys.stdout = old_stdout

        # Write captured stdout (verbose chain trace)
        chain_output = captured.getvalue()
        if chain_output.strip():
            write_to_log(chain_output + "\n")

        # Write detailed callback logs (prompts, SPARQL results, final LLM prompt)
        callback_log = log_handler.get_log()
        if callback_log.strip():
            write_to_log(callback_log + "\n\n")

        # Extract final response
        if isinstance(response, dict):
            result = response.get(chain.output_key) or response.get("result") or str(response)
        else:
            result = str(response)

        # Log the final response
        write_to_log(f"[FINAL RESPONSE]:\n{result}\n=========================================\n\n")

        return result
    except Exception as e:
        error_msg = str(e)
        write_to_log(f"[ERROR]: {error_msg}\n=========================================\n\n")
        if "Expected {SelectQuery" in error_msg:
            return "The model generated an invalid query format (Markdown). Please try again."
        return f"Error processing query: {error_msg}"


st.title("Chat with Oncology Knowledge Graph")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="Hello! I am an oncology research assistant powered by a Knowledge Graph. Ask me about cancer types, treatments, or clinical findings."
        ),
    ]

# Render Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_query = st.chat_input("Type your medical question here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        with st.spinner("Querying Knowledge Graph..."):
            response_text = get_response(user_query)
            st.write(response_text)

    st.session_state.chat_history.append(AIMessage(content=response_text))