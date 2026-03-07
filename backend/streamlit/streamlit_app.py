import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

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

# Custom Prompt with CORRECTED Directionality
GRAPH_QA_PROMPT_TEMPLATE = """
Task: Generate a SPARQL query to answer the user's question.
You are querying an Oncology Knowledge Graph.

**PREFIXES (REQUIRED):**
PREFIX ont: <{ontology_uri}>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

**DATA STRUCTURE:**
1. **Treatments** are usually the SUBJECT (Source).
2. **Diseases** are usually the OBJECT (Target).
   - Triple Pattern: `<Treatment> <treats> <Disease>`
   - Query Pattern: `?treatment ont:treat ?disease` (via Reification node)

**INSTRUCTIONS:**
1. **Finding Entities**: 
   - Look for `?rel ont:relation_object ?disease`.
   - Look for `?rel ont:relation_subject ?treatment`.
2. **Filtering**: ALWAYS use case-insensitive substring matching using `REGEX(STR(?label), "search.*term", "i")`. Replace spaces in the user's search term with `.*` to handle complex multi-word entities (e.g., "Breast.*Cancer", "anthracyclines.*plus.*vinorelbine").
3. **Labels**: Always join `rdfs:label` to get readable names.

**EXAMPLE 1: User asks for treatments for a specific disease**
*Question:* "What are the various treatments for breast cancer?"
*Query:*
SELECT ?treatmentLabel ?score ?phase ?efficacy ?citation WHERE {{
    ?rel a ont:RELATION .
    ?rel ont:relation_predicate ont:treat .
    
    # Direction: Treatment (Subject) -> treats -> Disease (Object)
    ?rel ont:relation_subject ?treatment .
    ?rel ont:relation_object ?disease .

    # Filter the Disease (Object) using broad regex and STR() conversion
    ?disease rdfs:label ?diseaseLabel .
    FILTER(REGEX(STR(?diseaseLabel), "breast.*cancer", "i")) . 

    # Get Treatment Label
    OPTIONAL {{ ?treatment rdfs:label ?treatmentLabel }} .

    # Metadata
    OPTIONAL {{ ?rel ont:reliabilityScore ?score }} .
    OPTIONAL {{ ?rel ont:trialPhase ?phase }} .
    OPTIONAL {{ ?rel ont:efficacyEvidence ?efficacy }} .
    OPTIONAL {{ ?rel prov:wasDerivedFrom ?citation }} .
}} ORDER BY DESC(?score) LIMIT 10

**EXAMPLE 2: User asks what a specific treatment is used for**
*Question:* "What diseases are treated by vinorelbine?"
*Query:*
SELECT ?diseaseLabel ?score ?phase ?efficacy ?citation WHERE {{
    ?rel a ont:RELATION .
    ?rel ont:relation_predicate ont:treat .
    
    # Direction: Treatment (Subject) -> treats -> Disease (Object)
    ?rel ont:relation_subject ?treatment .
    ?rel ont:relation_object ?disease .

    # Filter the Treatment (Subject) using broad regex and STR() conversion
    ?treatment rdfs:label ?treatmentLabel .
    FILTER(REGEX(STR(?treatmentLabel), "vinorelbine", "i")) . 

    # Get Disease Label
    OPTIONAL {{ ?disease rdfs:label ?diseaseLabel }} .

    # Metadata
    OPTIONAL {{ ?rel ont:reliabilityScore ?score }} .
    OPTIONAL {{ ?rel ont:trialPhase ?phase }} .
    OPTIONAL {{ ?rel ont:efficacyEvidence ?efficacy }} .
    OPTIONAL {{ ?rel prov:wasDerivedFrom ?citation }} .
}} ORDER BY DESC(?score) LIMIT 10

The ontology schema is:
{schema}

Question: {prompt}

CRITICAL: 
- Output ONLY the raw SPARQL query starting with PREFIX.
- Do NOT use Markdown backticks.

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

def get_response(user_input: str) -> str:
    try:
        enhanced_input = (
            f"{user_input} "
            "Important: Answer based on the graph results. "
            "For each finding, mention the Reliability Score (0-100), Trial Phase, and specific Efficacy found in the data. "
            "List supporting PMIDs as [PMID](url)."
        )

        response = chain.invoke({chain.input_key: enhanced_input})

        if isinstance(response, dict):
            return response.get(chain.output_key) or response.get("result") or str(response)
        return str(response)
    except Exception as e:
        if "Expected {SelectQuery" in str(e):
            return "The model generated an invalid query format (Markdown). Please try again."
        return f"Error processing query: {str(e)}"

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