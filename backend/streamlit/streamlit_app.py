import os
import streamlit as st
import re
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

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
# FIX: Removed the specific named GRAPH clause. Now querying the default graph so it finds schema regardless of context.
graph = OntotextGraphDBGraph(
    query_endpoint=f"{GRAPHDB_BASE_URL}/repositories/{REPO_ID}",
    query_ontology="""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        CONSTRUCT {
            ?s rdf:type ?type .
            ?s rdfs:subClassOf ?super .
            ?s rdfs:domain ?domain .
            ?s rdfs:range ?range .
            ?s rdfs:label ?label .
        } 
        WHERE {
            ?s rdf:type ?type .
            FILTER (?type IN (owl:Class, owl:ObjectProperty, owl:DatatypeProperty))
            OPTIONAL { ?s rdfs:subClassOf ?super }
            OPTIONAL { ?s rdfs:domain ?domain }
            OPTIONAL { ?s rdfs:range ?range }
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

# Custom Prompt to force fetching Citations (PMIDs) and Confidence Scores
GRAPH_QA_PROMPT_TEMPLATE = """
Task: Generate a SPARQL query to answer the user's question with evidence confidence.
You are querying an Oncology Knowledge Graph.
The graph structure uses Reification for relations:
- Relations are nodes of type <{ontology_uri}RELATION>.
- A relation node has:
  * <{ontology_uri}relation_subject> (The source entity)
  * <{ontology_uri}relation_predicate> (The type of relation, e.g., <{ontology_uri}treat>)
  * <{ontology_uri}relation_object> (The target entity)
  * <{ontology_uri}confidenceScore> (Integer 0-100)
  * <{ontology_uri}trialPhase> (String)
  * <{ontology_uri}efficacyEvidence> (String)
- Provenance is linked via <http://www.w3.org/ns/prov#wasDerivedFrom>.

**IMPORTANT DATA FORMAT NOTES:**
1. Entities do NOT have rdfs:label. You must filter by the URI string.
   - INCORRECT: ?s rdfs:label "Trastuzumab"
   - CORRECT: FILTER(REGEX(STR(?s), "Trastuzumab", "i"))
2. Relation predicates are objects of the triplet.
   - Example: ?rel <{ontology_uri}relation_predicate> <{ontology_uri}treat> .

The ontology schema is:
{schema}

Question: {prompt}

Guideline:
- Select ?subject ?object ?score ?phase ?efficacy ?citation
- Use FILTER(REGEX(STR(?var), "SearchTerm", "i")) to find entities.
- Order results by DESC(?score).
- Limit to 10 results.

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

# Create QA Chain with the custom prompt
chain = OntotextGraphDBQAChain.from_llm(
    llm,
    graph=graph,
    sparql_generation_prompt=PROMPT, 
    verbose=True,
    allow_dangerous_requests=True, 
)

def get_response(user_input):
    try:
        # Enhanced input description for the answer generation step (which happens inside the chain after SPARQL execution)
        enhanced_input = (
            f"{user_input} "
            "Important: Answer based on the graph results. "
            "For each finding, mention the Confidence Score (0-100), Trial Phase, and specific Efficacy found in the data. "
            "List supporting PMIDs as [PMID](url)."
        )
        
        # The chain takes the input and runs the SPARQL generation -> Execution -> Answer generation
        response = chain.invoke({chain.input_key: enhanced_input})
        
        return response[chain.output_key]
    except Exception as e:
        # Fallback error handling if parsing failed due to markdown
        if "Expected {SelectQuery" in str(e):
             return "The model generated an invalid query format (Markdown). Please try again."
        return f"Error processing query: {str(e)}"

# App config
st.set_page_config(page_title="Chat with OncoGraphRAG", page_icon="🧬")
st.title("Chat with Oncology Knowledge Graph")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am an oncology assistant powered by your local Knowledge Graph. Ask me about cancer types, treatments, or clinical findings."),
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
if user_query is not None and user_query != "":
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)
    
    # Get and add AI response
    with st.chat_message("AI"):
        with st.spinner("Querying Knowledge Graph..."):
            response_text = get_response(user_query)
            st.write(response_text)
            
    st.session_state.chat_history.append(AIMessage(content=response_text))
