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
graph = OntotextGraphDBGraph(
    query_endpoint=f"{GRAPHDB_BASE_URL}/repositories/{REPO_ID}",
    query_ontology=f"CONSTRUCT {{?s ?p ?o}} FROM <{ONTOLOGY_BASE_URI}> WHERE {{?s ?p ?o}}",
)

# Initialize Local LLM
llm = ChatOpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key=LMSTUDIO_API_KEY,
    model=LMSTUDIO_MODEL,
    temperature=0, 
)

# Custom Prompt to force fetching Citations (PMIDs)
# FIX: Added strict formatting instructions to preventing Markdown wrapping
GRAPH_QA_PROMPT_TEMPLATE = """
Task: Generate a SPARQL query to answer the user's question.
You are querying an Oncology Knowledge Graph.
The graph structure uses Reification for relations:
- Relations are nodes of type <{ontology_uri}RELATION>.
- A relation node links a Subject, Predicate, and Object via:
  <{ontology_uri}relation_subject>, <{ontology_uri}relation_predicate>, <{ontology_uri}relation_object>.
- Provenance/Citation is linked to the RELATION node via <http://www.w3.org/ns/prov#wasDerivedFrom>.
- The object of 'wasDerivedFrom' is a Publication URI (e.g., ont:PUB_123456). Extract the numeric ID part for the citation.

The ontology schema is:
{schema}

Question: {prompt}

Guideline:
- Always select ?citation (the PMID or source ID) alongside the answer fields.
- If multiple publications support a fact, group them using GROUP_CONCAT.

CRITICAL: 
- DO NOT use Markdown backticks (```).
- DO NOT include explanations.
- Output ONLY the raw SPARQL query starting with PREFIX.

SPARQL Query:
"""

PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], 
    partial_variables={"ontology_uri": ONTOLOGY_BASE_URI},
    template=GRAPH_QA_PROMPT_TEMPLATE,
)

# Subclass to Clean SPARQL if the LLM ignores instructions (Local LLMs often do)
class RobustGraphQAChain(OntotextGraphDBQAChain):
    def _call(self, inputs):
        # Intercept mostly to ensure we can debug or patch if needed, 
        # but LangChain doesn't make it easy to hook just the SPARQL generation output 
        # without overriding `_call` or `_get_sparql`.
        # For simplicity, we stick to the prompt engineering fix first.
        # However, checking if existing `ontotext_graphdb` chain has a hook.
        return super()._call(inputs)

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
        enhanced_input = (
            f"{user_input} "
            "Important: specify the answer and list the supporting PubMed IDs (PMIDs). "
            "Format citations as markdown links: [PMID](https://pubmed.ncbi.nlm.nih.gov/PMID/)."
        )
        
        # The chain takes 'query' as the input key for the generic Invoke method,
        # and blindly maps it to the 'prompt' variable inside the template.
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

# User input
user_query = st.chat_input("Type your medical question here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.spinner("Querying Knowledge Graph..."):
        response_text = get_response(user_query)
        
    st.session_state.chat_history.append(AIMessage(content=response_text))

# Render Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
             st.write(message.content) 
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
