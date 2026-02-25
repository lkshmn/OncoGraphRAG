import os
import streamlit as st
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Configuration
GRAPHDB_BASE_URL = os.getenv("GRAPHDB_URL", "http://localhost:7200")
REPO_ID = os.getenv("REPO_ID", "oncograph")
GRAPHDB_USERNAME = os.getenv("GRAPHDB_USERNAME", "admin")
GRAPHDB_PASSWORD = os.getenv("GRAPHDB_PASSWORD", "root")

# Ontology Namespace
ONTOLOGY_BASE_URI = "http://www.semanticweb.org/lkshmn/ontologies/2026/1/OncoGraphRAG#"

# LM Studio Configuration
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = "lm-studio"  # Any string works for local LM Studio
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model") 

# Initialize GraphDB Connection
graph = OntotextGraphDBGraph(
    query_endpoint=f"{GRAPHDB_BASE_URL}/repositories/{REPO_ID}",
    query_ontology=f"CONSTRUCT {{?s ?p ?o}} FROM <{ONTOLOGY_BASE_URI}> WHERE {{?s ?p ?o}}",
)

# Initialize Local LLM via LM Studio
llm = ChatOpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key=LMSTUDIO_API_KEY,
    model=LMSTUDIO_MODEL,
    temperature=0, 
)

# Create QA Chain
chain = OntotextGraphDBQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True, 
)

def get_response(user_input):
    try:
        # invoke returns a dictionary, usually with the key "result" or defined output_key
        response = chain.invoke({chain.input_key: user_input})
        return response[chain.output_key]
    except Exception as e:
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
    # Add user message immediately
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Get response
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
