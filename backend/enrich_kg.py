import os
import uuid
import ast
import logging

from openai import OpenAI
from SPARQLWrapper import SPARQLWrapper, POST
from dotenv import load_dotenv
from prompts import system_prompt, generate_user_prompt
from disambiguation.disambiguation import sanitize_entity_name


load_dotenv()

# Configure client for LM Studio
client = OpenAI(
    base_url=os.getenv('LMSTUDIO_URL', 'http://localhost:1234/v1'),
    api_key="lm-studio"
)

GRAPHDB_URL = os.getenv('GRAPHDB_URL', 'http://localhost:7200')
ONT_BASE = "http://www.semanticweb.org/lkshmn/ontologies/2026/1/OncoGraphRAG#"

# Default model identifier for LM Studio
LMSTUDIO_MODEL = os.getenv('LMSTUDIO_MODEL', 'openai/gpt-oss-20b') 

def generate_relations(text):
    user_prompt = generate_user_prompt(text)

    try:
        chat_response = client.chat.completions.create(
            model=LMSTUDIO_MODEL,
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ]
        )
        return chat_response
    except Exception as e:
        logging.error(f"Error generating relations with LM Studio: {e}")
        return None


def convert_relations(response):
    if not response or not response.choices:
        return []
        
    relations = []
    content = response.choices[0].message.content.strip()
    
    # Handle potential markdown code blocks from local LLMs
    if "```" in content:
        content = content.replace("```json", "").replace("```python", "").replace("```", "")
    
    # Split by newline and parse dictionaries
    # Local LLMs might output a list or individual lines
    try:
        # Try parsing as a full JSON list first
        relations = ast.literal_eval(content)
        if isinstance(relations, list):
            return relations
    except:
        pass

    # Fallback to line-by-line parsing
    relation_strings = content.split('\n')
    relations = []
    for relation_string in relation_strings:
        relation_string = relation_string.strip()
        if relation_string.startswith('{') and relation_string.endswith('}'):
            try:
                relation_dict = ast.literal_eval(relation_string)
                relations.append(relation_dict)
            except:
                continue

    return relations


def create_sparql_query(relations):
    prefixes = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ont: <{ONT_BASE}>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    """

    query = prefixes + f"INSERT DATA {{ GRAPH <{ONT_BASE}> {{\n"

    for relation in relations:
        # Sanitize entity names and types
        subject_name = sanitize_entity_name(relation.get("entity1_name", ""))
        object_name = sanitize_entity_name(relation.get("entity2_name", ""))
        relation_type = sanitize_entity_name(relation.get("relation_type", ""))
        entity1_type = sanitize_entity_name(relation.get("entity1_type", "")).upper()
        entity2_type = sanitize_entity_name(relation.get("entity2_type", "")).upper()

        if not (subject_name and object_name and relation_type):
            continue

        # Generate a unique URI for the relation
        relation_id = f"REL_{uuid.uuid4()}"
        relation_uri = f"ont:{relation_id}"

        # Define URIs
        subject_uri = f"ont:{subject_name}"
        predicate_uri = f"ont:{relation_type}"
        object_uri = f"ont:{object_name}"

        # Start constructing the triple
        triple = (
            f"{relation_uri} rdf:type ont:RELATION ;\n"
            f"  ont:relation_subject {subject_uri} ;\n"
            f"  ont:relation_predicate {predicate_uri} ;\n"
            f"  ont:relation_object {object_uri} ;\n"
        )

        # Check if 'pub_id' exists and is not empty
        pub_id = relation.get("pub_id")
        if pub_id:
            publication_uri = f"ont:PUB_{sanitize_entity_name(pub_id)}"
            triple += f"  prov:wasDerivedFrom {publication_uri} .\n"
            # Create the publication instance
            query += f"{publication_uri} rdf:type ont:PUBLICATION .\n"
        else:
            triple += " ."

        # Add type declarations for entities
        query += triple + "\n"
        query += (
            f"{subject_uri} rdf:type ont:{entity1_type} .\n"
            f"{object_uri} rdf:type ont:{entity2_type} .\n"
        )

    query += "}}"

    return query


def return_relations(text):
    gpt_response = generate_relations(text)
    relations = convert_relations(gpt_response)

    return relations


def add_relations_to_kg(relations, repo_id):
    if not relations:
        return
    query = create_sparql_query(relations)
    sparql = SPARQLWrapper(f"{GRAPHDB_URL}/repositories/{repo_id}/statements")
    sparql.setMethod(POST)
    sparql.setQuery(query)
    sparql.query()