import os
import re
import ast
import logging
import uuid
import requests
import json
from dotenv import load_dotenv
from SPARQLWrapper import SPARQLWrapper, POST
from prompts import generate_user_prompt, system_prompt
from disambiguation.disambiguation import sanitize_entity_name, refine_relations

load_dotenv()

GRAPHDB_URL = os.getenv('GRAPHDB_URL', 'http://localhost:7200')
LMSTUDIO_URL = os.getenv('LMSTUDIO_URL', 'http://localhost:1234/v1/chat/completions')
LMSTUDIO_MODEL = os.getenv('LMSTUDIO_MODEL', 'local-model') 

ONT_BASE = "http://www.semanticweb.org/lkshmn/ontologies/2026/1/OncoGraphRAG#"

def calculate_confidence_score(trial_phase, efficacy_text):
    """
    Calculates a confidence score (0-100) based on trial phase and efficacy wording.
    Expects efficacy_text to start with tags like "Positive:", "Negative:", etc.
    """
    score = 50 # Base score for Unknown/Other
    
    # Highest priority trials first
    phase_map = {
        'randomized controlled trial': 95,
        'phase iv': 92,
        'phase 4': 92,
        'phase iii': 90,
        'phase 3': 90,
        'phase ii': 75,
        'phase 2': 75,
        'phase i': 60,
        'phase 1': 60
    }
    
    # normalize inputs
    t_phase = str(trial_phase).lower()
    eff_text = str(efficacy_text).strip().lower()
    
    # 1. Base score from Phase
    for key, val in phase_map.items():
        if key in t_phase:
            score = val
            break
            
    # 2. Adjust based on precise Efficacy Tags (from prompts.py)
    
    if eff_text.startswith("negative"):
        score -= 20
    elif eff_text.startswith("neutral") or eff_text.startswith("inconclusive"):
        score -= 10
    elif eff_text.startswith("not stated") or eff_text == "not stated":
        score -= 20
    elif eff_text.startswith("positive"):
        score += 5
    
    # Fallback to keyword search if LLM missed the tag format
    elif any(x in eff_text for x in ["no significant", "did not improve", "ineffective", "no benefit", "failed"]):
        score -= 20
    elif any(x in eff_text for x in ["significant", "p<0.0", "increased survival", "effective", "improved"]):
        score += 5
        
    return min(max(score, 0), 100)  # Clamp between 0 and 100

def generate_relations_lmstudio(text):
    """Generate relations using LM Studio API (local LLM)"""
    user_prompt = generate_user_prompt(text)
    
    try:
        response = requests.post(
            LMSTUDIO_URL,
            json={
                "model": LMSTUDIO_MODEL, 
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                "temperature": 0,
                "max_tokens": 2500,
            },
            timeout=180, 
        )
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logging.error(f"LM Studio API call failed: {e}")
        return None

def is_valid_relation(data):
    valid_entity_types = {'disease', 'symptom', 'treatment', 'risk_factor', 'test', 'gene', 'biomarker', 'complication',
                          'prognosis', 'comorbidity', 'progression', 'body_part'}
    valid_relation_types = {'cause', 'treat', 'present', 'diagnose', 'aggravate', 'prevent', 'improve', 'affect'}

    return (
            data.get('relation_type') in valid_relation_types and
            data.get('entity1_type') in valid_entity_types and
            data.get('entity2_type') in valid_entity_types and
            isinstance(data.get('entity1_name'), str) and
            isinstance(data.get('entity2_name'), str)
    )

def validate_output(output, metadata_phase="Unknown"):
    # Updated pattern to capture optional efficacy
    pattern = r"\{'relation_type': '.*?', 'entity1_type': '.*?', 'entity1_name': '.*?', 'entity2_type': '.*?', 'entity2_name': '.*?', 'efficacy': '.*?'\}"
    
    matches = re.findall(pattern, output)
    dicts = []
    
    # Fallback to loose json line parsing if regex misses multiline dictionaries
    if not matches and "{" in output:
        for line in output.split('\n'):
             line = line.strip()
             if line.startswith("{") and line.endswith("}"):
                 matches.append(line)

    for match in matches:
        try:
            data = ast.literal_eval(match)
            if is_valid_relation(data):
                # Inject metadata passed from the download script
                data['trial_phase'] = metadata_phase
                
                # Calculate Score
                score = calculate_confidence_score(metadata_phase, data.get('efficacy', ''))
                data['confidence_score'] = score
                
                dicts.append(data)
            else:
                logging.warning(f"Invalid data fields in: {data}")
        except (SyntaxError, ValueError) as e:
            logging.warning(f"Failed to parse match: {match}. Error: {e}")
    return dicts

def extract_relations(abstracts, model="lmstudio", verbose=True):
    """
    Main entry point for extracting relations. 
    Accepts a list of abstract dictionaries.
    verbose: If False, suppresses print statements (useful for item-by-item processing loops).
    """
    relations = []
    total = len(abstracts)
    
    if verbose:
        print(f"Starting extraction for {total} abstracts using LM Studio...")
    
    for i, abstract in enumerate(abstracts):
        if verbose:
            print(f"Processing {i + 1}/{total} (ID: {abstract.get('id', 'unknown')})")
        
        # Pull trial phase from metadata if available, else default
        phase_meta = abstract.get('trial_phase_metadata', 'Unknown')
        
        response = generate_relations_lmstudio(abstract["text"])
        
        if response and response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            
            # Pass metadata to validator to inject into relations
            matches = validate_output(content, metadata_phase=phase_meta)
            
            pub_id = {'pub_id': abstract.get('id')}
            for match in matches:
                match.update(pub_id)
            
            relations.extend(matches)
        else:
            if verbose:
                print(f"Skipping index {i}: Failed to get valid response.")

    return relations

def create_sparql_queries_for_bulk_import(relations, batch_size=200):
    prefixes = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ont: <{ONT_BASE}>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    """
    sparql_queries = []

    # Split the relations into batches
    for i in range(0, len(relations), batch_size):
        batch_relations = relations[i:i+batch_size]
        
        # Build INSERT DATA query
        query_body = ""
        for relation in batch_relations:
            # Sanitize inputs
            subject_name = sanitize_entity_name(relation["entity1_name"])
            object_name = sanitize_entity_name(relation["entity2_name"])
            relation_type = sanitize_entity_name(relation["relation_type"])
            entity1_type = sanitize_entity_name(relation["entity1_type"]).upper()
            entity2_type = sanitize_entity_name(relation["entity2_type"]).upper()
            
            # Metadata
            score = relation.get('confidence_score', 50)
            phase = relation.get('trial_phase', 'Unknown').replace("'", "").replace('"', '')
            efficacy = relation.get('efficacy', 'Not stated').replace("'", "").replace('"', '')

            # Generate URIs
            relation_id = f"REL_{uuid.uuid4()}"
            relation_uri = f"ont:{relation_id}"
            subject_uri = f"ont:{subject_name}"
            predicate_uri = f"ont:{relation_type}"
            object_uri = f"ont:{object_name}"

            # Relation Node Triple with Data Properties
            triple = (
                f"{relation_uri} rdf:type ont:RELATION ;\n"
                f"  ont:relation_subject {subject_uri} ;\n"
                f"  ont:relation_predicate {predicate_uri} ;\n"
                f"  ont:relation_object {object_uri} ;\n"
                f"  ont:confidenceScore \"{score}\"^^xsd:integer ;\n"
                f"  ont:trialPhase \"{phase}\"^^xsd:string ;\n"
                f"  ont:efficacyEvidence \"{efficacy}\"^^xsd:string "
            )

            # Provenance
            pub_id_val = relation.get("pub_id")
            if pub_id_val:
                publication_uri = f"ont:{sanitize_entity_name(pub_id_val)}"
                triple += f" ;\n  prov:wasDerivedFrom {publication_uri} ."
                # Define Publication Type
                query_body += f"{publication_uri} rdf:type ont:PUBLICATION .\n"
            else:
                triple += " ."

            query_body += triple + "\n"
            
            # Entity Type Triples
            query_body += f"{subject_uri} rdf:type ont:{entity1_type} .\n"
            query_body += f"{object_uri} rdf:type ont:{entity2_type} .\n"

        if query_body:
            full_query = prefixes + f"INSERT DATA {{ GRAPH <{ONT_BASE}> {{\n{query_body}\n}}}}"
            sparql_queries.append(full_query)

    return sparql_queries

def add_bulk_relations_to_kg(relations, repo_id):
    """
    Refines relations (deduplication, type resolution) and uploads to GraphDB.
    """
    if not relations:
        print("No relations to upload.")
        return

    print("Refining relations...")
    refined_relations = refine_relations(relations)
    
    print(f"Generating SPARQL queries for {len(refined_relations)} relations...")
    queries = create_sparql_queries_for_bulk_import(refined_relations)
    
    sparql = SPARQLWrapper(f"{GRAPHDB_URL}/repositories/{repo_id}/statements")
    sparql.setMethod(POST)
    
    print(f"Uploading in {len(queries)} batches...")
    for idx, query in enumerate(queries):
        try:
            sparql.setQuery(query)
            sparql.query()
            print(f"Batch {idx + 1} uploaded successfully.")
        except Exception as e:
            logging.error(f"Failed to upload batch {idx + 1}: {e}")
